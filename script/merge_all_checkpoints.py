# coding=utf-8
"""
merge_all_checkpoints.py — 批量合并 LoRA checkpoint 到 Base 模型

将 output/ 下的所有 checkpoint-epoch-* 目录中的 LoRA adapter 合并到 Base 模型，
输出为独立的完整 HF 模型，可直接上传。

前提：checkpoint 目录包含以下文件（由 sft_12hz_lora.py 生成）：
    adapter_model.safetensors           # LoRA 权重（A、B 矩阵）
    adapter_config.json                 # PEFT adapter 配置
    trained_speaker_embedding.npy       # 训练好的 speaker embedding（slot spk_id）
    config.json                          # 含 spk_id 映射

用法：
    python script/merge_all_checkpoints.py \
        --base_model ../models/Qwen3-TTS-12Hz-1.7B-Base \
        --checkpoints_dir ../output \
        --output_parent ../output/merged_models \
        --speaker_name elaina

输出结构：
    output/merged_models/
    ├── checkpoint-epoch-2-merged/
    ├── checkpoint-epoch-4-merged/
    ├── checkpoint-epoch-8-merged/
    ├── checkpoint-epoch-16-merged/
    └── checkpoint-epoch-32-merged/
"""
import argparse
import json
import os
import shutil

import torch
import numpy as np
from peft import PeftModel
from safetensors.torch import load_file, save_file


def find_checkpoints(checkpoints_dir: str, epochs: list = None):
    """找到所有 checkpoint-epoch-* 目录（需包含 adapter_model.safetensors）
    若指定 epochs，则只返回匹配的那些 epoch。
    """
    checkpoints = []
    for name in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, name)
        adapter_file = os.path.join(path, "adapter_model.safetensors")
        if os.path.isdir(path) and name.startswith("checkpoint-epoch-") and os.path.exists(adapter_file):
            try:
                epoch = int(name.replace("checkpoint-epoch-", ""))
            except ValueError:
                continue
            if epochs is None or epoch in epochs:
                checkpoints.append((epoch, name, path))
    checkpoints.sort(key=lambda x: x[0])
    return [(name, path) for _, name, path in checkpoints]


def merge_single_checkpoint(base_model_path: str, checkpoint_path: str, output_dir: str,
                             speaker_name: str):
    """
    合并单个 checkpoint：

    1. 从 base_model_path 加载完整 base 模型
    2. 从 checkpoint 的 model.safetensors 取出 speaker embedding 注入到 slot 3000
    3. 用 PeftModel 加载 adapter + merge
    4. 保存完整合并模型
    """
    print(f"\n{'=' * 50}")
    print(f"合并: {checkpoint_path}")
    print(f"输出: {output_dir}")

    # 从 checkpoint config 中读取 spk_id
    checkpoint_config_path = os.path.join(checkpoint_path, "config.json")
    with open(checkpoint_config_path, "r", encoding="utf-8") as f:
        ckpt_config = json.load(f)
    spk_id_map = ckpt_config.get("talker_config", {}).get("spk_id", {})
    if spk_id_map:
        train_spk_id = list(spk_id_map.values())[0]  # 训练时使用的 slot id
    else:
        train_spk_id = 3000

    # 1. 加载 base 模型（完整权重）
    print("  [1/4] 加载 Base 模型...")
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    base_model = Qwen3TTSModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
    )

    # 2. 注入训练好的 speaker embedding（从 trained_speaker_embedding.npy 读取）
    spk_emb_path = os.path.join(checkpoint_path, "trained_speaker_embedding.npy")
    if os.path.exists(spk_emb_path):
        print("  [2/4] 注入训练好的 speaker embedding...")
        trained_emb = torch.from_numpy(np.load(spk_emb_path)).float()
        with torch.no_grad():
            base_model.model.talker.model.codec_embedding.weight[train_spk_id] = trained_emb.clone()
        print(f"       speaker embedding 已注入到 slot {train_spk_id}（来自 trained_speaker_embedding.npy）")
    else:
        print(f"  ⚠️  未找到 trained_speaker_embedding.npy，跳过 speaker embedding 注入")

    # 3. 加载 LoRA adapter 并 merge
    print("  [3/4] 加载 LoRA adapter 并合并...")
    peft_model = PeftModel.from_pretrained(
        base_model.model,
        checkpoint_path,
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()
    base_model.model = merged_model

    # 4. 保存完整合并模型
    print("  [4/4] 保存完整合并模型...")
    os.makedirs(output_dir, exist_ok=True)

    # 复制 Base 模型的所有文件
    for f in os.listdir(base_model_path):
        src = os.path.join(base_model_path, f)
        dst = os.path.join(output_dir, f)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # 保存合并后的权重（覆盖 model.safetensors）
    state_dict = {k: v.cpu() for k, v in base_model.model.state_dict().items()}

    # 删除 speaker_encoder（frozen，不参与推理）
    keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder.")]
    for k in keys_to_drop:
        del state_dict[k]

    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    # 更新 config.json
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["tts_model_type"] = "custom_voice"
    config["_name_or_path"] = output_dir
    talker_config = config.get("talker_config", {})
    talker_config["spk_id"] = {speaker_name: train_spk_id}
    talker_config["spk_is_dialect"] = {speaker_name: False}
    config["talker_config"] = talker_config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  ✅ 已保存: {output_dir}")

    # 清理显存
    del base_model, peft_model, merged_model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="批量合并 LoRA checkpoints 到 Base 模型")
    parser.add_argument("--base_model", type=str,
                        default="models/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Base 模型路径")
    parser.add_argument("--checkpoints_dir", type=str,
                        default="outputs",
                        help="checkpoint-epoch-* 所在目录")
    parser.add_argument("--output_parent", type=str,
                        default="./outputs/merged_models",
                        help="合并模型输出父目录")
    parser.add_argument("--speaker_name", type=str, default="elaina",
                        help="说话人名称（更新 config.json 用）")
    parser.add_argument("--suffix", type=str, default="-merged",
                        help="输出目录后缀，如 checkpoint-epoch-2-merged")
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                        help="指定合并哪些 epoch，如 --epochs 2 4 8 16 32。默认 None 表示全部")
    args = parser.parse_args()

    # 统一转绝对路径，避免相对路径被 HuggingFace Hub 当成 repo_id 处理
    # 相对于脚本文件位置解析，不依赖 cwd
    import pathlib
    script_path = pathlib.Path(__file__).resolve()
    project_root = script_path.parent.parent

    def resolve_path(path):
        p = pathlib.Path(path)
        if p.is_absolute():
            return str(p)
        return str((project_root / p).resolve())

    args.base_model = resolve_path(args.base_model)
    args.checkpoints_dir = resolve_path(args.checkpoints_dir)
    args.output_parent = resolve_path(args.output_parent)

    checkpoints = find_checkpoints(args.checkpoints_dir, epochs=args.epochs)
    if not checkpoints:
        print(f"❌ 未找到任何包含 adapter_model.safetensors 的 checkpoint:")
        print(f"   目录: {args.checkpoints_dir}")
        print(f"   请先运行 sft_12hz_lora.py 生成 checkpoint")
        return

    print(f"找到 {len(checkpoints)} 个 checkpoints:")
    for name, _ in checkpoints:
        print(f"  - {name}")

    os.makedirs(args.output_parent, exist_ok=True)

    for name, path in checkpoints:
        epoch_num = name.replace("checkpoint-epoch-", "")
        output_name = f"checkpoint-epoch-{epoch_num}{args.suffix}"
        output_dir = os.path.join(args.output_parent, output_name)

        try:
            merge_single_checkpoint(
                base_model_path=args.base_model,
                checkpoint_path=path,
                output_dir=output_dir,
                speaker_name=args.speaker_name,
            )
        except Exception as e:
            print(f"  ❌ 合并失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"✅ 全部完成！合并模型保存在: {args.output_parent}")
    print("每个目录都是完整的 HF 模型，可直接上传。")


if __name__ == "__main__":
    main()
