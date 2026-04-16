# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# LoRA-modified version of sft_12hz.py
# Based on: Qwen3-TTS/finetuning/sft_12hz.py
#
# Changes:
#   - Added PEFT LoRA integration (get_peft_model)
#   - Added LoRA-specific CLI arguments
#   - Frozen code_predictor embeddings (no gradient)
#   - Optimizer only updates LoRA parameters

import argparse
import json
import os
import shutil

import torch
import torch.nn as nn
import numpy as np
from accelerate import Accelerator
from dataset import TTSDataset
from peft import LoraConfig, TaskType, get_peft_model
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None
TRAIN_SPK_ID = None  # 训练时的 speaker slot id，由 setup 填充


def train():
    global target_speaker_embedding

    # 相对路径解析：基于脚本文件位置，不依赖 cwd
    import pathlib
    script_path = pathlib.Path(__file__).resolve()
    project_root = script_path.parent.parent

    def resolve_path(path):
        p = pathlib.Path(path)
        if p.is_absolute():
            return str(p)
        return str((project_root / p).resolve())

    parser = argparse.ArgumentParser()

    # === 官方原有参数 ===
    parser.add_argument("--init_model_path", type=str,
                        default="models/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="outputs")
    parser.add_argument("--train_jsonl", type=str, default="outputs/train_with_codes.jsonl")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)   # LoRA 建议用小 lr
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--save_epochs", type=int, nargs="+", default=None,
                        help="指定保存 checkpoint 的 epoch 列表，如 --save_epochs 2 4 8 16 32。默认 None 表示每个 epoch 都保存")

    # === LoRA 新增参数 ===
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank, 越大表达能力越强，首次建议 16")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha，通常设为 2*r")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="LoRA 目标层，逗号分隔")
    parser.add_argument("--train_code_predictor", action="store_true",
                        help="是否也训练 code_predictor embedding（默认 False，Frozen）")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处继续训练")
    parser.add_argument("--resume_from_epoch", type=int, default=None,
                        help="从第几个 epoch 继续训练（需配合 resume）")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="直接指定 checkpoint 目录路径继续训练")

    args = parser.parse_args()

    # 统一转绝对路径
    args.init_model_path = resolve_path(args.init_model_path)
    args.output_model_path = resolve_path(args.output_model_path)
    args.train_jsonl = resolve_path(args.train_jsonl)
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = resolve_path(args.resume_from_checkpoint)

    # === Accelerator (LoRA + Accelerator 兼容) ===
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16"
    )

    # === 加载模型 ===
    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # ============================================================
    # 改动点 1：Freeze code_predictor embedding（不参与训练）
    # ============================================================
    if not args.train_code_predictor:
        for i in range(1, 16):
            for param in qwen3tts.model.talker.code_predictor.get_input_embeddings()[i - 1].parameters():
                param.requires_grad = False

    # ============================================================
    # 改动点 2：应用 PEFT LoRA
    # ============================================================
    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    qwen3tts.model = get_peft_model(qwen3tts.model, lora_config)

    # ============================================================
    # 改动点 3：新增可训练的 speaker embedding（记忆音色）
    # codec_embedding[spk_id] 可训练，speaker_encoder 输出直接注入，梯度可回传
    # ============================================================
    global TRAIN_SPK_ID
    tc = qwen3tts.model.model.config.talker_config
    # talker_config 是 dataclass/对象，用 getattr 安全取值
    tc_dict = getattr(tc, 'to_dict', lambda: {})()
    if not tc_dict:
        tc_dict = {
            'hidden_size': getattr(tc, 'hidden_size', 2048),
            'spk_id': getattr(tc, 'spk_id', {}),
        }
    TRAIN_SPK_ID = tc_dict.get("spk_id", {}).get(args.speaker_name, 3000)
    spk_id = TRAIN_SPK_ID
    codec_hidden_size = tc_dict.get("hidden_size", 2048)

    # codec_embedding[spk_id] 初始化为 0 并设为可训练
    nn.init.zeros_(qwen3tts.model.talker.model.codec_embedding.weight[spk_id])
    qwen3tts.model.talker.model.codec_embedding.weight[spk_id].requires_grad_(True)

    print(f"[Speaker Embedding] speaker_name={args.speaker_name}, spk_id={spk_id}, "
          f"codec_hidden={codec_hidden_size}")

    # 打印 LoRA 参数数量（方便确认）
    trainable_params, total_params = 0, 0
    for p in qwen3tts.model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        total_params += p.numel()
    print(f"[LoRA] Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # ============================================================
    # 数据加载
    # ============================================================
    train_data = open(args.train_jsonl, encoding="utf-8").readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    # ============================================================
    # 改动点 3：Optimizer 只优化 LoRA 参数（全模型 requires_grad=False 时）
    # ============================================================
    optimizer = AdamW(
        qwen3tts.model.parameters(),   # 只有 LoRA 的 A、B 矩阵的 requires_grad=True
        lr=args.lr,
        weight_decay=0.01
    )

    # === 断点续训：加载已保存的进度（必须在 accelerator.prepare 之前） ===
    start_epoch = 0
    global_step = 0
    ckpt_path = None

    if args.resume:
        if args.resume_from_checkpoint:
            import re
            if args.resume_from_checkpoint == "latest":
                latest_link = os.path.join(args.output_model_path, "checkpoint-latest")
                if os.path.exists(latest_link):
                    # symlink 可能指向 checkpoint-latest 本身（相对路径），解析真实路径
                    ckpt_path = os.path.join(args.output_model_path, "checkpoint-latest")
                    if os.path.islink(ckpt_path):
                        ckpt_path = os.path.realpath(ckpt_path)
                    m = re.search(r'checkpoint-epoch-(\d+)', ckpt_path)
                    start_epoch = int(m.group(1)) if m else 0
                else:
                    print("[Resume] 未找到 checkpoint-latest，从头开始训练")
            else:
                m = re.search(r'checkpoint-epoch-(\d+)', args.resume_from_checkpoint)
                start_epoch = int(m.group(1)) if m else 0
                ckpt_path = args.resume_from_checkpoint
        elif args.resume_from_epoch is not None:
            start_epoch = args.resume_from_epoch
            ckpt_path = os.path.join(args.output_model_path, f"checkpoint-epoch-{start_epoch}")
        else:
            import re
            existing = []
            for d in os.listdir(args.output_model_path):
                mm = re.match(r'checkpoint-epoch-(\d+)', d)
                if mm:
                    existing.append(int(mm.group(1)))
            if existing:
                start_epoch = max(existing)
                ckpt_path = os.path.join(args.output_model_path, f"checkpoint-epoch-{start_epoch}")
            else:
                print("[Resume] 未找到已有 checkpoint，从头开始训练")

        if start_epoch > 0 and ckpt_path:
            if not os.path.exists(os.path.join(ckpt_path, "adapter_model.safetensors")):
                print(f"[Resume] 未找到 adapter_model.safetensors，从头开始训练")
                start_epoch = 0
                ckpt_path = None

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    # 在 prepared 后的 model 上注入 resume 的 LoRA 权重
    if start_epoch > 0 and ckpt_path:
        print(f"[Resume] 从 epoch {start_epoch} 继续训练，加载 LoRA: {ckpt_path}")
        from safetensors.torch import load_file
        lora_state = load_file(os.path.join(ckpt_path, "adapter_model.safetensors"))
        # clone 避免 Windows memory-mapped file I/O 问题
        lora_state = {k: v.clone() for k, v in lora_state.items() if "lora_" in k}
        model.load_state_dict(lora_state, strict=False)
        print(f"[Resume] LoRA 权重已加载 ({len(lora_state)} 个参数)")

    num_epochs = args.num_epochs
    save_epochs = set(args.save_epochs) if args.save_epochs else None  # None = 每个 epoch 都保存

    model.train()

    for epoch in range(start_epoch, num_epochs + 1):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                # Speaker embedding from encoder (梯度可回传，直接用不需投影)
                spk_emb = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                )  # (batch, 2048)

                # 训练时：更新 codec_embedding[spk_id]，取 batch 平均
                with torch.no_grad():
                    model.talker.model.codec_embedding.weight[spk_id].copy_(spk_emb.mean(0))

                # 注入到当前 batch 的 position 6（跟原来逻辑一致）
                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(
                    input_text_ids
                ) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(
                    input_codec_ids
                ) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = spk_emb

                # 初始化 input_embeddings = text_embedding + codec_embedding(0)
                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                # Sub-talker loss（code_predictor frozen 时理论上为 0，但保留兼容）
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss_val:.4f}"
                )

        # ============================================================
        # Checkpoint 保存（每个 epoch 自动保存 + 指定 epoch 额外保存）
        # ============================================================
        if accelerator.is_main_process:
            epoch_name = f"checkpoint-epoch-{epoch}"
            tmp_dir = os.path.join(args.output_model_path, f"{epoch_name}-tmp")
            final_dir = os.path.join(args.output_model_path, epoch_name)

            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            shutil.copytree(MODEL_PATH, tmp_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(tmp_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: TRAIN_SPK_ID}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)

            # 保存 LoRA 权重
            peft_state_dict = {k: v.clone() for k, v in unwrapped_model.state_dict().items() if 'lora_' in k}
            keys_to_drop = [k for k in peft_state_dict.keys() if k.startswith("speaker_encoder.")]
            for k in keys_to_drop:
                del peft_state_dict[k]

            save_file(peft_state_dict, os.path.join(tmp_dir, "adapter_model.safetensors"))

            # 保存训练好的 speaker embedding（codec_embedding[spk_id]）
            trained_spk_emb = unwrapped_model.talker.model.codec_embedding.weight[TRAIN_SPK_ID].clone()
            np.save(os.path.join(tmp_dir, "trained_speaker_embedding.npy"),
                    trained_spk_emb.float().cpu().numpy())
            print(f"[Speaker Embedding] 已保存 trained_speaker_embedding.npy (spk_id={TRAIN_SPK_ID})")

            peft_config_dict = unwrapped_model.peft_config["default"].to_dict()

            def _to_serializable(d):
                if isinstance(d, dict):
                    return {k: _to_serializable(v) for k, v in d.items()}
                elif isinstance(d, (list, tuple)):
                    return [_to_serializable(x) for x in d]
                elif isinstance(d, set):
                    return list(d)
                return d
            peft_config_dict = _to_serializable(peft_config_dict)
            with open(os.path.join(tmp_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(peft_config_dict, f, indent=2)

            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            os.rename(tmp_dir, final_dir)
            print(f"✅ Checkpoint saved: {final_dir}")

            latest_dir = os.path.join(args.output_model_path, "checkpoint-latest")
            if os.path.exists(latest_dir):
                shutil.rmtree(latest_dir)
            shutil.copytree(final_dir, latest_dir)
            print(f"🔗 checkpoint-latest → {final_dir}")


if __name__ == "__main__":
    train()
