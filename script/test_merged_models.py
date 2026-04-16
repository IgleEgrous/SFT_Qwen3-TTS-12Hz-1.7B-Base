"""
test_merged_models.py — 测试所有合并好的 LoRA 模型（custom_voice 模式，不使用 ref_audio）

用法：
    python script/test_merged_models.py
"""

import os
import glob
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 自动发现所有 checkpoint-epoch-N-merged 目录
outputs_dir = "outputs/merged_models"
pattern = os.path.join(outputs_dir, "checkpoint-epoch-*-merged")
model_dirs = sorted(glob.glob(pattern), key=lambda x: int(x.split("-")[-2]))

print(f"发现 {len(model_dirs)} 个合并模型：")
for d in model_dirs:
    print(f"  {os.path.basename(d)}")

# 要合成的测试文本（来自 test_epoch2.py）
test_text = "この魔女の証であるブローチを付け、灰色の髪を靡かせ、その美しさと才能の輝きに太陽さえも思わず目を細めてしまうほどの美女は、誰でしょうか。そう、私です！"
language = "Japanese"

for model_path in model_dirs:
    name = os.path.basename(model_path)
    epoch = name.split("-")[-2]
    print(f"\n{'='*60}")
    print(f"测试模型: {name}")

    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    wavs, sr = model.generate_custom_voice(
        text=test_text,
        language=language,
        speaker="elaina",
    )

    out_file = f"output_epoch{epoch}.wav"
    sf.write(out_file, wavs[0], sr)
    print(f"  ✅ 已生成: {out_file}")

    # 释放显存
    del model
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("全部测试完成！")
