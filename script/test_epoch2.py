"""
test_epoch2.py — 快速测试合并后的 epoch2 模型

用法：
    python script/test_epoch2.py
"""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_PATH = "outputs/merged_models/checkpoint-epoch-2-merged"

print(f"加载模型: {MODEL_PATH}")
model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

test_text = "この魔女の証であるブローチを付け、灰色の髪を靡かせて、その美しさと才能の輝きに太陽さえも思わず目を細めてしまうほどの美女は、誰でしょうか。そう、私です！"

print(f"合成音频: {test_text}")
wavs, sr = model.generate_custom_voice(
    text=test_text,
    language="Japanese",
    speaker="elaina",
)

out_file = "output_epoch2_test.wav"
sf.write(out_file, wavs[0], sr)
print(f"✅ 已保存: {out_file}")
