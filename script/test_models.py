"""
test_models.py — smoke test for Qwen3-TTS Base model

Usage:
    python test_models.py
"""

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

ref_audio = "./datasets/audio/datasets--yeeko--Elaina_WanderingWitch_audio_JA/snapshots/4d1dfd8b112bbb084073c4825f3fc8aa602c1368/train/vocal_广播剧魔女之旅 DRAMA CD p01 Volumen 8 Capítulo 1 BV1yP4y1h7Bz_p1.wav_10.wav_0047253440_0047434560.wav"
ref_text = "学校行くのは嫌だけど 私みたいな人間は一日行かなかっただけで クラスの皆から存在を忘れられてしまうんだよ"

# voice clone
wavs, sr = model.generate_voice_clone(
    text="こんばんは、こっちはボッチです。現実世界の人たちは 誰も私なんか興味ないと思ってた。",
    language="Japanese",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)
print("Done! Saved to output_voice_clone.wav")
