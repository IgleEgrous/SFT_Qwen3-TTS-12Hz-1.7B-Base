# SFT Qwen3-TTS-12Hz-1.7B-Base

基于 Qwen3-TTS-12Hz-1.7B-Base 的 LoRA 微调项目，使用伊蕾娜（魔女之旅/本渡枫）语音数据进行训练。

## 项目背景

零样本语音克隆方便快捷，但存在音色不稳定、跨语言带口音、情感表达不够自然等问题。微调后的模型能够：
- **音色稳定**：在各种文本上都保持一致的声音特征
- **情感丰富**：支持自然语言语气/语速指示
- **无需参考音频**：音色已固化在模型中，直接输入文本即可生成伊蕾娜音色

## 目录结构

```
SFT_Qwen3-TTS-12Hz-1.7B-Base/
├── models/
│   ├── Qwen3-TTS-12Hz-1.7B-Base/     # Base 模型权重
│   └── Qwen3-TTS-Tokenizer-12Hz/    # 专用 tokenizer
├── datasets/                          # 原始数据集
├── outputs/                          # 训练输出目录
│   ├── train_with_codes.jsonl        # tokenize 后的训练数据
│   ├── checkpoint-epoch-{N}/         # 各 epoch checkpoint
│   └── merged_models/                 # 合并后的完整模型
│       └── checkpoint-epoch-{N}-merged/
├── script/
│   ├── prepare_data.py               # 数据 tokenize 脚本
│   ├── sft_12hz_lora.py              # LoRA SFT 训练脚本
│   ├── merge_all_checkpoints.py       # 批量合并 LoRA → 完整模型
│   ├── test_merged_models.py          # 多模型横向对比测试
│   ├── tts_server.py                  # FastAPI TTS 服务
│   └── tts_webui.html                 # WebUI 界面
└── README.md
```

## 快速开始

### 1. Tokenize 数据

```bash
python script/prepare_data.py --mode local \
  --parquet_url ./datasets/audio/datasets--yeeko--Elaina_WanderingWitch_audio_JA/snapshots/xxx/train/metadata.parquet \
  --audio_col file_name --text_col transcription \
  --audio_dir ./datasets/audio \
  --ref_audio "D:/path/to/ref.wav" \
  --tokenizer_model_path "D:/Productivity/github/models/Qwen3-TTS-Tokenizer-12Hz" \
  --output_jsonl ./outputs/train_with_codes.jsonl \
  --batch_size 32 --device cuda:0
```

### 2. 训练

```bash
# 训练到 32 epoch
python script/sft_12hz_lora.py \
  --num_epochs 32 --batch_size 2 --speaker_name elaina

# 或断点续训
python script/sft_12hz_lora.py --num_epochs 32 \
  --resume --resume_from_epoch 2
```

### 3. 合并

```bash
# 批量合并多个 epoch
python script/merge_all_checkpoints.py --epochs 4 8 16 24 32 --speaker_name elaina
```

### 4. 部署 WebUI

```bash
python script/tts_server.py
# 浏览器打开 http://localhost:8000
```

---

## 完整工作流程

```
Step 1: Tokenize 数据
Step 2: LoRA SFT 训练（epoch 2/4/8/16/24/32）
Step 3: 横向对比各 epoch 效果（选最优）
Step 4: 批量合并所有 LoRA checkpoint → 完整 HF 模型
Step 5: 部署 WebUI / API 服务
```

---

## 数据集

### 伊蕾娜语音数据集

- **HF Repo**: [yeeko/Elaina_WanderingWitch_audio_JA](https://huggingface.co/datasets/yeeko/Elaina_WanderingWitch_audio_JA)
- **内容**: 1444 条伊蕾娜日语语音 + transcription
- **采样率**: 原始音频 24kHz
- **格式**: `train/metadata.parquet` + `train/` 下 1444 个音频文件

### JSONL 训练数据格式

tokenize 后输出格式（`train_with_codes.jsonl`）：

```jsonl
{"audio": "/path/to/audio.wav", "text": "其实我真的有发现...", "audio_codes": [...], "ref_audio": "/path/to/ref.wav"}
```

---

## 推理使用方法

### 方式 1：WebUI（推荐）

```bash
python script/tts_server.py
# 访问 http://localhost:8000
```

功能：打字 → 选择 epoch → 生成 → 在线播放 + 下载

### 方式 2：Python API

```python
from qwen_tts import Qwen3TTSModel
import soundfile as sf

model = Qwen3TTSModel.from_pretrained(
    "outputs/merged_models/checkpoint-epoch-32-merged",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

wavs, sr = model.generate_custom_voice(
    text="この魔女の証であるブローチを付け...",
    language="Japanese",
    speaker="elaina",
)

sf.write("output.wav", wavs[0], sr)
```

### 方式 3：HTTP API

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "文本", "epoch": "32", "language": "Japanese"}' \
  -o output.wav
```

---

## 训练参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--init_model_path` | `models/Qwen3-TTS-12Hz-1.7B-Base` | Base 模型路径 |
| `--train_jsonl` | **必填** | tokenize 后的训练数据 |
| `--output_model_path` | `outputs` | checkpoint 输出目录 |
| `--speaker_name` | `speaker_test` | 自定义说话人名称 |
| `--batch_size` | `2` | 每 GPU batch size |
| `--lr` | `2e-6` | 学习率 |
| `--num_epochs` | `3` | 训练轮数 |
| `--save_epochs` | `None` | 指定保存的 epoch 列表 |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | 2×r |
| `--resume` | `False` | 是否断点续训 |
| `--resume_from_epoch` | `None` | 从第几 epoch 继续 |

---

## 硬件需求

| 模型大小 | 最低显存 | 推荐显存 |
|---|---|---|
| 0.6B | 8GB | 16GB |
| 1.7B（LoRA） | 8GB | 12GB |
| 1.7B（全量微调） | 16GB | 24GB |

LoRA 模式下训练参数量约为全模型的 0.1%~1%，大幅降低显存需求。

---

## 技术细节

### LoRA 配置

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

### 训练配置

- **混合精度**: bfloat16
- **梯度累积**: 4 steps
- **优化器**: AdamW (weight_decay=0.01)
- **梯度裁剪**: max_norm=1.0
- **Loss**: `outputs.loss + 0.3 * sub_talker_loss`

### Speaker Embedding 固化机制

每个 step 用 `speaker_encoder` 提取参考音频 embedding，同步到 `codec_embedding[spk_id]`。训练完成后 slot 里存的就是"记住的音色"，推理时不需参考音频。

---

## 后续方向

### RVC 变声器（适合实时聊天）

TTS 方案延迟较高，适合**即时聊天**的替代方案：

1. **RVC（Retrieval-Based Voice Conversion）**：输入你的声音 → 输出伊蕾娜音色，延迟可做到 50-200ms
2. 项目地址：https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
3. 你的伊蕾娜音频数据可直接用于训练 RVC 模型

### TTS 量化部署

如需进一步优化延迟，可对模型进行 INT8/FP8 量化，减少显存占用并提升推理速度。

---

## 相关资料

- [Qwen3-TTS 官方文档](https://www.mintlify.com/QwenLM/Qwen3-TTS/advanced/fine-tuning)
- [PEFT 库](https://github.com/huggingface/peft)
- [LoRA 原文](https://arxiv.org/abs/2106.09685)
- [RVC 变声器](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [伊蕾娜语音数据集](https://huggingface.co/datasets/yeeko/Elaina_WanderingWitch_audio_JA)
