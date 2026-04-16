# AI README — SFT_Qwen3-TTS-12Hz-1.7B-Base

> 本文件记录项目相关的所有修改和操作，由 AI 助手维护。

---

## 项目概述

基于 Qwen3-TTS-12Hz-1.7B-Base 的 LoRA 微调项目。目标是通过 SFT 让模型学会伊蕾娜（Elaina）的音色，推理时不提供参考音频。

---

## 文件结构

```
SFT_Qwen3-TTS-12Hz-1.7B-Base/
├── script/
│   ├── sft_12hz_lora.py              # LoRA SFT 训练脚本
│   ├── merge_all_checkpoints.py       # 批量合并脚本
│   ├── test_epoch2.py                 # 单模型快速测试
│   ├── test_merged_models.py          # 多模型横向对比测试
│   ├── tts_server.py                  # FastAPI TTS 服务
│   └── tts_webui.html                 # WebUI 界面
├── models/
│   └── Qwen3-TTS-12Hz-1.7B-Base/     # Base 模型
├── outputs/
│   ├── train_with_codes.jsonl         # 训练数据
│   ├── checkpoint-epoch-{N}/          # 各 epoch checkpoint
│   └── merged_models/                 # 合并后的完整模型
│       └── checkpoint-epoch-{N}-merged/
└── datasets/
    └── audio/datasets--yeeko--Elaina_WanderingWitch_audio_JA/  # 原始数据
```

---

## 核心问题与解决方案

### 问题：合并后模型音色不像

**根本原因：**

之前的 LoRA 训练只更新了 attention 层的权重（lora_A/lora_B），`codec_embedding.weight[3000]`（speaker slot）始终是 frozen 的初始值。LoRA 学的是"怎么处理 speaker embedding"，而不是"记住 speaker embedding 是什么"。

推理时 `generate_custom_voice` 查 `codec_embedding[3000]`，但这个位置从未被训练过，所以音色完全不对。

**解决方案：**

让 `codec_embedding[spk_id]` 可训练。每个 step 用 `speaker_encoder` 提取参考音频的 embedding，同步到 `codec_embedding[spk_id]`（梯度可回传）。训练完成后 slot 里存的就是"记住的音色"，推理时不需参考音频。

---

## 已修改的脚本

### 1. `script/sft_12hz_lora.py`

#### 改动 1：路径解析（避免相对路径问题）

所有路径改为基于脚本文件位置解析，不依赖 cwd：

```python
script_path = pathlib.Path(__file__).resolve()
project_root = script_path.parent.parent

def resolve_path(path):
    p = pathlib.Path(path)
    if p.is_absolute():
        return str(p)
    return str((project_root / p).resolve())
```

#### 改动 2：Speaker Embedding 可训练

```python
# codec_embedding[spk_id] 初始化为 0 并设为可训练
nn.init.zeros_(qwen3tts.model.talker.model.codec_embedding.weight[spk_id])
qwen3tts.model.talker.model.codec_embedding.weight[spk_id].requires_grad_(True)
```

#### 改动 3：训练循环中去掉 `.detach()`

```python
spk_emb = model.speaker_encoder(ref_mels...)  # 梯度可回传

# 每个 step 同步到 codec_embedding[spk_id]
with torch.no_grad():
    model.talker.model.codec_embedding.weight[spk_id].copy_(spk_emb.mean(0))
```

#### 改动 4：Checkpoint 保存时额外保存 speaker embedding

```python
trained_spk_emb = unwrapped_model.talker.model.codec_embedding.weight[TRAIN_SPK_ID].clone()
np.save(os.path.join(tmp_dir, "trained_speaker_embedding.npy"),
        trained_spk_emb.float().cpu().numpy())
```

---

### 2. `script/merge_all_checkpoints.py`

#### 改动 1：从 `trained_speaker_embedding.npy` 注入 speaker embedding

```python
spk_emb_path = os.path.join(checkpoint_path, "trained_speaker_embedding.npy")
if os.path.exists(spk_emb_path):
    trained_emb = torch.from_numpy(np.load(spk_emb_path)).float()
    with torch.no_grad():
        base_model.model.talker.model.codec_embedding.weight[train_spk_id] = trained_emb.clone()
```

#### 改动 2：动态读取 spk_id

从 checkpoint 的 `config.json` 读取 `talker_config.spk_id`，不再写死 3000。

#### 改动 3：支持批量合并多个 epoch

```bash
python script/merge_all_checkpoints.py --epochs 4 8 16 24 32 --speaker_name elaina
```

---

### 3. `script/test_merged_models.py`（新增）

自动发现所有合并模型，横向对比测试：

```bash
python script/test_merged_models.py
```

输出：`output_epoch4.wav`, `output_epoch8.wav`, `output_epoch16.wav`, `output_epoch24.wav`, `output_epoch32.wav`

---

### 4. `script/tts_server.py`（新增）

FastAPI TTS 服务，支持多 epoch 模型切换：

```bash
python script/tts_server.py
# 访问 http://localhost:8000 查看 WebUI
```

接口：
- `POST /tts` — 合成音频，返回 WAV
- `GET /models` — 列出所有可用 epoch
- `GET /health` — 健康检查

---

## 训练操作流程

### 标准流程（每两个 epoch 验证一次）

```bash
# 训练 epoch 1-2
python script/sft_12hz_lora.py --batch_size 2 --num_epochs 2 --speaker_name elaina

# 合并 epoch 2
python script/merge_all_checkpoints.py --epochs 2 --speaker_name elaina

# 测试
python script/test_merged_models.py

# 从 epoch 2 继续训练到 epoch 4
python script/sft_12hz_lora.py --num_epochs 4 --save_epochs 2 4 --resume --resume_from_epoch 2

# 合并并测试
python script/merge_all_checkpoints.py --epochs 4 --speaker_name elaina
```

### 一口气训练到 32 epoch

```bash
python script/sft_12hz_lora.py --num_epochs 32 --batch_size 2 --speaker_name elaina

# 批量合并所有 epoch
python script/merge_all_checkpoints.py --epochs 4 8 16 24 32 --speaker_name elaina
```

---

## 当前状态

- [x] 训练脚本已修改（speaker embedding 可训练）
- [x] 合并脚本已修改（从 npy 注入 embedding）
- [x] 训练完成：epoch 2, 4, 8, 16, 24, 32
- [x] 所有 epoch 已合并
- [x] TTS WebUI 已完成（`tts_webui.html`）
- [x] TTS FastAPI 服务已部署（`tts_server.py`）
- [ ] 客观评估各 epoch 效果（Speaker Similarity）
- [ ] RVC 变声器方案（替代 TTS 用于实时聊天）

---

## 历史记录

### 2026-04-16

**已完成**
- 所有 epoch（2/4/8/16/24/32）训练完成
- 批量合并脚本支持多 epoch 一次性合并
- TTS WebUI 上线（http://localhost:8000）
- FastAPI 服务支持多模型切换

**已知问题**
- TTS 生成延迟较高，不适合即时聊天场景（完整句子生成需等待）
- 中文发音带有日语语调（数据集全为日语）
- 多模型同时部署显存不足（需按需加载）

**后续方向**
1. RVC 变声器（替代 TTS）：低延迟，适合实时聊天
2. 量化部署：INT8/FP8 减少显存占用
3. 流式音频：一边生成一边播放，降低感知延迟

---

## 参考信息

- Base 模型：`models/Qwen3-TTS-12Hz-1.7B-Base/`
- codec_embedding dim：3072 × 2048（vocab_size=3072，hidden_size=2048）
- speaker_encoder 输出 dim：2048（等于 codec_hidden_size，不需要投影）
- 默认 speaker slot id：3000（`talker_config.spk_id["elaina"] = 3000`）
- 合并后模型：`outputs/merged_models/checkpoint-epoch-{N}-merged/`
- 数据量：1444 条日语语音（train_with_codes.jsonl）
