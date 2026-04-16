"""
tts_server.py — Qwen3-TTS FastAPI 服务

用法：
    python script/tts_server.py

API：
    POST /tts
    {
        "text": "合成文本",
        "epoch": "32",
        "language": "Japanese"
    }
    → 返回 audio/wav
"""

import os
import glob
import torch
import soundfile as sf
from io import BytesIO
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel
from fastapi.middleware.cors import CORSMiddleware

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs", "merged_models")

# 自动发现所有 epoch 模型
MODEL_PATHS = {}
ckpt_dirs = glob.glob(os.path.join(OUTPUTS_DIR, "checkpoint-epoch-*-merged"))
for d in ckpt_dirs:
    epoch = os.path.basename(d).split("-")[-2]
    MODEL_PATHS[epoch] = d

MODEL_PATHS = dict(sorted(MODEL_PATHS.items(), key=lambda x: int(x[0])))
logger.info(f"可用模型: {list(MODEL_PATHS.keys())}")

# 当前加载的模型（LRU 缓存）
current_model = None
current_epoch = None

app = FastAPI(title="Qwen3-TTS Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(epoch: str):
    """加载指定 epoch 的模型到 GPU"""
    global current_model, current_epoch

    if current_epoch == epoch and current_model is not None:
        logger.info(f"模型已是 epoch {epoch}，复用")
        return current_model

    logger.info(f"切换模型: epoch {epoch}")
    if current_model is not None:
        del current_model
        torch.cuda.empty_cache()

    model_path = MODEL_PATHS[str(epoch)]
    logger.info(f"加载: {model_path}")

    current_model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    current_epoch = epoch
    logger.info(f"模型 epoch {epoch} 加载完成")
    return current_model


class TTSRequest(BaseModel):
    text: str
    epoch: str = "32"
    language: str = "Japanese"


@app.post("/tts")
def tts(req: TTSRequest):
    """合成 TTS 音频"""
    if req.epoch not in MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"无效 epoch: {req.epoch}，可用: {list(MODEL_PATHS.keys())}"
        )

    model = load_model(req.epoch)

    logger.info(f"合成: epoch={req.epoch}, lang={req.language}, text={req.text[:50]}...")
    wavs, sr = model.generate_custom_voice(
        text=req.text,
        language=req.language,
        speaker="elaina",
    )

    # 写入 BytesIO 返回
    buf = BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=epoch{req.epoch}.wav"
        }
    )


@app.get("/models")
def list_models():
    """返回所有可用模型"""
    return {
        "epochs": list(MODEL_PATHS.keys()),
        "loaded": current_epoch,
    }


@app.get("/")
def index():
    """WebUI 主页"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "tts_webui.html"))


@app.get("/health")
def health():
    return {"status": "ok", "loaded_epoch": current_epoch}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
