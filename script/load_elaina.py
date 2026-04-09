"""
加载伊蕾娜语音数据集

HF_REPO_ID   = yeeko/Elaina_WanderingWitch_audio_JA
音频路径     = train/{filename}
元数据       = train/metadata.parquet
本地缓存     = datasets/audio/（项目根目录下的 datasets/audio/）

Usage:
    python script/load_elaina.py              # 仅预览 metadata
    python script/load_elaina.py --download   # 预下载全部音频
"""
import os
import argparse
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

# ============ 配置 ============
HF_REPO_ID = "yeeko/Elaina_WanderingWitch_audio_JA"
LOCAL_AUDIO_DIR = Path(__file__).parent.parent / "datasets" / "audio"


def get_audio_url(filename: str) -> str:
    """HF 直链"""
    return f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/train/{filename}"


def get_local_audio_path(filename: str) -> str:
    """获取本地音频路径（自动下载）"""
    remote_path = f"train/{filename}"
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=remote_path,
        repo_type="dataset",
        cache_dir=str(LOCAL_AUDIO_DIR),
    )


def load_metadata():
    """加载 metadata"""
    parquet_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/train/metadata.parquet"
    df = pd.read_parquet(parquet_url)
    print(f"✅ metadata: {len(df)} 条 | 列名: {df.columns.tolist()}")
    return df


def preview(df: pd.DataFrame, n: int = 3):
    """预览前 n 条"""
    print(f"\n前 {n} 条:")
    for i, row in df.head(n).iterrows():
        print(f"  [{i}] {row['file_name']} -> {row['transcription']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="预下载全部音频到 datasets/audio/")
    parser.add_argument("--n", type=int, default=3, help="预览条数")
    args = parser.parse_args()

    df = load_metadata()
    preview(df, args.n)

    if args.download:
        print(f"\n📥 下载全部音频到: {LOCAL_AUDIO_DIR}")
        LOCAL_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        for i, row in df.iterrows():
            if (i + 1) % 100 == 0:
                print(f"  进度: {i + 1}/{len(df)}")
            get_local_audio_path(row["file_name"])
        print(f"✅ 下载完成: {LOCAL_AUDIO_DIR}")
