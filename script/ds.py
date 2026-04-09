import sys
sys.stdout.flush()  # 让 multiprocess 的警告不要乱入
import os
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"

from datasets import load_dataset

ds = load_dataset("yeeko/Elaina_WanderingWitch_audio_JA",
                  cache_dir="../datasets",
                  download_mode="force_redownload",
                  )
print(ds)                    # 看结构
print(ds["train"][0])        # 看第一条
print(ds["train"][0]["audio"])  # 访问 audio 会自动下载