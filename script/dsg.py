import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import snapshot_download
from datasets import load_dataset, Audio

if __name__ == "__main__":
    # Step 1: Download raw files (no multiprocessing)
    local_dir = snapshot_download(
        repo_id="yeeko/Elaina_WanderingWitch_audio_JA",
        repo_type="dataset",
        local_dir="../datasets/elaina",
    )

    # Step 2: Load from local folder
    ds = load_dataset(
        "audiofolder",
        data_dir="../datasets/elaina/train",
        num_proc=1,
    )

    print(ds)
    print(ds["train"].column_names)
    print(ds["train"][0])