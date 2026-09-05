"""Download the TinyStories dataset (25MB subset) for training the 10M LLM.

Usage:
    python scripts/download_tinystories.py

This downloads a 25MB subset of the TinyStories dataset from HuggingFace.
The full dataset is ~500MB, but 25MB is enough for meaningful training on CPU.
"""
import os, urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.join(DATA_DIR, "tinystories_train.txt")
URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
TARGET_BYTES = 25 * 1024 * 1024  # 25MB

os.makedirs(DATA_DIR, exist_ok=True)

existing = os.path.getsize(OUT) if os.path.exists(OUT) else 0
if existing >= TARGET_BYTES:
    print(f"TinyStories already downloaded: {OUT} ({existing/1024/1024:.1f}MB)")
    exit(0)

print(f"Downloading {TARGET_BYTES/1024/1024:.0f}MB of TinyStories to {OUT}...")
req = urllib.request.Request(URL, headers={"Range": f"bytes=0-{TARGET_BYTES}"})
with urllib.request.urlopen(req, timeout=60) as r:
    with open(OUT, "wb") as f:
        downloaded = 0
        while downloaded < TARGET_BYTES:
            chunk = r.read(65536)
            if not chunk: break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (5*1024*1024) < 65536:
                print(f"  {downloaded/1024/1024:.1f}MB")

print(f"\nDone! Downloaded {os.path.getsize(OUT)/1024/1024:.1f}MB")
print(f"Ready for training: python scripts/train_10m.py --mode awf")
