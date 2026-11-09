"""Copy a HF repo's safetensors shards into a Modal Volume, one container per shard.

  modal run --detach modal_app/fetch_weights.py --repo moonshotai/Kimi-K3.1-Pro --vol kimi-k31-pro
Re-running skips shards whose size already matches.
"""
import json
import os
import time

import modal

app = modal.App("smol-ram-fetch-weights")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]==0.36.0").env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"})


def _vol(name):
    return modal.Volume.from_name(name, create_if_missing=True, version=2)


VOL_NAME = os.environ.get("SMOL_VOL", "kimi-k31-pro")
vol = _vol(VOL_NAME)


@app.function(image=image, cpu=4, memory=8192, timeout=7200, volumes={"/w": vol}, retries=3,
              ephemeral_disk=64 * 1024)
def fetch(repo: str, fname: str, size: int):
    from huggingface_hub import hf_hub_download
    dst = f"/w/{fname}"
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return fname, "skip"
    t = time.time()
    p = hf_hub_download(repo, fname, local_dir="/tmp/dl")
    os.makedirs(os.path.dirname(dst) or "/w", exist_ok=True)
    os.replace(p, dst) if os.stat(p).st_dev == os.stat("/w").st_dev else __import__("shutil").move(p, dst)
    vol.commit()
    return fname, f"{size / 1e9 / (time.time() - t):.2f} GB/s"


@app.local_entrypoint()
def main(repo: str = "moonshotai/Kimi-K3.1-Pro"):
    import urllib.request
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main"))
    files = [(f["path"], f.get("size", 0)) for f in tree if f["type"] == "file" and not f["path"].startswith("assets")]
    print(len(files), "files ->", VOL_NAME, sum(s for _, s in files) / 1e12, "TB")
    for r in fetch.starmap([(repo, f, s) for f, s in files], order_outputs=False, return_exceptions=True):
        print(r, flush=True)
