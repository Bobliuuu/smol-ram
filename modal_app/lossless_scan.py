"""Exact lossless-compression scan of every tensor in a HF safetensors repo, on Modal.

One container per shard. Tensors are streamed with HTTP Range reads (no disk),
then for each tensor we record:
  * order-0 entropy of the natural symbol (FP4 nibble / scale byte / bf16 exponent)
  * order-1 nibble entropy (does the previous code predict the next?)
  * real zstd size (bf16: byte-plane split first), round-trip verified, + decode speed
Results go to the `smol-ram-results` Volume as one JSON per shard and are returned.

  modal run --detach modal_app/lossless_scan.py --repo moonshotai/Kimi-K3.1-Pro
"""
import json
import os
import time

import modal

app = modal.App("smol-ram-lossless-scan")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3", "zstandard==0.23.0", "requests==2.32.3")
vol = modal.Volume.from_name("smol-ram-results", create_if_missing=True)


def _get(url, start, end, sess):
    for attempt in range(6):
        try:
            r = sess.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            r.raise_for_status()
            if len(r.content) == end - start + 1:
                return r.content
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"range fetch failed {url} {start}-{end}")


def _h(counts):
    import numpy as np
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def analyze(name, dtype, raw):
    import numpy as np
    import zstandard as zstd
    out = {"name": name, "dtype": dtype, "bytes": len(raw)}
    a = np.frombuffer(raw, dtype=np.uint8)
    if dtype == "U8" and name.endswith("weight_packed"):
        lo, hi = a & 15, a >> 4
        nib = np.empty(a.size * 2, np.uint8)
        nib[0::2], nib[1::2] = lo, hi
        out["h_nibble"] = _h(np.bincount(nib, minlength=16))
        # order-1: H(n_i | n_{i-1})
        pair = np.bincount(nib[:-1].astype(np.int32) * 16 + nib[1:], minlength=256).reshape(16, 16)
        out["h1_nibble"] = _h(pair.ravel()) - _h(pair.sum(1))
        out["h_byte"] = _h(np.bincount(a, minlength=256))
        planes = [raw]
    elif dtype == "U8":  # scales
        out["h_byte"] = _h(np.bincount(a, minlength=256))
        out["distinct"] = int((np.bincount(a, minlength=256) > 0).sum())
        planes = [raw]
    elif dtype in ("BF16", "F16"):
        u = np.frombuffer(raw, dtype=np.uint16)
        if dtype == "BF16":
            exp = (u >> 7) & 0xFF
            out["h_exp"] = _h(np.bincount(exp, minlength=256))
            out["bits_entropy"] = 1 + 7 + out["h_exp"]  # sign + mantissa raw + coded exponent
        planes = [(u >> 8).astype(np.uint8).tobytes(), (u & 0xFF).astype(np.uint8).tobytes()]
    else:
        planes = [raw]
    c = zstd.ZstdCompressor(level=3, threads=-1)
    d = zstd.ZstdDecompressor()
    comp = [c.compress(p) for p in planes]
    t = time.perf_counter()
    back = [d.decompress(x) for x in comp]
    dt = time.perf_counter() - t
    assert all(x == y for x, y in zip(back, planes)), f"round-trip mismatch {name}"
    out["zstd_bytes"] = sum(len(x) for x in comp)
    out["zstd_decode_GBps"] = len(raw) / max(dt, 1e-9) / 1e9
    return out


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={"/results": vol}, retries=2)
def scan_shard(repo: str, shard: str, tag: str):
    import requests
    import struct
    sess = requests.Session()
    url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
    tok = os.environ.get("HF_TOKEN")
    if tok:
        sess.headers["Authorization"] = f"Bearer {tok}"
    n = struct.unpack("<Q", _get(url, 0, 7, sess))[0]
    hdr = json.loads(_get(url, 8, 8 + n - 1, sess))
    hdr.pop("__metadata__", None)
    base = 8 + n
    items = sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
    rows, t0 = [], time.time()
    # Fetch in ~256 MB contiguous windows to keep request count low.
    i = 0
    while i < len(items):
        j, s0 = i, items[i][1]["data_offsets"][0]
        while j < len(items) and items[j][1]["data_offsets"][1] - s0 <= 256 << 20:
            j += 1
        j = max(j, i + 1)
        e0 = items[j - 1][1]["data_offsets"][1]
        blob = _get(url, base + s0, base + e0 - 1, sess)
        for name, m in items[i:j]:
            s, e = m["data_offsets"]
            rows.append(analyze(name, m["dtype"], blob[s - s0:e - s0]))
        i = j
    res = {"shard": shard, "seconds": time.time() - t0, "tensors": rows}
    os.makedirs(f"/results/{tag}", exist_ok=True)
    with open(f"/results/{tag}/{shard}.json", "w") as f:
        json.dump(res, f)
    vol.commit()
    return {"shard": shard, "n": len(rows), "seconds": res["seconds"]}


@app.local_entrypoint()
def main(repo: str = "moonshotai/Kimi-K3.1-Pro", tag: str = "kimi-k3.1-pro", limit: int = 0):
    import urllib.request
    idx = json.load(urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
    if limit:
        shards = shards[:limit]
    print(f"{len(shards)} shards")
    for r in scan_shard.starmap([(repo, s, tag) for s in shards], return_exceptions=True):
        print(r)
