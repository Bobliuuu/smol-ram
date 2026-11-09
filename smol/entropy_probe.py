#!/usr/bin/env python3
"""Measure how compressible packed low-bit tensors are, losslessly.

Fetches a sample of tensors by HTTP Range (no full download) and reports the
order-0 Shannon entropy of 4-bit codes (two per byte) and of per-group scale
bytes. Entropy is the floor for a rANS/Huffman coder that keeps the bits exact,
so this is the guaranteed zero-quality-loss SSD saving.

  python3 -m smol.entropy_probe moonshotai/Kimi-K3.1-Pro --layers 1 30 60 91 --experts 0 7 300 895
"""
import argparse
import json
import math
import struct
import urllib.request
from collections import Counter

from smol.inspect_model import _hf_url, _range


def entropy(counts):
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values() if c), n


def header(repo, shard, cache={}):
    if shard not in cache:
        url = _hf_url(repo, shard)
        n = struct.unpack("<Q", _range(url, 0, 7))[0]
        cache[shard] = (8 + n, json.loads(_range(url, 8, 8 + n - 1)))
    return cache[shard]


def fetch(repo, wmap, name):
    shard = wmap[name]
    base, h = header(repo, shard)
    s, e = h[name]["data_offsets"]
    return _range(_hf_url(repo, shard), base + s, base + e - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--prefix", default="language_model.model.layers.{L}.block_sparse_moe.experts.{E}.{W}")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 30, 60, 91])
    ap.add_argument("--experts", type=int, nargs="+", default=[0, 7, 300, 895])
    ap.add_argument("--mats", nargs="+", default=["w1", "w2", "w3"])
    a = ap.parse_args()
    req = urllib.request.Request(_hf_url(a.repo, "model.safetensors.index.json"))
    wmap = json.load(urllib.request.urlopen(req, timeout=120))["weight_map"]
    nib_all, sc_all = Counter(), Counter()
    rows = []
    for L in a.layers:
        for E in a.experts:
            for W in a.mats:
                p = a.prefix.format(L=L, E=E, W=W)
                packed = fetch(a.repo, wmap, p + ".weight_packed")
                scale = fetch(a.repo, wmap, p + ".weight_scale")
                nib = Counter()
                for b, c in Counter(packed).items():
                    nib[b & 15] += c
                    nib[b >> 4] += c
                sc = Counter(scale)
                hn, _ = entropy(nib)
                hs, _ = entropy(sc)
                nib_all.update(nib)
                sc_all.update(sc)
                rows.append((L, E, W, hn, hs))
                print(f"L{L:<3} E{E:<4} {W}: code H={hn:.3f} bits/4  scale H={hs:.3f} bits/8 "
                      f"({len(sc)} distinct)")
    hn, nn = entropy(nib_all)
    hs, ns = entropy(sc_all)
    raw = 4 + 8 / 32  # bits per weight: 4-bit code + one u8 scale per 32
    coded = hn + hs / 32
    print(f"\nPOOLED: codes {hn:.3f}/4 bits, scales {hs:.3f}/8 bits")
    print(f"bits/weight raw {raw:.3f} -> entropy-coded {coded:.3f}  (saves {100 * (1 - coded / raw):.1f}%)")
    print("code histogram:", {k: round(v / nn, 4) for k, v in sorted(nib_all.items())})


if __name__ == "__main__":
    main()
