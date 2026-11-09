#!/usr/bin/env python3
"""Aggregate per-shard lossless scan JSON into a per-role table.

  python3 -m smol.aggregate_lossless <dir with shard jsons> [--out reports/x.md]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from smol.inspect_model import role_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "*.json")))
    R = defaultdict(lambda: defaultdict(float))
    for f in files:
        for t in json.load(open(f))["tensors"]:
            n = t["name"]
            role = role_of(n)
            if role == "routed_expert":
                role += ".codes" if n.endswith("packed") else ".scales"
            r = R[role]
            b = t["bytes"]
            r["bytes"] += b
            r["zstd"] += t["zstd_bytes"]
            r["dec_time"] += b / max(t["zstd_decode_GBps"], 1e-6) / 1e9
            if "h_nibble" in t:   # entropy-coded size of 4-bit codes
                r["ent"] += b * t["h_nibble"] / 4
                r["ent1"] += b * t["h1_nibble"] / 4
            elif t["dtype"] == "U8":
                r["ent"] += b * t["h_byte"] / 8
                r["ent1"] += b * t["h_byte"] / 8
            elif "bits_entropy" in t:
                r["ent"] += b * t["bits_entropy"] / 16
                r["ent1"] += b * t["bits_entropy"] / 16
            else:
                r["ent"] += t["zstd_bytes"]
                r["ent1"] += t["zstd_bytes"]
    lines = [f"Shards scanned: {len(files)}", "",
             "| role | raw GB | zstd-3 GB | zstd % | entropy GB (order-0) | entropy % | order-1 GB | zstd decode GB/s (1 stream) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    tot = defaultdict(float)
    for role, r in sorted(R.items(), key=lambda kv: -kv[1]["bytes"]):
        for k in ("bytes", "zstd", "ent", "ent1", "dec_time"):
            tot[k] += r[k]
        lines.append(f"| {role} | {r['bytes'] / 1e9:.2f} | {r['zstd'] / 1e9:.2f} | {100 * r['zstd'] / r['bytes']:.1f} | "
                     f"{r['ent'] / 1e9:.2f} | {100 * r['ent'] / r['bytes']:.1f} | {r['ent1'] / 1e9:.2f} | "
                     f"{r['bytes'] / r['dec_time'] / 1e9:.2f} |")
    lines.append(f"| **total** | **{tot['bytes'] / 1e9:.2f}** | **{tot['zstd'] / 1e9:.2f}** | {100 * tot['zstd'] / tot['bytes']:.1f} | "
                 f"**{tot['ent'] / 1e9:.2f}** | {100 * tot['ent'] / tot['bytes']:.1f} | {tot['ent1'] / 1e9:.2f} | "
                 f"{tot['bytes'] / tot['dec_time'] / 1e9:.2f} |")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
