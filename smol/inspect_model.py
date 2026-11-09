#!/usr/bin/env python3
"""Header-only inspection of a safetensors model (local dir or Hugging Face repo).

Reads only the JSON header of every shard (HTTP Range requests for HF repos),
so a 1.5 TB checkpoint is inspected in seconds without downloading weights.

Output: per-tensor-role byte totals, plus the bytes touched per decoded token
for MoE models (always-active tensors + top-k routed experts).

Usage:
  python3 -m smol.inspect_model moonshotai/Kimi-K3.1-Pro --out reports/k31_tensors.json
  python3 -m smol.inspect_model /path/to/local/model
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import struct
import sys
import urllib.request

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "F8_E8M0": 1, "U32": 4, "U16": 2, "F4": 0.5}

# Ordered (role, regex). First match wins. Generic across DeepSeek/Kimi/Qwen/Mixtral/GLM naming.
ROLES = [
    ("vision", r"(vision_tower|visual|mm_projector|vision_model|multi_modal_projector)"),
    ("routed_expert", r"\.experts\.\d+\.|\.experts\.(gate_up|down|w\d)"),
    ("shared_expert", r"shared_expert"),
    ("router", r"(\.gate\.(weight|e_score_correction_bias)$|\.router\.|gate\.bias$)"),
    ("moe_latent_proj", r"routed_expert_(up|down)_proj|routed_expert_norm"),
    # Row-lookup tables: a token touches only a few rows, so they are not "always active".
    ("embed", r"embed_tokens|wte|word_embeddings|engram\.embed|(^|\.)embed\.(weight|scale)$"),
    ("lm_head", r"lm_head|(^|\.)head\.weight$|output\.weight$"),
    ("norm", r"norm|A_log|dt_bias|conv1d|\.bias$"),
    ("attention", r"(self_attn|attention|attn)"),
    ("dense_mlp", r"\.mlp\."),
    ("residual_mix", r"_res_proj"),
]


def role_of(name):
    for role, rx in ROLES:
        if re.search(rx, name):
            return role
    return "other"


def _hf_url(repo, fname, rev="main"):
    return f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"


def _range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                               "User-Agent": "smol-ram-inspect/0.1"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def read_header(src, fname, remote):
    if remote:
        url = _hf_url(src, fname)
        n = struct.unpack("<Q", _range(url, 0, 7))[0]
        raw = _range(url, 8, 8 + n - 1)
    else:
        with open(os.path.join(src, fname), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            raw = f.read(n)
    h = json.loads(raw)
    h.pop("__metadata__", None)
    return fname, h


def load_text(src, fname, remote):
    if remote:
        req = urllib.request.Request(_hf_url(src, fname), headers={"User-Agent": "smol-ram-inspect/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read().decode()
    with open(os.path.join(src, fname)) as f:
        return f.read()


def inspect(src):
    remote = not os.path.isdir(src)
    try:
        idx = json.loads(load_text(src, "model.safetensors.index.json", remote))
        shards = sorted(set(idx["weight_map"].values()))
    except Exception:
        shards = ["model.safetensors"]
    cfg = json.loads(load_text(src, "config.json", remote))
    tensors = {}
    with cf.ThreadPoolExecutor(16) as ex:
        for fname, h in ex.map(lambda s: read_header(src, s, remote), shards):
            for name, meta in h.items():
                s, e = meta["data_offsets"]
                tensors[name] = {"shard": fname, "dtype": meta["dtype"],
                                 "shape": meta["shape"], "bytes": e - s}
    return cfg, tensors


def text_cfg(cfg):
    return cfg.get("text_config", cfg)


def summarize(cfg, tensors):
    t = text_cfg(cfg)
    n_exp = t.get("num_experts") or t.get("n_routed_experts") or t.get("num_local_experts")
    top_k = t.get("num_experts_per_token") or t.get("num_experts_per_tok")
    by_role = {}
    for name, m in tensors.items():
        r = role_of(name)
        d = by_role.setdefault(r, {"tensors": 0, "bytes": 0, "dtypes": {}})
        d["tensors"] += 1
        d["bytes"] += m["bytes"]
        d["dtypes"][m["dtype"]] = d["dtypes"].get(m["dtype"], 0) + m["bytes"]
    total = sum(d["bytes"] for d in by_role.values())
    # Per-token touched bytes (text decode): all non-vision, non-routed tensors, embed is a row lookup.
    always = sum(d["bytes"] for r, d in by_role.items()
                 if r not in ("vision", "routed_expert", "embed"))
    routed = by_role.get("routed_expert", {}).get("bytes", 0)
    per_expert = routed / (n_exp * max(1, count_moe_layers(tensors))) if n_exp and routed else 0
    moe_layers = count_moe_layers(tensors)
    active_routed = per_expert * (top_k or 0) * moe_layers
    return {
        "total_bytes": total,
        "by_role": by_role,
        "moe": {"num_experts": n_exp, "top_k": top_k, "moe_layers": moe_layers,
                "bytes_per_expert": per_expert, "routed_bytes": routed},
        "per_token": {"always_active_bytes": always, "active_routed_bytes": active_routed,
                      "total_touched_bytes": always + active_routed},
    }


def count_moe_layers(tensors):
    layers = set()
    for name in tensors:
        m = re.search(r"layers\.(\d+)\..*\.experts\.", name)
        if m:
            layers.add(int(m.group(1)))
    return len(layers)


def gb(x):
    return f"{x / 1e9:9.2f} GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="HF repo id or local dir")
    ap.add_argument("--out", help="write full per-tensor json here")
    a = ap.parse_args()
    cfg, tensors = inspect(a.src)
    s = summarize(cfg, tensors)
    print(f"{a.src}: {len(tensors)} tensors, {gb(s['total_bytes'])}")
    for r, d in sorted(s["by_role"].items(), key=lambda x: -x[1]["bytes"]):
        dt = ", ".join(f"{k}:{v / 1e9:.1f}G" for k, v in d["dtypes"].items())
        print(f"  {r:16s} {gb(d['bytes'])}  {100 * d['bytes'] / s['total_bytes']:5.1f}%  [{dt}]")
    m, p = s["moe"], s["per_token"]
    if m["num_experts"]:
        print(f"  MoE: {m['moe_layers']} layers x {m['num_experts']} experts, top-{m['top_k']}, "
              f"{m['bytes_per_expert'] / 1e6:.2f} MB/expert")
    print(f"  per token: always-active {gb(p['always_active_bytes'])} + routed {gb(p['active_routed_bytes'])}"
          f" = {gb(p['total_touched_bytes'])}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"src": a.src, "summary": s, "tensors": tensors}, f)
        print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
