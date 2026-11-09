#!/usr/bin/env python3
"""SSD-size / RAM-cap planner for any model budget produced by inspect_model.

Given a recipe (bits/weight per role, fraction of routed experts pruned) it reports:
  * on-disk size
  * bytes read from SSD per decoded token when only `ram_gb` is available, filling RAM greedily
    with the bytes that save the most reads per token (always-active tensors first, then hot experts)
  * SSD-bound decode speed at `ssd_gbps`

  python3 -m smol.plan reports/kimi-k3.1-pro_budget.json --ram-gb 8 --recipe lossless
  python3 -m smol.plan reports/*_budget.json --recipe all
"""
import argparse
import json
import os

# Native bits/weight of each dtype as stored (u8 MXFP4 counts as 4.25 incl. scales).
NATIVE_BPW = {"BF16": 16, "F16": 16, "F32": 32, "F8_E4M3": 8, "I8": 8, "U8": 8, "F8_E8M0": 8}

# Recipes: role -> target bits/weight (None = keep native), prune = fraction of routed experts removed.
# lossless numbers come from the measured entropy floor (reports/kimi-k3.1-pro_lossless.md).
# Measured lossless (entropy-coded) size factor per stored dtype; unmeasured dtypes stay 1.0.
LOSSLESS = {"U8": (1279.22 + 7.44) / 1446.46, "BF16": 0.665}   # reports/kimi-k3.1-pro_lossless.md

RECIPES = {
    "native": {},
    "lossless": {"routed_expert": ("lossless",), "dense": ("lossless",)},
    "d4": {"routed_expert": ("lossless",), "dense": ("bpw", 4.5)},
    "d4_e3": {"routed_expert": ("bpw", 3.5), "dense": ("bpw", 4.5)},
    "d4_e2": {"routed_expert": ("bpw", 2.5), "dense": ("bpw", 4.5)},
    "d4_e2_p25": {"routed_expert": ("bpw", 2.5), "dense": ("bpw", 4.5), "prune": 0.25},
    "d3_e2_p50": {"routed_expert": ("bpw", 2.5), "dense": ("bpw", 3.5), "prune": 0.5},
    "d3_e1.6_p50": {"routed_expert": ("bpw", 1.6), "dense": ("bpw", 3.5), "prune": 0.5},
}
DENSE_ROLES = ("attention", "shared_expert", "moe_latent_proj", "dense_mlp", "lm_head", "other")


def role_params(d):
    """Approximate parameter count of a role from its bytes and native dtypes."""
    p = 0.0
    for dt, b in d["dtypes"].items():
        if dt == "U8":         # MXFP4 packed codes (+ scales also U8: 4.25 bits per weight overall)
            p += b * 8 / 4.25
        elif dt == "F8_E8M0":  # scales: do not count as params
            continue
        else:
            p += b * 8 / NATIVE_BPW.get(dt, 16)
    return p


def apply(summary, recipe):
    by = summary["by_role"]
    out = {}
    for role, d in by.items():
        if role == "vision":
            continue
        rule = recipe.get(role) or (recipe.get("dense") if role in DENSE_ROLES else None)
        if rule is None:
            out[role] = d["bytes"]
        elif rule[0] == "lossless":
            out[role] = sum(b * LOSSLESS.get(dt, 1.0) for dt, b in d["dtypes"].items())
        else:
            out[role] = role_params(d) * rule[1] / 8
    out["routed_expert"] = out.get("routed_expert", 0) * (1 - recipe.get("prune", 0.0))
    return out


def per_token(summary, sizes, ram_gb, recipe, hot_skew=0.0):
    """SSD bytes read per token after filling RAM. hot_skew in [0,1): fraction of routed traffic
    served by the hottest experts that fit; 0 = uniform routing (pessimistic)."""
    m = summary["moe"]
    always = sum(v for r, v in sizes.items() if r not in ("routed_expert", "embed"))
    routed_total = sizes.get("routed_expert", 0)
    n_exp = (m["num_experts"] or 0) * (1 - recipe.get("prune", 0.0))
    active_frac = (m["top_k"] / n_exp) if n_exp else 0
    routed_tok = routed_total * active_frac
    budget = ram_gb * 1e9 - 1.5e9          # OS + runtime + KV/state
    cached_always = min(always, max(budget, 0))
    budget -= cached_always
    read = always - cached_always
    if routed_total and budget > 0:
        frac_cached = min(1.0, budget / routed_total)
        # uniform: hit rate = fraction cached; with skew, hot experts serve more traffic
        hit = min(1.0, frac_cached + hot_skew * (1 - frac_cached) * frac_cached)
        read += routed_tok * (1 - hit)
    else:
        read += routed_tok
    return always, routed_tok, read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("budgets", nargs="+")
    ap.add_argument("--recipe", default="all")
    ap.add_argument("--ram-gb", type=float, default=8)
    ap.add_argument("--ssd-gbps", type=float, default=7.0)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    names = list(RECIPES) if a.recipe == "all" else a.recipe.split(",")
    print(f"RAM {a.ram_gb} GB, SSD {a.ssd_gbps} GB/s (reads are SSD-bound; compute ignored)\n")
    print("| model | recipe | SSD GB | always-active GB/tok | routed GB/tok | read GB/tok @RAM | tok/s |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for f in a.budgets:
        s = json.load(open(f))["summary"]
        name = os.path.basename(f).replace("_budget.json", "")
        for r in names:
            rec = RECIPES[r]
            sizes = apply(s, rec)
            always, routed, read = per_token(s, sizes, a.ram_gb, rec)
            print(f"| {name} | {r} | {sum(sizes.values()) / 1e9:.0f} | {always / 1e9:.1f} | {routed / 1e9:.1f} | "
                  f"{read / 1e9:.1f} | {a.ssd_gbps * 1e9 / max(read, 1):.2f} |")


if __name__ == "__main__":
    main()
