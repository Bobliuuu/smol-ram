#!/usr/bin/env python3
"""Derive README/video numbers from reports/*. Writes results/derived/readme_data.json. Nothing hand-typed."""
import json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: os.path.join(ROOT, "reports", f)

def md_table(path, start_after=None):
    rows, hdr = [], None
    for ln in open(path):
        if not ln.startswith("|"):
            if rows: break
            continue
        cells = [c.strip().strip("`*") for c in ln.strip().strip("|").split("|")]
        if hdr is None: hdr = cells; continue
        if set(ln.strip()) <= set("|-: "): continue
        rows.append(dict(zip(hdr, cells)))
    return rows

def num(s):
    return float(re.match(r"-?[\d.]+", s.replace(",", "")).group(0))

out = {}
b = json.load(open(R("kimi-k3.1-pro_budget.json")))["summary"]
gb = lambda x: round(x / 1e9, 1)
out["budget"] = {"total_gb": gb(b["total_bytes"]), "experts": b["moe"]["num_experts"], "top_k": b["moe"]["top_k"], "moe_layers": b["moe"]["moe_layers"],
                 "always_gb": gb(b["per_token"]["always_active_bytes"]), "routed_gb": gb(b["per_token"]["active_routed_bytes"])}
roles = [("routed experts", "routed_expert"), ("attention", "attention"), ("shared experts", "shared_expert"), ("latent MoE projections", "moe_latent_proj"),
         ("embed + lm_head", None)]
rows = []
for lab, k in roles:
    if k: rows.append({"role": lab, "gb": gb(b["by_role"][k]["bytes"])})
    else: rows.append({"role": lab, "gb": gb(b["by_role"]["embed"]["bytes"] + b["by_role"]["lm_head"]["bytes"])})
used = {"routed_expert", "attention", "shared_expert", "moe_latent_proj", "embed", "lm_head"}
rows.append({"role": "router, dense, vision, norms", "gb": gb(sum(v["bytes"] for k, v in b["by_role"].items() if k not in used))})
out["roles"] = rows

ll = md_table(R("kimi-k3.1-pro_lossless.md"))
tot = [r for r in ll if r["role"] == "total"][0]
out["lossless"] = {"native_gb": num(tot["raw GB"]), "entropy_gb": num(tot["entropy GB (order-0)"]), "entropy_pct": num(tot["entropy %"]),
                   "zstd_gb": num(tot["zstd-3 GB"]), "zstd_pct": num(tot["zstd %"])}
out["lossless_rows"] = [{"stream": r["role"], "raw_gb": num(r["raw GB"]), "entropy_pct": num(r["entropy %"])} for r in ll if r["role"] in
                        ("routed_expert.codes", "routed_expert.scales", "attention", "shared_expert")]

p1 = json.load(open(R("kimi-k3.1-pro_pass1.json")))
tbl = {r["variant"]: r for r in md_table(R("kimi-k3.1-pro_pass1.md"))}
levers = []
for name, v in p1["results"].items():
    levers.append({"variant": name, "ssd_gb": num(tbl[name]["SSD GB (est.)"]), "kl": round(v["kl_vs_ref"], 3), "ppl": round(v["ppl"], 3)})
out["levers"] = levers
out["pass1"] = {"layers": p1["layers"], "tokens": p1["results"]["ref"]["tokens"]}
byv = {l["variant"]: l for l in levers}
out["lever_gb"] = {"prune25_saves": round(byv["ref"]["ssd_gb"] - byv["prune_25"]["ssd_gb"]), "int4_saves": round(byv["ref"]["ssd_gb"] - byv["dense_int4g64"]["ssd_gb"]),
                   "best_gb": byv["dense_int4g64+exp_int2g64+prune_25"]["ssd_gb"], "ref_gb": byv["ref"]["ssd_gb"]}

plan = md_table(R("plan_8gb.md"))
out["plan"] = [{"model": r["model"], "recipe": r["recipe"], "ssd_gb": num(r["SSD GB"]), "toks": num(r["tok/s"])} for r in plan]
nat = {r["model"]: r for r in out["plan"] if r["recipe"] == "native"}
out["plan_models"] = len(nat)
k = {r["recipe"]: r for r in out["plan"] if r["model"] == "kimi-k3.1-pro"}
out["kimi_plan"] = {"native_toks": k["native"]["toks"], "best_toks": k["d3_e1.6_p50"]["toks"], "best_gb": k["d3_e1.6_p50"]["ssd_gb"]}
# per-model: native vs best recipe tok/s
out["plan_chart"] = [{"model": m, "recipe": rec, "toks": next(r["toks"] for r in out["plan"] if r["model"] == m and r["recipe"] == rec)}
                     for m in nat for rec in ("native", "d3_e1.6_p50")]
out["plan_ssd"] = [{"model": m, "recipe": rec, "ssd_gb": next(r["ssd_gb"] for r in out["plan"] if r["model"] == m and r["recipe"] == rec)}
                   for m in nat for rec in ("native", "d3_e1.6_p50")]
os.makedirs(os.path.join(ROOT, "results/derived"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results/derived/readme_data.json"), "w"), indent=1)
print(json.dumps(out)[:900])
