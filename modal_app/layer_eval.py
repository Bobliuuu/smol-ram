"""Layer-streaming quality evaluator for huge MoE checkpoints (Kimi K3.1 Pro first).

One pass over the weights on one GPU. Each decoder layer is loaded once, then run for:
  * a calibration stream (reference weights) -> per-expert routing counts, used to choose pruning
  * N variant streams over the eval tokens (reference + compressed variants), each carrying its own
    hidden state, so every variant is evaluated end to end, not just layer-wise.
At the end: NLL / PPL per variant and KL(ref || variant) over the full vocabulary.

Eval sequences are <= 2048 tokens, so K3.1's DSA indexer (index_topk=2048) selects every key and
the K3 dense-attention modeling code is exact for K3.1.

  modal run --detach modal_app/layer_eval.py --variants ref,dense_int4,exp_int3 --n-seq 8 --seq-len 1024
"""
import json
import os
import sys
import threading
import time

import modal

WEIGHTS_VOL = "kimi-k31-pro"
K3_RAW = "https://huggingface.co/moonshotai/Kimi-K3/raw/main"
K31_RAW = "https://huggingface.co/moonshotai/Kimi-K3.1-Pro/raw/main"

app = modal.App("smol-ram-layer-eval")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("curl", "git")
    .pip_install("torch==2.10.0")
    .pip_install("fla-core==0.6.0", "transformers==4.57.6", "safetensors==0.6.2", "einops==0.8.1",
                 "tiktoken==0.12.0", "pyarrow==21.0.0", "numpy==2.1.3", "packaging", "blobfile")
    .run_commands(
        "mkdir -p /opt/kimi",
        "touch /opt/kimi/__init__.py",
        f"curl -sfL {K3_RAW}/modeling_kimi_linear.py -o /opt/kimi/modeling_kimi_linear.py",
        f"curl -sfL {K31_RAW}/configuration_kimi_k3.py -o /opt/kimi/configuration_kimi_k3.py",
        f"curl -sfL {K31_RAW}/tokenization_kimi.py -o /opt/kimi/tokenization_kimi.py",
        # only decoder layers are used; make whole-model helpers optional across transformers versions
        "python -c \"import re,pathlib; p=pathlib.Path('/opt/kimi/modeling_kimi_linear.py'); s=p.read_text(); "
        "s=s.replace('from transformers.utils.generic import OutputRecorder, check_model_inputs', "
        "'try:\\n    from transformers.utils.generic import OutputRecorder, check_model_inputs\\n"
        "except ImportError:\\n    OutputRecorder = lambda *a, **k: None\\n    check_model_inputs = lambda f=None, **k: (f if f else (lambda g: g))'); p.write_text(s)\"",
        "python -c \"import transformers, fla; print('transformers', transformers.__version__, 'fla', fla.__version__)\"",
    )
    .add_local_python_source("smol")
)
wvol = modal.Volume.from_name(WEIGHTS_VOL, version=2)
rvol = modal.Volume.from_name("smol-ram-results", create_if_missing=True)

PFX = "language_model.model."


# ----------------------------------------------------------------------------- weights
class Shards:
    """Lazy safetensors access across shards via the index (mmap; nothing is read until used)."""

    def __init__(self, root):
        from safetensors import safe_open
        self.root = root
        self.wmap = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
        self._open = {}
        self._so = safe_open

    def get(self, name, device="cpu"):
        f = self.wmap[name]
        if f not in self._open:
            self._open[f] = self._so(f"{self.root}/{f}", framework="pt", device="cpu")
        return self._open[f].get_tensor(name)

    def layer_keys(self, i):
        p = f"{PFX}layers.{i}."
        return [k for k in self.wmap if k.startswith(p)]


def load_layer_tensors(sh, i):
    """Return (dense: {local_name: tensor}, packed experts: (w1p,w1s,w3p,w3s,w2p,w2s) stacked)."""
    import torch
    p = f"{PFX}layers.{i}."
    dense, ex = {}, {}
    for k in sh.layer_keys(i):
        loc = k[len(p):]
        if ".experts." in loc:
            _, _, rest = loc.partition(".experts.")
            e, w, kind = rest.split(".", 2)   # "<e>.w1.weight_packed"
            ex.setdefault((w, kind), {})[int(e)] = sh.get(k)
        else:
            dense[loc] = sh.get(k)
    packed = None
    if ex:
        n = max(ex[("w1", "weight_packed")]) + 1
        packed = {f"{w}_{'p' if kind.endswith('packed') else 's'}":
                  torch.stack([ex[(w, kind)][e] for e in range(n)])
                  for (w, kind) in ex}
    return dense, packed


# ----------------------------------------------------------------------------- MXFP4
def fp4_lut(device):
    import torch
    v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return torch.tensor(v + [-x for x in v], dtype=torch.float32, device=device)


def dequant_mxfp4(packed, scale, lut):
    """packed u8 [..., out, in/2] (low nibble = even column), scale u8 E8M0 [..., out, in/32] -> f32."""
    import torch
    lo = lut[(packed & 15).long()]
    hi = lut[(packed >> 4).long()]
    w = torch.stack([lo, hi], dim=-1).flatten(-2)                 # [..., out, in]
    s = torch.exp2(scale.float() - 127.0)                          # [..., out, in/32]
    w = w.unflatten(-1, (s.shape[-1], 32)) * s.unsqueeze(-1)
    return w.flatten(-2)


# ----------------------------------------------------------------------------- variants
def rtn(w, bits, group, sym=False):
    """Round-to-nearest group quantization, returns dequantized weight (same dtype)."""
    import torch
    shp, dt = w.shape, w.dtype
    x = w.float().reshape(-1, group)
    if sym:
        qmax = 2 ** (bits - 1) - 1
        s = x.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
        q = (x / s).round().clamp(-qmax - 1, qmax)
        y = q * s
    else:
        lo, hi = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
        s = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-12)
        q = ((x - lo) / s).round().clamp(0, 2 ** bits - 1)
        y = q * s + lo
    return y.reshape(shp).to(dt)


# bits per weight on disk for each format (codes + group metadata, fp16 scale [+ fp16 zero])
def bpw_rtn(bits, group, sym=False):
    return bits + (16 if sym else 32) / group


DENSE_QUANT_SKIP = ("norm", "A_log", "dt_bias", "conv1d", "e_score_correction_bias", "res_proj", "gate.weight")


def make_variant(spec):
    """spec string -> dict(dense_fn, expert_fn, prune_frac, bpw_dense, bpw_expert)."""
    v = {"name": spec, "dense_fn": None, "expert_fn": None, "prune_frac": 0.0,
         "bpw_dense": 16.0, "bpw_expert": 3.786}  # lossless-coded MXFP4 is the reference expert cost
    for part in spec.split("+"):
        if part == "ref":
            continue
        kind, _, arg = part.partition("_")
        if kind == "dense":            # dense_int4g64
            b, g = int(arg[3]), int(arg.split("g")[1])
            v["dense_fn"] = (lambda b, g: lambda w: rtn(w, b, g))(b, g)
            v["bpw_dense"] = bpw_rtn(b, g)
        elif kind == "exp":            # exp_int3g64
            b, g = int(arg[3]), int(arg.split("g")[1])
            v["expert_fn"] = (lambda b, g: lambda w: rtn(w, b, g))(b, g)
            v["bpw_expert"] = bpw_rtn(b, g)
        elif kind == "prune":          # prune_25  (percent of least-used experts per layer)
            v["prune_frac"] = int(arg) / 100.0
        else:
            raise ValueError(part)
    return v


# ----------------------------------------------------------------------------- layer exec
def build_layer(cfg, i, dense, device):
    import torch
    from kimi.modeling_kimi_linear import KimiDecoderLayer
    for k in [k for k in dense if ".indexer." in k]:   # DSA indexer: selects all keys at <=2048 tokens
        dense.pop(k)
    with torch.device("meta"):
        layer = KimiDecoderLayer(cfg, i)
    moe = getattr(layer, "block_sparse_moe", None)
    if moe is not None:
        moe.experts = torch.nn.ModuleList()   # replaced by packed experts
    layer = layer.to_empty(device=device).to(torch.bfloat16)
    # Checkpoints may zero-pad small vectors (K3/K3.1 store KDA A_log as [128] for 96 heads).
    shapes = {k: p.shape for k, p in layer.state_dict().items()}
    for k, t in list(dense.items()):
        want = shapes.get(k)
        if want is not None and t.shape != want:
            assert t.dim() == len(want) and t.shape[1:] == want[1:] and t.shape[0] > want[0], (k, t.shape, want)
            assert not t[want[0]:].any(), f"{k}: non-zero tail, refusing to slice"
            dense[k] = t[: want[0]].contiguous()
    missing, unexpected = layer.load_state_dict({k: v.to(device) for k, v in dense.items()}, strict=False)
    unexpected = [k for k in unexpected if ".indexer." not in k]   # DSA indexer unused at <=2048 tokens
    assert not missing and not unexpected, (i, missing[:5], unexpected[:5])
    for k, t in dense.items():   # keep tensors the checkpoint stores in fp32 (A_log, dt_bias, conv, bias) in fp32
        if t.dtype == torch.float32 and ".indexer." not in k:
            layer.get_parameter(k).data = t.to(device)
    layer.eval()
    return layer


def set_dense_weights(layer, dense, fn, device):
    import torch
    with torch.no_grad():
        for k, t in dense.items():
            if t.dim() < 2 or any(s in k for s in DENSE_QUANT_SKIP):
                continue
            mod_param = layer.get_parameter(k)
            src = t.to(device)
            mod_param.copy_(fn(src) if fn is not None else src)


def install_packed_moe(layer, packed, lut, expert_fn, keep_mask, counts_out):
    """Patch moe_infer: per-expert MXFP4 dequant on the fly (+ optional lossy transform / pruning)."""
    import torch
    moe = layer.block_sparse_moe
    act = moe.shared_experts.act_fn if hasattr(moe.shared_experts, "act_fn") else None
    dev = lut.device
    if keep_mask is not None:
        with torch.no_grad():
            moe.gate.e_score_correction_bias.masked_fill_(~keep_mask.to(dev), -1e9)

    def W(name, e):
        w = dequant_mxfp4(packed[f"{name}_p"][e], packed[f"{name}_s"][e], lut).to(torch.bfloat16)
        return expert_fn(w) if expert_fn is not None else w

    @torch.no_grad()
    def moe_infer(x, topk_ids, topk_weight):
        T, K = topk_ids.shape
        flat = topk_ids.view(-1)
        cnt = torch.bincount(flat, minlength=moe.num_experts)
        if counts_out is not None:
            counts_out.add_(cnt.cpu())
        idxs = flat.argsort()
        sorted_tokens = x[idxs // K]
        out = torch.empty_like(sorted_tokens)
        start = 0
        for e, n in enumerate(cnt.tolist()):
            if n == 0:
                continue
            xs = sorted_tokens[start:start + n]
            xs = xs.to(torch.bfloat16)
            g = xs @ W("w1", e).T
            u = xs @ W("w3", e).T
            h = act(torch.cat([g, u], -1)) if act is not None else torch.nn.functional.silu(g) * u
            out[start:start + n] = (h @ W("w2", e).T).to(out.dtype)
            start += n
        new_x = torch.empty_like(out)
        new_x[idxs] = out
        return (new_x.view(T, K, -1).float() * topk_weight.unsqueeze(-1)).sum(1).to(x.dtype)

    moe.moe_infer = moe_infer


# ----------------------------------------------------------------------------- data
def tokenize_corpus(tok_dir, n_seq, seq_len, split):
    """wikitext-2-raw test (eval) / train (calibration) -> [n_seq, seq_len] token ids."""
    import io
    import urllib.request
    import pyarrow.parquet as pq
    import torch
    sys.path.insert(0, "/opt")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    url = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/"
           f"{split}-00000-of-00001.parquet")
    text = "".join(pq.read_table(io.BytesIO(urllib.request.urlopen(url).read())).column("text").to_pylist())
    ids = tok(text[: 40_000_000], add_special_tokens=False)["input_ids"]
    need = n_seq * seq_len
    assert len(ids) >= need, (len(ids), need)
    return torch.tensor(ids[:need]).view(n_seq, seq_len)


# ----------------------------------------------------------------------------- main pass
@app.function(image=image, gpu="H100", cpu=16, memory=262144, timeout=6 * 3600,
              volumes={"/w": wvol, "/results": rvol})
def run(variants: str, n_seq: int, seq_len: int, n_cal: int, tag: str, max_layers: int = 0):
    import torch
    sys.path.insert(0, "/opt")
    # make the tokenizer + config loadable from the weights dir
    from kimi.configuration_kimi_k3 import KimiLinearConfig
    dev = torch.device("cuda")
    torch.set_grad_enabled(False)
    t0 = time.time()
    log = []

    def say(*a):
        msg = f"[{time.time() - t0:7.1f}s] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.append(msg)

    full = json.load(open("/w/config.json"))
    tc = dict(full["text_config"])
    tc.pop("auto_map", None)
    tc.pop("quantization_config", None)
    cfg = KimiLinearConfig(**tc)
    cfg._attn_implementation = "sdpa"
    n_layers = cfg.num_hidden_layers if not max_layers else max_layers
    specs = [make_variant(s) for s in variants.split(",")]
    say("variants:", [s["name"] for s in specs], "layers:", n_layers)

    ids = tokenize_corpus("/w", n_seq, seq_len, "test").to(dev)
    cal = tokenize_corpus("/w", n_cal, seq_len, "train").to(dev) if n_cal else None
    say("tokens eval", tuple(ids.shape), "cal", None if cal is None else tuple(cal.shape))

    sh = Shards("/w")
    emb = sh.get(PFX + "embed_tokens.weight").to(dev)
    lut = fp4_lut(dev)
    H = cfg.hidden_size

    def init_state(tok):
        h = emb[tok]
        return [h, h.new_zeros(h.shape[0] * h.shape[1], 0, H)]

    states = [init_state(ids) for _ in specs]
    cal_state = init_state(cal) if cal is not None else None
    del emb
    usage = {}
    werr = {}

    # prefetch next layer's tensors on a thread while the GPU runs this one
    nxt = {}

    def prefetch(i):
        nxt[i] = load_layer_tensors(sh, i)

    th = threading.Thread(target=prefetch, args=(0,))
    th.start()
    for i in range(n_layers):
        th.join()
        dense, packed = nxt.pop(i)
        if i + 1 < n_layers:
            th = threading.Thread(target=prefetch, args=(i + 1,))
            th.start()
        tl = time.time()
        if packed is not None:
            packed = {k: v.to(dev) for k, v in packed.items()}   # ~15.7 GB packed MXFP4 per layer
        layer = build_layer(cfg, i, dense, dev)
        keep = None
        if packed is not None and cal_state is not None:
            counts = torch.zeros(cfg.num_experts, dtype=torch.long)
            install_packed_moe(layer, packed, lut, None, None, counts)
            cal_state[0], cal_state[1] = layer(cal_state[0], block_residual=cal_state[1])
            usage[i] = counts.tolist()
        for s, st in zip(specs, states):
            set_dense_weights(layer, dense, s["dense_fn"], dev)
            if packed is not None:
                keep = None
                if s["prune_frac"] > 0:
                    assert i in usage, "pruning needs a calibration stream (--n-cal > 0)"
                    order = torch.tensor(usage[i]).argsort()          # least used first
                    keep = torch.ones(cfg.num_experts, dtype=torch.bool)
                    keep[order[: int(s["prune_frac"] * cfg.num_experts)]] = False
                with torch.no_grad():   # restore bias (masked_fill is in-place per variant)
                    layer.block_sparse_moe.gate.e_score_correction_bias.copy_(
                        dense["block_sparse_moe.gate.e_score_correction_bias"].to(dev))
                install_packed_moe(layer, packed, lut, s["expert_fn"], keep, None)
            st[0], st[1] = layer(st[0], block_residual=st[1])
        # weight-space error for expert transforms, sampled on 4 experts
        if packed is not None:
            for s in specs:
                if s["expert_fn"] is None:
                    continue
                errs = []
                for e in (0, 1, 2, 3):
                    w = dequant_mxfp4(packed["w1_p"][e].to(dev), packed["w1_s"][e].to(dev), lut)
                    errs.append(((s["expert_fn"](w.bfloat16()).float() - w).norm() / w.norm()).item())
                werr.setdefault(s["name"], []).append(sum(errs) / len(errs))
        del layer, dense, packed
        import gc
        gc.collect()   # moe_infer closure <-> module cycle holds the packed experts otherwise
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        say(f"layer {i} done in {time.time() - tl:.1f}s; mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # head
    from kimi.modeling_kimi_linear import _apply_attn_res, KimiRMSNorm
    onorm = KimiRMSNorm(H, eps=cfg.rms_norm_eps).to(dev)
    onorm.weight.copy_(sh.get(PFX + "output_attn_res_norm.weight").to(dev))
    oproj = torch.nn.Linear(H, 1, bias=False).to(dev, torch.bfloat16)
    oproj.weight.copy_(sh.get(PFX + "output_attn_res_proj.weight").to(dev))
    fnorm = KimiRMSNorm(H, eps=cfg.rms_norm_eps).to(dev)
    fnorm.weight.copy_(sh.get(PFX + "norm.weight").to(dev))
    head = sh.get("language_model.lm_head.weight").to(dev)

    tgt = ids[:, 1:].reshape(-1)
    res = {}
    ref_lp = None
    for s, st in zip(specs, states):
        h, br = st
        h = _apply_attn_res(h.view(-1, H), br, oproj, onorm).view(*h.shape) if br.shape[1] else h
        h = fnorm(h)[:, :-1].reshape(-1, H).to(head.dtype)
        nll, kl, n = 0.0, 0.0, 0
        lps = []
        for c in range(0, h.shape[0], 1024):
            lp = torch.log_softmax((h[c:c + 1024] @ head.T).float(), -1)
            nll += -lp.gather(1, tgt[c:c + 1024, None]).sum().item()
            if ref_lp is not None:
                r = ref_lp[c // 1024]
                kl += (r.exp() * (r - lp)).sum().item()
            elif s["name"] == "ref":
                lps.append(lp.half())   # keep reference log-probs for KL (fp16 to save memory)
            n += lp.shape[0]
        if s["name"] == "ref":
            ref_lp = [x.float() for x in lps]
        res[s["name"]] = {"nll": nll / n, "ppl": float(torch.tensor(nll / n).exp()),
                          "kl_vs_ref": (kl / n) if ref_lp is not None and s["name"] != "ref" else 0.0,
                          "bpw_dense": s["bpw_dense"], "bpw_expert": s["bpw_expert"],
                          "prune_frac": s["prune_frac"],
                          "expert_rel_werr": (sum(werr[s["name"]]) / len(werr[s["name"]])) if s["name"] in werr else 0.0,
                          "tokens": n}
        say(s["name"], json.dumps(res[s["name"]]))
    out = {"tag": tag, "variants": variants, "n_seq": n_seq, "seq_len": seq_len, "n_cal": n_cal,
           "layers": n_layers, "results": res, "usage": usage, "log": log, "seconds": time.time() - t0}
    os.makedirs("/results/eval", exist_ok=True)
    with open(f"/results/eval/{tag}.json", "w") as f:
        json.dump(out, f)
    rvol.commit()
    return res


@app.local_entrypoint()
def main(variants: str = "ref", n_seq: int = 8, seq_len: int = 1024, n_cal: int = 4,
         tag: str = "", max_layers: int = 0):
    tag = tag or f"eval-{int(time.time())}"
    print(json.dumps(run.remote(variants, n_seq, seq_len, n_cal, tag, max_layers), indent=1))
