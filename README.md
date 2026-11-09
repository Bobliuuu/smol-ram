# smol-ram

Run very large models under a hard **RAM cap (8 GB)** while pushing their **SSD footprint** down, for any
safetensors model. Kimi K3.1 Pro (2.78T params, 1.56 TB) is the first target.

With 8 GB RAM almost nothing stays resident, so each token streams its weights from SSD.
Bytes on disk are close to bytes read per token, so a smaller artifact is also a faster one.

## Tools

| tool | what it does |
|---|---|
| `python3 -m smol.inspect_model <hf-repo>` | byte budget by tensor role from shard headers only (HTTP Range, no download) |
| `python3 -m smol.entropy_probe <hf-repo>` | lossless floor of packed low-bit tensors from sampled experts |
| `modal run --detach modal_app/lossless_scan.py` | exact entropy + verified zstd round-trip of every tensor, one container per shard |
| `modal run --detach modal_app/fetch_weights.py` | copy a repo into a Modal Volume (one container per shard) |
| `modal run --detach modal_app/layer_eval.py --variants ...` | layer-streamed end-to-end quality of compression variants on one GPU |
| `python3 -m smol.plan reports/*_budget.json` | SSD size, per-token reads and SSD-bound tok/s under a RAM cap |

## Kimi K3.1 Pro: where the bytes are

| role | GB | stored as |
|---|---:|---|
| routed experts (92 layers × 896, top-16) | 1446.5 | MXFP4 (QAT), E8M0 scale per 32 |
| attention (69 KDA + 24 MLA/DSA layers) | 72.7 | bf16 |
| shared experts | 24.3 | bf16 |
| latent MoE projections (7168↔3584) | 9.5 | bf16 |
| embed, lm_head, router, dense layer 0, vision, norms | 8.2 | bf16 / f32 |

Each token reads **111.5 GB of always-active bf16** plus 25.8 GB of routed experts. Unlike DeepSeek V4.1,
GLM-5.3, MiniMax M3 or Qwen3.5-397B (10 to 24 GB always-active), K3.1's dense part dominates at 8 GB RAM.
See `reports/plan_8gb.md`.

## Lossless (bit-exact): 1561 → 1363 GB

Full scan of all 96 shards (`reports/kimi-k3.1-pro_lossless.md`):

| stream | native GB | entropy floor | zstd-3 |
|---|---:|---:|---:|
| FP4 expert codes | 1361.4 | 94.0% | 94.3% |
| E8M0 expert scales | 85.1 | 8.7% | 16.4% |
| bf16 tensors | 114.7 | 66.5% | 71.4% |
| **total** | **1561.2** | **1362.9 (87.3%)** | **1380.0 (88.4%)** |

Order-1 context on FP4 codes gains less than 0.2%, so neighbouring codes are close to independent.

## Lossy levers, measured end to end (`reports/kimi-k3.1-pro_pass1.md`)

All 93 layers, one H100, 16K eval tokens; KL is against the unmodified model. Quantizers are naive RTN (g64).

| variant | SSD GB | KL (nats/tok) |
|---|---:|---:|
| reference, lossless-coded | 1366 | 0 |
| prune 25% least-routed experts | 1044 | 0.053 |
| dense → int4 | 1323 | 0.053 |
| experts → int3 | 1268 | 0.059 |
| prune 50% | 721 | 0.284 |
| dense → int3 | 1316 | 0.322 |
| experts → int2 | 928 | 0.387 |
| dense int4 + experts int2 + prune 25% | 673 | 0.590 |

Pruning is the cheapest lever per GB: −322 GB for the same KL that int4 dense costs for −43 GB.
Naive RTN below 4 bits is too lossy on this QAT model.

## Next

1. Better quantizers: GPTQ/HQQ with per-expert calibration Hessians; codebook/trellis at 2 to 2.5 bpw.
2. Better pruning: REAP-style saliency (gate weight × output norm) and expert merging, with more calibration tokens.
3. Runtime: mmap/stream engine under `memory.max=8G` with an entropy decoder fast enough for 7 GB/s.
