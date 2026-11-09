# smol-ram

Run very large models with a hard **RAM cap (8 GB)** while pushing the **on-disk (SSD) size** as low as it goes.

The first version ran Kimi K3.1 in 8 GB RAM from a ~1.6 TB SSD checkpoint. This version generalizes it:
given *any* safetensors model and a RAM budget, find the smallest SSD artifact that keeps quality, and
measure the quality and speed cost of each step.

## Why SSD size matters even when only RAM is the constraint

With 8 GB RAM almost nothing stays resident, so **every token streams its weights from SSD**.
Bytes on disk ≈ bytes read per token, so shrinking the artifact also speeds up decode.

## Measured facts: Kimi K3.1 Pro (header-only, no download)

`python3 -m smol.inspect_model moonshotai/Kimi-K3.1-Pro`

| role | GB | dtype |
|---|---:|---|
| routed experts (92 layers × 896, top-16) | 1446.5 | MXFP4 (QAT), u8 E8M0 scale / 32 |
| attention (KDA ×69, MLA+DSA ×24) | 72.7 | bf16 |
| shared experts (2/layer) | 24.3 | bf16 |
| latent MoE up/down proj (7168↔3584) | 9.5 | bf16 |
| embed + lm_head | 4.7 | bf16 |
| router, dense layer 0, norms, vision | 3.6 | bf16/f32 |
| **total** | **1561** | |

Per decoded token: **111.5 GB always-active (bf16)** + 25.8 GB routed experts (16 × 92 × 17.55 MB) = **137 GB read**.
The always-active bf16 part, not the experts, dominates streaming at 8 GB RAM.

**Lossless floor for experts** (`python3 -m smol.entropy_probe`, 27 sampled expert matrices, layers 1/46/91):
FP4 codes carry 3.763 of 4 bits, and the E8M0 scales take only 3 to 4 distinct values (0.745 of 8 bits).
So **4.25 → 3.79 bits/weight, −10.9%, bit-exact**.

## Plan (levers, in the order we test them)

| # | lever | target | quality risk |
|---|---|---|---|
| L0 | lossless: rANS on FP4 codes + scales, bf16 exponent coding for dense tensors | 1561 → ~1360 GB | none (bit-exact) |
| L1 | dense bf16 → 4-bit (HQQ/GPTQ-style, group 64), lossless-coded | 111 → ~30 GB; per-token read 137 → ~50 GB | low |
| L2 | expert usage profiling → prune never/rarely-routed experts (REAP-style) | −25 to −50% of experts | measured |
| L3 | experts MXFP4 → 2–3 bit trellis/codebook, bits allocated by routing frequency | 3.79 → ~2.2 bits | measured; QAT model, so a careful re-quant |
| L4 | expert-delta: shared basis + low-rank residual per layer | experimental | measured |

Target: **Kimi K3.1 Pro under ~300 GB on SSD, still 8 GB RAM**, with perplexity and task deltas reported per lever.

## Method (generic)

1. `inspect_model`: role/byte budget from headers only (works on any HF repo).
2. `entropy_probe`: lossless floor per tensor class from Range-fetched samples.
3. Layer-streaming evaluator: run the model one decoder layer at a time on 1 GPU, reading each layer's weights
   once per pass, to get reference vs compressed logits (KL, PPL) without holding the model.
4. Runtime: mmap/stream engine under a cgroup `memory.max=8G` (page cache counted), reporting tok/s.

## Reports

See `reports/` for raw numbers.
