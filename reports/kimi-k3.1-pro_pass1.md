# Kimi K3.1 Pro, pass 1: quality of each SSD lever

All 93 layers, layer-streamed on one H100 (Modal), 16x1024 eval tokens (wikitext-2 test),
8x1024 calibration tokens (wikitext-2 train) used only for expert-usage counts. Run time 38 min.
Quantizers are plain round-to-nearest (RTN) with group 64, fp16 scale + zero; this is the naive baseline.
SSD GB assumes lossless coding for anything left at native precision (scan: bf16 -> 66.5%, MXFP4 -> 3.786 bpw).

| variant | dense bpw | expert bpw | experts kept | SSD GB (est.) | KL vs ref (nats/tok) | PPL |
|---|---:|---:|---:|---:|---:|---:|
| `ref` | 16.00 (lossless 10.6) | 3.79 | 100% | 1366 | 0.000 | 1.415 |
| `dense_int4g64` | 4.50 | 3.79 | 100% | 1323 | 0.053 | 1.483 |
| `dense_int3g64` | 3.50 | 3.79 | 100% | 1316 | 0.322 | 1.935 |
| `exp_int3g64` | 16.00 (lossless 10.6) | 3.50 | 100% | 1268 | 0.059 | 1.493 |
| `exp_int2g64` | 16.00 (lossless 10.6) | 2.50 | 100% | 928 | 0.387 | 2.073 |
| `prune_25` | 16.00 (lossless 10.6) | 3.79 | 75% | 1044 | 0.053 | 1.475 |
| `prune_50` | 16.00 (lossless 10.6) | 3.79 | 50% | 721 | 0.284 | 1.856 |
| `dense_int4g64+exp_int2g64+prune_25` | 4.50 | 2.50 | 75% | 673 | 0.590 | 2.527 |

Reference PPL 1.41 means wikitext is largely memorized; KL vs the unmodified model is the metric that matters.
Calibration routing: on average 45 of 896 experts per layer got zero tokens in 8192 tokens.
