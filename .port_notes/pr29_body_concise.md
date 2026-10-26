Tracking PR for the Qwen3.8-Flash-Next monokernel: it holds the end-to-end results and is not meant to merge. The code lands in 4 stacked PRs, in this order: #50 (gfx950 kernel), #51 (NEXTN verify spec), #52 (check mode), #53 (MTP draft).

## Motivation

Qwen3.8-Flash-Next (`amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8`) decodes with its MTP head (NEXTN) at small batch sizes, where launch gaps dominate. This brings the FlyDSL monokernel of sgl-project/sglang#45353 into sgl-amd-mono behind SGLang's monokernel interface (RFC #43949), selected with `--monokernel-backend amd`.

- A verify step of 1-8 requests x 4 tokens (embedding, PLE, 48 GDN / QSA layers, final mixer) runs in 13 persistent launches.
- Each MTP draft step (draft extend R x 4 rows, draft decode R x 1) runs in one launch. API v1 does not route a draft's own steps, so the draft spec routes them with a wrapper of the draft's forward.
- gfx950, TP1, bf16 KV, NEXTN topk 1 / 3 steps / 4 draft tokens. Steps above 8 requests run SGLang's path.

## Accuracy Tests

1 x MI355X, TP1, real acceptance; "Off" is the same server without `--monokernel-backend amd`.

| Test | On | Off |
|---|---|---|
| GSM8K, 400 q | 0.970 (accept length 3.52) | 0.975 (accept length 3.52) |
| GSM8K 5-shot chat, 200 q, at 1 / 4 / 16 requests | 0.98 / 0.98 / 0.97 | 0.98 / 0.985 / 0.975 |
| AIME25, 30 q x 4 seeds, pass@1 (thinking on, T 0.6, top-p 0.95) | 96.67% | 98.33% |
| AIME25 pass@4 | 100% | 100% |
| GPQA-Diamond pass@1 (same settings) | 0.855 (sem .006) | 0.836 (sem .004) |
| GPQA-Diamond majority@4 / pass@4 | 0.871 / 0.909 | 0.884 / 0.899 |
| KL decode vs prefill, mean over fresh servers (bound 0.1) | 0.062 | 0.059 |
| RULER NIAH at ~261K tokens (4 subtasks x 25) | 100 | 100 |

## Speed Tests and Profiling

1 x MI355X, TP1, same flags with and without `--monokernel-backend amd`, real acceptance, 3 rounds on 3 GPUs:

```bash
python -m sglang.bench_serving --backend sglang --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts <max(10 x conc, 20)> --max-concurrency <conc> --warmup-requests 2 --seed 1
```

| C | Off: median TPOT / output tok/s | On: median TPOT / output tok/s | Speedup (TPOT) | Accept length off / on |
|---|---|---|---|---|
| 1 | 4.04 ms / 220 | 1.60 ms / 517 | **2.53x** | 3.36 / 3.38 |
| 2 | 4.58 ms / 400 | 2.17 ms / 812 | **2.12x** | 3.39 / 3.43 |
| 4 | 4.99 ms / 669 | 3.21 ms / 1063 | **1.56x** | 3.39 / 3.40 |
| 8 | 6.22 ms / 1088 | 5.86 ms / 1180 | **1.06x** | 3.38 / 3.40 |

Re-run on SGLang main 904691e0b5, C=1: 1.59 ms / 530 tok/s on vs 4.11 ms / 223 off (2.59x); the PRs need no SGLang change.
