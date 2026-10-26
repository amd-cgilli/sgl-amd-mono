Title: [Feat] [MI355X] qwen38_flash_next: Add the Qwen3.8-Flash-Next 180B monokernel

## Motivation

Qwen3.8-Flash-Next (SGLang's Qwen4-Exp, `amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8`) decodes with its MTP head (NEXTN) at small batch sizes, where launch gaps dominate. This PR brings the FlyDSL monokernel (sgl-project/sglang#45353) into sgl-amd-mono behind SGLang's monokernel interface (RFC #43949). `--monokernel-backend amd` selects it, and SGLang needs no model hooks or environment flags.

- A verify step of 1-8 requests x 4 tokens (embedding, PLE, 48 GDN / QSA layers and the final mixer) runs in 13 persistent launches.
- Each MTP draft step of 1-8 requests runs in one launch: a draft extend (R x 4 rows) or a draft decode step (R x 1 rows).

Monokernel API v1 routes every runner of a speculative server by the target's verify key, which the NEXTN draft never runs. The draft spec therefore routes its own steps: a wrapper in front of the draft's `model.forward`. It can move to SGLang if a later monokernel API routes a draft's own step modes.

## Modifications

- `sgl_amd_mono/qwen38_flash_next/`:
  - `spec.py`: the target spec, registered for `Qwen4ExpForConditionalGeneration` on gfx950, TP1, 1-8 requests x 4 tokens. It refuses unsupported configurations with reasons:
    - NEXTN/EAGLE with topk 1, 3 steps and 4 draft tokens only;
    - no ReplaySSM;
    - bf16 KV;
    - `SGLANG_USE_AITER`;
    - the Quark MXFP4 checkpoint;
    - `--attention-backend` triton or aiter;
    - plus per-layer config and weight-layout checks.
  - `draft.py`: the MTP draft spec, registered for `Qwen4ExpForCausalLMMTP`.
    - `_build` installs a wrapper in front of the draft's `model.forward`.
    - The wrapper sends draft extend (`DRAFT_EXTEND_V2`, R x 4 rows) and draft decode (`DECODE`, R x 1 rows) steps of R <= 8 requests to the draft launch, and every other step to SGLang's forward.
    - The decision depends only on the step's shape, so a captured graph replays it.
    - It requires the draft's QSA attention backends (`"qsa"`), logs each draft step at its first launch, and refuses during capture a draft-shaped step the launch cannot read.
  - `check.py`: `SGLANG_AMD_MONO_CHECK=1` checks the target only:
    - at build, it compares the layer weights against the checkpoint;
    - on each eager verify step, it compares the logits and hidden states against SGLang's forward of the same step (`NUMERICS`).
  - `gfx950/`:
    - the kernel (`kernel.py`), its layout, packing and config;
    - the layer views read from SGLang's params in place (`weights.py`);
    - the host-side runners (`runner.py`). The draft runner copies `input_ids` and new hidden states into buffers it owns, so its pointer tables stay valid under graph capture.
- `sgl_amd_mono/__init__.py`: SPECS entries for the target and the MTP draft.
- `README.md` (Supported Models, Usage, Structure, Testing) and `.github/CODEOWNERS` (`**/*qwen38_flash_next* @salykova`).
- Tests:
  - `test/unit/test_qwen38_flash_next_{spec,layout}.py`, CPU. These cover the specs, refusals and layout, and the draft wrapper:
    - routing of draft extend and decode steps;
    - the fallback to SGLang's forward;
    - stacking under the runtime's forward switch;
    - the per-mode backend check;
    - the capture-time refusal;
    - the fixed-address input buffers.
  - `test/kernels/test_qwen38_flash_next_compile.py`, GPU.
  - `test/e2e/{qwen38_flash_next_server,test_qwen38_flash_next_server}.py`, GPU:
    - GSM8K at 1/4/16 requests, accept length;
    - the target and draft launch lines;
    - KL decode vs prefill;
    - 126k passkey;
    - check mode;
    - fault injection (target launches).

## Accuracy Tests

1 x MI355X, TP1, SGLang upstream main 2229345dff, NEXTN 3 steps / 4 draft tokens / topk 1, real acceptance. "Off" is the same server without `--monokernel-backend amd`.

| Test | Monokernel on | Off |
|---|---|---|
| GSM8K, 400 q | 0.970 (accept length 3.52) | 0.975 (accept length 3.52) |
| GSM8K 5-shot chat, 200 q, at 1 / 4 / 16 requests | 0.98 / 0.98 / 0.97 | 0.98 / 0.985 / 0.975 |
| AIME25, 30 q x 4 seeds, pass@1 (thinking on, T 0.6, top-p 0.95) | 96.67% (116 of 120) | 98.33% (118 of 120) |
| AIME25, 30 q x 4 seeds, pass@4 | 100% | 100% |
| KL decode vs prefill, mean over fresh servers (bound 0.1) | 0.062 (5 runs, 0.041 to 0.085) | 0.059 (4 runs, 0.051 to 0.069) |
| RULER NIAH at ~261K tokens (niah_single_1/2/3, niah_multikey_1, 25 each) | 100 | 100 |

- **AIME25:** the 1.67-point difference is two problems. Paired by question it is not significant: 95% CI [-4.2, 0.0] points, sign test p = 0.50.
- **Which code:** the rows above ran with the verify steps on the monokernel and the draft on SGLang's forward. With the draft on the monokernel as well, GSM8K-200 (thinking on, T 0.6, top-p 0.95) is 97.5% with accept length 2.99, against 98.0% / 2.97 with only verify on the monokernel.

## Speed Tests and Profiling

1 x MI355X, TP1, the same server flags with and without `--monokernel-backend amd`, CUDA graphs up to 8 requests. Verify and draft are both on the monokernel.

Client:

```bash
python -m sglang.bench_serving --backend sglang --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts <max(10 x conc, 20)> --max-concurrency <conc> --warmup-requests 2 --seed 1
```

**Real acceptance:**

- 3 rounds on 3 GPUs, a fresh server per arm, with the arm order alternated across GPUs. Each server first ran one C=1 priming pass, which is not counted.
- Median TPOT is over all requests of the 3 rounds, with a bootstrap 95% CI. Output tok/s is the mean of the 3 rounds.
- Accept length is generated tokens / verify calls, from the server metrics.

| C | Off: median TPOT / output tok/s | On: median TPOT / output tok/s | Speedup: TPOT [95% CI] / tok/s | Accept length off / on |
|---|---|---|---|---|
| 1 | 4.04 ms / 220 | 1.60 ms / 517 | **2.53x** [2.28, 2.84] / 2.35x | 3.36 / 3.38 |
| 2 | 4.58 ms / 400 | 2.17 ms / 812 | **2.12x** [1.89, 2.26] / 2.03x | 3.39 / 3.43 |
| 4 | 4.99 ms / 669 | 3.21 ms / 1063 | **1.56x** [1.51, 1.71] / 1.59x | 3.39 / 3.40 |
| 8 | 6.22 ms / 1088 | 5.86 ms / 1180 | **1.06x** [1.02, 1.09] / 1.08x | 3.38 / 3.40 |

Acceptance is the same in both arms, so the gain comes from the kernel.

**Acceptance pinned at 2.32** (InferenceX's golden accept length), median TPOT and P90 interactivity:

| C | Monokernel off | Monokernel on | Gain |
|---|---|---|---|
| 1 | 6.21 ms / 160 tok/s/user | 2.46 ms / 401 tok/s/user | **2.52x** |
| 2 | 6.70 ms / 148 tok/s/user | 3.23 ms / 305 tok/s/user | **2.08x** |
| 4 | 7.40 ms / 131 tok/s/user | 4.46 ms / 214 tok/s/user | **1.66x** |

From 9 requests a verify step has more than 8 requests, and both arms run SGLang's path.

Still to do:
- [ ] Re-run AIME25, KL and RULER NIAH with the draft on the monokernel.
- [ ] GPQA-Diamond (the dataset is gated on the Hub).
- [ ] A kernel test against SGLang's ops with graph replay across widths (`test/kernels` only compiles the launches today).

## Checklist

- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit tested against: upstream main 2229345dff.
- [x] Accuracy and speed results.
- [x] `.github/CODEOWNERS` line (`@salykova`; please confirm the handle).

## Review and Merge Process

1. The `lint` and `Check PR title` checks must pass.
2. Ask a code owner from `.github/CODEOWNERS` to review.
3. Merge with "Squash and merge"; consider `Co-authored-by: salykova`.
