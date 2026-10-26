Title: [Feat] [MI355X] qwen38_flash_next: Add the Qwen3.8-Flash-Next 180B monokernel

## Motivation

Qwen3.8-Flash-Next (SGLang's Qwen4-Exp, `amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8`) decodes with its MTP head (NEXTN) at small batch sizes, where launch gaps dominate. This PR brings the FlyDSL monokernel (sgl-project/sglang#45353) into sgl-amd-mono behind SGLang's monokernel interface (RFC #43949). `--monokernel-backend amd` selects it, and SGLang needs no model hooks or environment flags.

A verify step of 1-8 requests x 4 tokens (embedding, PLE, 48 GDN / QSA layers and the final mixer) runs in 13 persistent launches.

Result on 1 x MI355X, TP1, NEXTN with acceptance pinned at 2.32:

- With one request, the median TPOT decreases from 6.21 ms to 2.87 ms (2.17x). At 2 and 4 requests the speedup is 1.84x and 1.54x, and at 8 requests 1.02x.
- No accuracy difference is detectable. AIME25 (30 problems x 4) pass@1 is 98.33% (off) and 96.67% (on), paired 95% CI [-4.2, 0.0] points, sign test p = 0.50. GSM8K-200 is 96.5% / 97.0%, and RULER NIAH at 256K is 100 / 100.

**The MTP draft model does not run on the monokernel yet.** Monokernel API v1 routes a draft runner by the target's ("target_verify", R, 4) keys: `shape_key` returns None for DRAFT_EXTEND_V2, and the NEXTN draft's one-token DECODE steps are never routed. `draft.py` is ported but not registered; it needs an SGLang change that lets a spec declare its own step modes. Until then the draft runs SGLang's forward.

## Modifications

- `sgl_amd_mono/qwen38_flash_next/`:
  - `spec.py`: the target spec, registered for `Qwen4ExpForConditionalGeneration` on gfx950, TP1, 1-8 requests x 4 tokens. It refuses unsupported configurations with reasons: NEXTN/EAGLE with topk 1, 3 steps and 4 draft tokens only; no ReplaySSM; bf16 KV; `SGLANG_USE_AITER`; the Quark MXFP4 checkpoint; `--attention-backend` triton or aiter; plus per-layer config and weight-layout checks.
  - `draft.py`: the MTP draft spec (not registered, see above).
  - `check.py`: `SGLANG_AMD_MONO_CHECK=1` compares the layer weights against the checkpoint at build, and each eager step's logits and hidden states against SGLang's forward of the same step (`NUMERICS`).
  - `gfx950/`: the kernel (`kernel.py`), its layout, packing and config, the layer views read from SGLang's params in place (`weights.py`), and the host-side runners (`runner.py`).
- `sgl_amd_mono/__init__.py`: SPECS entry for the target.
- `README.md` (Supported Models, Usage, Structure, Testing) and `.github/CODEOWNERS` (`**/*qwen38_flash_next* @salykova`).
- Tests:
  - `test/unit/test_qwen38_flash_next_{spec,layout}.py`, CPU.
  - `test/kernels/test_qwen38_flash_next_compile.py`, GPU.
  - `test/e2e/{qwen38_flash_next_server,test_qwen38_flash_next_server}.py`, GPU: GSM8K at 1/4/16 requests, accept length, KL decode vs prefill, 126k passkey, check mode, fault injection.

## Accuracy Tests

Setup (also for the speed tests):

- Hardware: 1 x MI355X (gfx950), TP1.
- Stack: ROCm 10.0.0, torch 2.11.0+rocm10.0.0, FlyDSL 0.4.0, AITER ee5f7a82.
- SGLang: upstream main 2229345dff. The tested tree also carries the two #45353 commits (5b97337680, 883c4c0a01); their environment flags were unset in every run.
- sgl-amd-mono: this PR at 8bbfb99, installed with `pip install --no-deps --target`.
- Model: `amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8` at bd6e49d9ab.
- Server flags (both arms, without the host, port and model path):
  ```text
  --trust-remote-code --quantization quark --attention-backend triton --tp-size 1
  --page-size 64 --chunked-prefill-size 16384 --mamba-ssm-dtype bfloat16
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4
  --speculative-eagle-topk 1 --mem-fraction-static 0.85 --max-running-requests 8
  --cuda-graph-max-bs-decode 8 --scheduler-recv-interval 10 --stream-interval 50 --enable-metrics
  ```
  - Environment: `SGLANG_USE_AITER=1 SGLANG_USE_AITER_UNIFIED_ATTN=1 AITER_FLYDSL_FORCE=1 SGLANG_ENABLE_SPEC_V2=1 SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`.
  - The evals add `--reasoning-parser qwen3 --context-length 262144`.
- The only difference between the arms is `--monokernel-backend amd`. Each arm runs on a fresh server.
- Accuracy uses real acceptance (no `SGLANG_SIMULATE_ACC_*`).
- Evals: sgl-eval 0.1.3, thinking on (`enable_thinking=true`), T 0.6, top-p 0.95 (top-k 20 from `generation_config.json`), 65,536 max tokens, 8 or fewer requests at a time. So with the monokernel on, every verify step ran the launches. "4 seeds" is `--n-repeats 4`.

| Eval | Monokernel off | Monokernel on | Paired difference |
| --- | --- | --- | --- |
| AIME25 (30 x 4) pass@1 | 98.33% (118 of 120) | 96.67% (116 of 120) | -1.67 points, bootstrap 95% CI [-4.2, 0.0]; 0 better, 2 worse; sign test p = 0.50 |
| AIME25 (30 x 4) pass@4 | 100% | 100% | |
| GSM8K (first 200) | 96.50% | 97.00% | 2 correct only on, 1 correct only off; exact McNemar p = 1.00 |
| RULER NIAH at 256K (4 subtasks x 25) | 100.0 | 100.0 | 0 discordant |

- AIME25: problem 13 fails in 1 of 4 samples (off) and 2 of 4 (on), each time at the token limit. Problem 14 fails once in each arm. With the monokernel on, one sample of problem 9 also fails. 2 samples per arm reached the token limit. The median length was 5,422 tokens (off) and 4,838 tokens (on).
- GSM8K is near its ceiling for this model, so it finds only large errors.
- RULER NIAH: NVIDIA/RULER c3f5e3b, `niah_single_1/2/3` and `niah_multikey_1`, 25 samples each, the same file for both arms. Prompts are 261,353 to 261,996 tokens (`max_position_embeddings` is 262,144). Thinking off, greedy, 128 tokens. Each answer decodes as verify steps of 1 or 2 rows at about 261K positions.
- GPQA-Diamond did not run, because the dataset is gated on the Hub.
- Accept length (real acceptance), off / on: 2.99 / 2.98 on GSM8K, 2.89 / 2.91 on AIME25.

e2e suite (`pytest test/e2e -k qwen38_flash_next`): 6 passed. "Off" is the same `TestMonokernelVerify` class without `--monokernel-backend amd`; there, the routing test fails as expected and the others pass.

| Test | Monokernel on | Off |
| --- | --- | --- |
| GSM8K-200 at 1 / 4 / 16 requests | 0.975 / 0.985 / 0.985 | 0.980 / 0.980 / 0.975 |
| Accept length | 3.089 | 3.082 |
| KL decode vs prefill, 8 prompts in one batch (bound 0.1) | 0.085 | 0.051 |
| Passkey in a 126k-token prompt | pass | pass |

- At 16 requests the batch crosses the 8-request width and back.
- KL on fresh servers (5 runs on, 4 off): 0.041 to 0.085, mean 0.062 (on); 0.051 to 0.069, mean 0.059 (off).

Decode against prefill per batch size (`sglang.test.kl_test_utils` method, 36 to 40 prompts for each batch, mean KL, off / on):

| Batch | 1 | 2 | 3 | 5 | 8 | 9 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mean KL | 0.0735 / 0.0555 | 0.0627 / 0.0501 | 0.0561 / 0.0440 | 0.0596 / 0.0514 | 0.0551 / 0.0547 | 0.0600 / 0.0662 | 0.0605 / 0.0626 |

- At batches 9 and 12, both arms use SGLang's path.

Check mode (`SGLANG_AMD_MONO_CHECK=1`, eager):

- Weights of layers 0, 1 and 3 equal the checkpoint.
- 798 steps checked (221 at 1 request, 13 at 2, 47 at 3, 517 at 4). No value is outside its bound.

| Output | Bound | Worst |
| --- | --- | --- |
| logits, cosine | >= 0.1 | 0.499 |
| logits, relative error | <= 2.5 | 1.31 |
| hidden, cosine | >= 0.3 | 0.54 |
| hidden, relative error | <= 4.0 | 2.08 |

Bitwise: on the same state, the launches are bit-identical to the sgl-project/sglang#45353 kernel on 1,253 of 1,253 eager routed verify steps at 1 to 8 requests (13 layer-boundary residual streams and logits; the mixer output also on the 628 steps of the run that records it).

- A step's first launch can differ from a re-run of the same step in one draft row (slot 1 to 3): 32 of 628 steps in serving order, logits relative difference 0.015 to 0.22, argmax unchanged. So serving with the monokernel on is not bitwise deterministic.

## Speed Tests and Profiling

Method:

- The setup is the same as for the accuracy tests, with `--max-running-requests 16 --cuda-graph-max-bs-decode 16`.
- Speed uses acceptance pinned at 2.32, the InferenceX golden accept length: `SGLANG_SIMULATE_ACC_LEN=2.32 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`. The measured accept length is 2.29 to 2.34 in every cell.
- Workload: `sglang.bench_serving`, random dataset, 8192 in / 1024 out (`--random-range-ratio 1.0`), max(16, 4c) prompts after a warmup.
- Each arm runs on a fresh server, in two rounds. The second round swaps the arm order.
- The table gives the median TPOT per request in ms over both rounds, with a bootstrap 95% CI. P90 interactivity is 1000 / P90 TPOT.

Verify only (the MTP draft runs SGLang's forward):

| c | Monokernel off (ms) | Monokernel on (ms) | Speedup | P90 interactivity off / on (tok/s/user) |
| --- | --- | --- | --- | --- |
| 1 | 6.21 [6.19, 6.25] | 2.87 [2.86, 2.88] | 2.17x | 158.6 / 344.4 |
| 2 | 6.67 [6.65, 6.70] | 3.62 [3.59, 3.65] | 1.84x | 148.6 / 270.4 |
| 4 | 7.46 [7.39, 7.48] | 4.84 [4.79, 4.89] | 1.54x | 131.4 / 198.1 |
| 8 | 8.49 [8.35, 8.66] | 8.35 [8.23, 8.54] | 1.02x | 108.9 / 110.1 |
| 9 | 9.37 [9.10, 9.48] | 9.38 [9.08, 9.47] | 1.00x | 98.8 / 98.9 |
| 12 | 10.07 [9.75, 10.19] | 10.12 [9.80, 10.28] | 1.00x | 90.9 / 90.2 |
| 16 | 12.27 [11.95, 12.51] | 12.23 [11.93, 12.57] | 1.00x | 73.7 / 73.4 |

- From c9, a verify step has more than 8 requests, so both arms use SGLang's path.
- The medians of the two rounds are within 0.15 ms of each other in each cell, and within 0.1 ms at c1 to c8.

Profile (torch profiler, median of 30 steps):

| Requests | Monokernel off | Monokernel on |
| --- | --- | --- |
| 1, verify | 1,630 kernels in 13.68 ms | 16 kernels in 4.49 ms (13 launches, 4.28 ms) |
| 1, whole step | 1,940 kernels in 16.19 ms | 326 kernels in 6.99 ms |
| 8, verify | 1,824 kernels in 17.88 ms | 16 kernels in 16.18 ms (13 launches, 15.96 ms) |
| 8, whole step | 2,141 kernels in 20.99 ms | 333 kernels in 19.27 ms |

At C=1 with the monokernel on, the MTP draft path (draft decode and draft extend) is about 2.24 ms of a 6.99 ms step, which is what registering `draft.py` targets.

- At 8 requests the verify is 1.11x faster. The 8K prefills between steps and the unchanged draft bring the TPOT gain to 1.02x.

Cost (off -> on):

- The spec builds 2,436.7 MiB at load: 1,440.3 MiB of weight copies (GDN 1,026.6, QSA 338.4, PLE 62.5, final mixer 12.8) and about 996 MiB of buffers and tables.
- The KV pool decreases by 1.93% (3,098,112 to 3,038,272 tokens), and the Mamba slots from 537 to 527.
- The verify graphs (bs 1-8, 10, 12, 14, 16) use 1.35 GB (on) against 1.32 GB (off).

Tests on 1 x MI355X, with the setup above:

- unit: 186 passed, 2 failed. All 68 `qwen38_flash_next` tests pass. The two failures are the `deepseek_v41` cases of `test_contract.py` (`hc_combine` and `dsv4/unified_kv_kernels/env_gate.py` are not in this SGLang tree). This PR does not change these files.
- kernels (`-k "qwen38_flash_next or guard"`): 40 passed.
- e2e (`-k qwen38_flash_next`): 6 passed. This includes the check mode and an injected wait fault (`0:1:200`): one fault at address 0x0 in `qwen38_flash_next_layers_s4_r1_pg_f0_b1`, the server stops, and the request gets no tokens.
- pre-commit passes.

Still to do:
- [ ] SGLang: let a monokernel spec declare its step modes (draft extend R x 4, decode R x 1), then register `draft.py` and measure the draft gain.
- [ ] Speed at C=1/2/4/8 with draft model as monokernel.
- [ ] A kernel test against SGLang's ops (pad rows, batch invariance, graph replay across widths); `test_qwen38_flash_next_compile.py` only compiles.
- [ ] GPQA-Diamond, once the dataset is available.

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
