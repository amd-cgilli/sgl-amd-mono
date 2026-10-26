Title: [Feat] [MI355X] qwen38_flash_next: Serve NEXTN verify steps on the monokernel

## Motivation

Part 2 of 4 of the Qwen3.8-Flash-Next monokernel; full results in #29. This PR turns on the monokernel for the NEXTN target-verify steps of `Qwen4ExpForConditionalGeneration`. Every other step, including the MTP draft, still runs SGLang's forward.

Depends on: PR 1 (#TBD).

## Modifications

- `sgl_amd_mono/qwen38_flash_next/spec.py`: the verify spec.
- `gfx950/weights.py`: weight checks and copies for the target model.
- `gfx950/runner.py`: the verify runner.
- `sgl_amd_mono/__init__.py`: `SPECS` entry for `Qwen4ExpForConditionalGeneration`. The package `__init__.py` adds the HF config constants.
- `test/unit/test_qwen38_flash_next_spec.py`, `test/e2e/qwen38_flash_next_server.py`, `test/e2e/test_qwen38_flash_next_server.py` (verify and fault tests). The compile test now uses the runner.
- README: supported-models row, launch command, tree, test line.

## Accuracy Tests

The full verify-only results are in #29 (monokernel on vs off): AIME25 pass@1 96.67% vs 98.33% (2 of 120, not significant: sign test p = 0.50), pass@4 100% both; KL decode vs prefill 0.062 vs 0.059 (bound 0.1); RULER NIAH at ~261K 100 vs 100; GSM8K 0.970 vs 0.975.

GPU check on this branch (1 x MI355X, SGLang main 904691e0b5, real acceptance), e2e `pytest test/e2e -k qwen38_flash_next`, all passed:
- KL decode vs prefill: 0.050 (bound 0.1).
- Passkey deep in a long prompt: passed.
- GSM8K at 1 / 4 / 16 parallel: 0.985 / 0.98 / 0.98, accept length 3.10 (bound 2.8).
- Fault test (stalled wait stops the server without tokens): passed.

Note: run `TestMonokernelFault` with `ulimit -c 0`. Otherwise ROCm writes a full GPU coredump (~73 GB) into the current directory.

## Speed Tests and Profiling

Verify only (draft on SGLang's forward), C=1, acceptance pinned at 2.32: median TPOT 6.21 ms off vs 2.87 ms on, **2.17x** [2.15, 2.18].

GPU check on this branch, C=1, 8K in / 1K out, real acceptance: median TPOT 1.84 ms, 459 tok/s, accept length 3.37.

## Testing

- `pytest test/unit`: 192 passed (62 of them qwen38).
- `pytest test/kernels -k qwen38_flash_next`: 36 passed.
- `pre-commit run --all-files` passes.

## Checklist

- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit or PR this was tested against is named above.
- [x] Accuracy and speed results are attached when kernel or forward code changes.
- [x] A new model adds one line to `.github/CODEOWNERS` that names its owner (already on `main`).
