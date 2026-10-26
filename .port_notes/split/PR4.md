Title: [Feat] [MI355X] qwen38_flash_next: Run the MTP draft on the monokernel

## Motivation

Part 4 of 4 of the Qwen3.8-Flash-Next monokernel; full results in #29. This PR also runs the MTP draft steps (`Qwen4ExpForCausalLMMTP`, draft extend and draft decode) on the monokernel. Their routing goes through a wrapper of the draft's forward, so it is safe under CUDA graphs.

Depends on: PR 3 (#TBD).

## Modifications

- `sgl_amd_mono/qwen38_flash_next/draft.py`: the draft spec and its wrapper.
- `gfx950/runner.py`: `DraftRunner` and its input buffers.
- `gfx950/weights.py`: MTP weight checks and copies.
- `sgl_amd_mono/__init__.py`: `SPECS` entry for `Qwen4ExpForCausalLMMTP`.
- `test/unit/test_qwen38_flash_next_spec.py`: draft tests. `test/e2e/test_qwen38_flash_next_server.py`: draft log assertions.
- README: supported-models row, routing note, tree.

## Accuracy Tests

GPU check on this branch (1 x MI355X, SGLang main 904691e0b5, real acceptance), full e2e file: 6 passed.
- KL decode vs prefill: 0.062 (bound 0.1).
- Passkey deep in a long prompt: passed.
- GSM8K at 1 / 4 / 16 parallel: 0.98 / 0.98 / 0.98, accept length 3.08 (bound 2.8).
- Check mode and the fault test both passed.

## Speed Tests and Profiling

C=1:
- Real acceptance, monokernel on vs off: 1.60 ms / 517 tok/s vs 4.04 ms / 220 tok/s, **2.53x** (#29, SGLang 2229345dff); re-run on SGLang main 904691e0b5: 1.59 ms / 530 vs 4.11 ms / 223, **2.59x** [2.29, 2.94].
- Draft on the monokernel vs verify-only, acceptance pinned at 2.32: **1.17x**.

GPU check on this branch, C=1, 8K in / 1K out, real acceptance: median TPOT 1.60 ms, 511 tok/s, accept length 3.37. Verify-only (PR 2) gave 1.84 ms.

## Testing

- `pytest test/unit`: 228 passed (98 of them qwen38).
- `pre-commit run --all-files` passes.

## Checklist

- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit or PR this was tested against is named above.
- [x] Accuracy and speed results are attached when kernel or forward code changes.
- [x] A new model adds one line to `.github/CODEOWNERS` that names its owner (already on `main`).
