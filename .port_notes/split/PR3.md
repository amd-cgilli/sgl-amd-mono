Title: [Feat] [MI355X] qwen38_flash_next: Add check mode

## Motivation

Part 3 of 4 of the Qwen3.8-Flash-Next monokernel; full results in #29. This PR adds check mode (`SGLANG_AMD_MONO_CHECK`), a debugging tool that compares the weights and every monokernel step against SGLang's forward. It is off by default, so serving does not change.

Depends on: PR 2 (#TBD).

## Modifications

- `sgl_amd_mono/qwen38_flash_next/check.py`: new.
- `spec.py`: check hooks, which import `check.py` lazily only when check mode is on.
- `test/e2e/test_qwen38_flash_next_server.py`: `TestMonokernelCheck`.
- README: tree.

## Accuracy Tests

Not applicable: the default path is unchanged.

## Speed Tests and Profiling

Not applicable: the default path is unchanged.

## Testing

- `pytest test/unit`: 194 passed (64 of them qwen38).
- `TestMonokernelCheck` passed on 1 x MI355X (in the PR 4 stack run).
- `pre-commit run --all-files` passes.

## Checklist

- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit or PR this was tested against is named above: SGLang main 904691e0b5.
- [x] Accuracy and speed results are attached when kernel or forward code changes (not applicable).
- [x] A new model adds one line to `.github/CODEOWNERS` that names its owner (already on `main`).
