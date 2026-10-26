Title: [Feat] [MI355X] qwen38_flash_next: Add the gfx950 kernel

## Motivation

Part 1 of 4 of the Qwen3.8-Flash-Next monokernel; full results in #29. This PR adds only the FlyDSL kernel for gfx950 with its weight layouts and packing, so it can be reviewed by itself. The kernel is adapted from sgl-project/sglang#45353. Nothing is registered in `SPECS`, so serving does not change.

Depends on: `main`.

## Modifications

- `sgl_amd_mono/qwen38_flash_next/gfx950/`: `kernel.py`, `layout.py`, `packing.py`, `config.py`, `device.py`, `__init__.py`. The MTP launch is included because it is part of the kernel.
- `sgl_amd_mono/qwen38_flash_next/__init__.py`: docstring update.
- `test/unit/test_qwen38_flash_next_layout.py`, `test/kernels/test_qwen38_flash_next_compile.py`.
- README: package tree and test commands.

## Accuracy Tests

Not applicable: nothing serves on the kernel yet. Results are in #29 and PR 2.

## Speed Tests and Profiling

Not applicable, for the same reason.

## Testing

- `pytest test/unit`: 138 passed (8 of them qwen38).
- `pytest test/kernels -k qwen38_flash_next` on 1 x MI355X: 36 passed.
- `pre-commit run --all-files` passes.

## Checklist

- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit or PR this was tested against is named above: SGLang main 904691e0b5.
- [ ] Accuracy and speed results are attached when kernel or forward code changes: in PR 2 and #29.
- [x] A new model adds one line to `.github/CODEOWNERS` that names its owner (already on `main`).
