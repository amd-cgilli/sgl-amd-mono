# CPU-only FlyDSL compile: no device is visible, a fake rocminfo names gfx950
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PATH=/sgl-workspace/sgl-amd-mono/.port_notes/fakebin:$PATH
export GPU_ARCHS=gfx950 COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_RUNTIME_ENABLE_CACHE=0
