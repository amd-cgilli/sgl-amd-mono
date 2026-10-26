import sys, torch
sys.path.insert(0, "/sgl-workspace/sgl-amd-mono/test/unit")
import test_qwen38_flash_next_spec as T
from sgl_amd_mono.qwen38_flash_next.gfx950 import weights
# real CPU tensors in place of the meta ones
def real(ns):
    for k, v in list(vars(ns).items()):
        if isinstance(v, torch.Tensor):
            r = torch.zeros(v.shape, dtype=v.dtype) if v.is_contiguous() else torch.zeros(tuple(reversed(v.shape)), dtype=v.dtype).t()
            for a in ("is_shuffled",):
                if hasattr(v, a): setattr(r, a, getattr(v, a))
            setattr(ns, k, r)
        elif hasattr(v, "__dict__") and not callable(v):
            real(v)
import types; F8 = types.ModuleType("fp8_utils"); F8.use_aiter_bpreshuffle_gemm = lambda n: n % 64 == 0; sys.modules["sglang.srt.layers.quantization.fp8_utils"] = F8
for kind, ple, i in (("gdn", True, 1), ("qsa", False, 3)):
    layer = T._layer(kind, ple=ple); real(layer)
    names = {}
    def walk(o, p):
        for k, v in vars(o).items():
            if isinstance(v, torch.Tensor): names[p + k] = v
            elif hasattr(v, "__dict__") and not callable(v): walk(v, p + k + ".")
    walk(layer, "")
    layer.named_parameters = lambda: names.items()
    copies = {}
    v = weights.layer_views(layer, i, kind, copies, set())
    print(kind, sorted(v), {k: tuple(t.shape) for k, t in copies.items() if "moe" not in str(k)} if kind == "qsa" else len(copies))
    print("ba/proj", [(k, tuple(v[k].shape)) for k in v if k in ("ip", "ba", "qp", "qp2", "out")])
