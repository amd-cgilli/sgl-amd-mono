"""Build a variant of the ported kernel (temporarily inside the package) and compile it on CPU."""
import os, shutil, subprocess, sys
PKG = "/sgl-workspace/sgl-amd-mono/sgl_amd_mono/qwen38_flash_next/gfx950"
name, toggles = sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else []
cfg = sys.argv[3] if len(sys.argv) > 3 else ""
s = open(f"{PKG}/kernel.py").read()
if "noguard" in toggles:
    s = s.replace("guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)", "None")
if "srcbo" in toggles:
    s = s.replace("from aiter.ops.flydsl.kernels import buffer_ops as bo\n", "import sys as _s; _s.path.insert(0, '/sgl-workspace/sglang/python')\nfrom sglang.kernels.ops.speculative.qwen4_exp_monokernel import buffer_ops as bo\n")
    s = s.replace("from sgl_amd_mono.common.ops import rsrc as _rsrc\n", "from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import rsrc as _rsrc0\ndef _rsrc(a, n=None):\n    return bo.create_buffer_resource_from_addr(a, num_records_bytes=n)\n")
if "dppumax" in toggles:
    s = s.replace("from sgl_amd_mono.common.ops import wave_umax as _wave_umax\n", "from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import wave_umax_dpp as _wave_umax\n")
if "lean" in toggles:
    s = s.replace("                v = spin_until(load_all, pending)\n", "                v = _lean_spin(load_all, pending)\n")
    s = s.replace("SOURCES = kernel_sources(__name__)\n", "SOURCES = kernel_sources(__name__)\nfrom .device import spin as _lean_spin\n")
if "nop" in toggles:
    # the source's unbounded spin, unguarded
    s = s.replace("                v = spin(load_all, pending)\n", "                v = load_all()\n                while pending(v):\n                    _spin_pause()\n                    v = load_all()\n")
    s = s.replace("SOURCES = kernel_sources(__name__)\n", "SOURCES = kernel_sources(__name__)\nfrom sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import spin_pause as _spin_pause\n")
mod = f"_variant_{name}"
open(f"{PKG}/{mod}.py", "w").write(s)
try:
    out = f"/sgl-workspace/sgl-amd-mono/.port_notes/asm_{name}"
    shutil.rmtree(out, ignore_errors=True)
    code = f"""
import sys; sys.path.insert(0, '/sgl-workspace/sgl-amd-mono'); sys.path.insert(0, '/sgl-workspace/sglang/python')
from sgl_amd_mono.qwen38_flash_next.gfx950.config import MonoBuild
from sgl_amd_mono.qwen38_flash_next.gfx950.{mod} import build_qwen38_flash_next_kernel as b
kw = dict(plan='gdn', nreq=1, ntok=4, rtt_stride=2048, pos_stride=0, prelude=True, first_src='embed')
kw.update(dict({cfg}))
b(MonoBuild(**kw))(0, 0, 0, 0, 0, 0)
"""
    env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=out)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        sys.exit(1)
    isa = [os.path.join(out, d, "22_final_isa.s") for d in os.listdir(out)][0]
    t = open(isa).read()
    import re
    g = lambda k: re.search(rf"\.{k}:\s+(\d+)", t).group(1)
    n = len(re.findall(r"^\s+(?:[sv]_|buffer_|global_|ds_|flat_)", t, re.M))
    print(f"{name}: vgpr {g('vgpr_count')} sgpr {g('sgpr_count')} scratch {g('private_segment_fixed_size')} instrs {n}")
finally:
    os.remove(f"{PKG}/{mod}.py")
