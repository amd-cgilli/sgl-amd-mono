"""Compile one launch config from the source kernel and the port on CPU; print ISA metrics."""
import os, re, shutil, subprocess, sys
name, src_call, port_kw = sys.argv[1], sys.argv[2], sys.argv[3]
def run(tag, code):
    out = f"/sgl-workspace/sgl-amd-mono/.port_notes/cmp/{name}_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=out)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode:
        return f"{tag}: FAILED {r.stderr[-1500:]}"
    isa = [os.path.join(out, d, "22_final_isa.s") for d in os.listdir(out)][0]
    t = open(isa).read()
    g = lambda k: re.search(rf"\.{k}:\s+(\d+)", t).group(1)
    n = len(re.findall(r"^\s+(?:[sv]_|buffer_|global_|ds_|flat_)", t, re.M))
    # keep only the ISA for later diffing
    keep = f"/sgl-workspace/sgl-amd-mono/.port_notes/cmp/{name}_{tag}.s"
    shutil.copy(isa, keep); shutil.rmtree(out)
    return f"{tag}: vgpr {g('vgpr_count')} sgpr {g('sgpr_count')} lds {g('group_segment_fixed_size')} scratch {g('private_segment_fixed_size')} instrs {n}"
src = f"""import sys; sys.path.insert(0, '/sgl-workspace/sglang/python')
from sglang.kernels.ops.speculative.qwen4_exp_monokernel import kernel as K
K.build_qwen4_exp_monokernel({src_call})(0, 0, 0, 0, 0, 0)"""
port = f"""import sys; sys.path.insert(0, '/sgl-workspace/sgl-amd-mono')
from sgl_amd_mono.qwen38_flash_next.gfx950.config import MonoBuild
from sgl_amd_mono.qwen38_flash_next.gfx950.kernel import build_qwen38_flash_next_kernel as b
b(MonoBuild({port_kw}))(0, 0, 0, 0, 0, 0)"""
os.makedirs("/sgl-workspace/sgl-amd-mono/.port_notes/cmp", exist_ok=True)
print(name, "|", run("src", src), "|", run("port", port), flush=True)
