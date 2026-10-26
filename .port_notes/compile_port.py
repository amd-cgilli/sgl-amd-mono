import sys, time
sys.path.insert(0, "/sgl-workspace/sgl-amd-mono")
from sgl_amd_mono.qwen38_flash_next.gfx950.config import MonoBuild
from sgl_amd_mono.qwen38_flash_next.gfx950.kernel import build_qwen38_flash_next_kernel
t = time.time()
kw = dict(plan="gdn", nreq=1, ntok=4, rtt_stride=2048, pos_stride=0, prelude=True, first_src="embed")
kw.update(eval("dict(" + (sys.argv[1] if len(sys.argv) > 1 else "") + ")"))
launch = build_qwen38_flash_next_kernel(MonoBuild(**kw))
launch(0, 0, 0, 0, 0, 0)
print("compiled port", time.time() - t)
