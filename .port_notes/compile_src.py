import sys, time
sys.path.insert(0, "/sgl-workspace/sglang/python")
from sglang.kernels.ops.speculative.qwen4_exp_monokernel import kernel as K
t = time.time()
launch = K.build_qwen4_exp_monokernel(("gdn",), 1, 4, 2048, 0, False, prelude=True, first_src="embed")
launch(0, 0, 0, 0, 0, 0)
print("compiled source", time.time() - t)
