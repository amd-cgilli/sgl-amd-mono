"""Mechanical port of the source kernel.py into sgl_amd_mono (run once; see NOT_VERIFIED.md)."""
import re
import sys

SRC = "/sgl-workspace/sglang/python/sglang/kernels/ops/speculative/qwen4_exp_monokernel/kernel.py"
DST = "/sgl-workspace/sgl-amd-mono/sgl_amd_mono/qwen38_flash_next/gfx950/kernel.py"
s = open(SRC).read()


def sub1(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"expected {count} of {old[:80]!r}, found {n}")
    s = s.replace(old, new)


# ---- 1. header, docstring and imports
start = s.index("POLL_MAX = 24")
s = '''# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang#45353 at 6ce98069c3 (Apache-2.0), by salykova:
# python/sglang/kernels/ops/speculative/qwen4_exp_monokernel/kernel.py
# ruff: noqa: B008, B023, E501, SIM102
"""Qwen3.8-Flash-Next (SGLang's Qwen4-Exp) monokernel: the decoder stack of a
speculative verify (and the MTP draft) of R requests x L tokens.

One persistent launch of 256 CTAs x 512 threads runs whole decoder layers (3 GDN +
1 QSA per launch, 13 launches a verify)::

    [prelude: embedding + n-gram history]  [PLE of layer 1]
    per layer: mix a -> GDN (in_proj -> conv + gated delta rule) | QSA (proj -> prep
      -> scores -> select -> merge select -> attention -> merge) -> out_proj
      -> mix m -> router -> MXFP4 gate / up -> down (+ the shared expert)
    [final mix: the model-level hyper-connection mixer]

R requests x L tokens = S rows (compile-time, S <= 32; row t = request t // L,
draft step t % L):

* FP8 attention / GDN projections (W8A8): ``mfma_scale_f32_16x16x128_f8f6f4`` on
  SGLang's AITER-shuffled FP8 weights (per-channel scales) and per-token dynamic FP8
  activations (AITER's quantization), with the S rows as MFMA N-columns in
  NCT = ceil(S / 16) column tiles that share the weight registers;
* MXFP4 experts read in place from SGLang's FusedMoE params (AITER layout, the
  shared expert fused at index 512);
* bf16 skinny GEMMs (``mfma_f32_16x16x32_bf16``) for the hyper-connection mixers,
  router, PLE and MTP input fusion;
* per-request state: positions / RoPE, KV locations (req_to_token row), mamba slot,
  verify-scratch row (row r of intermediate_ssm / intermediate_conv_window / PLE
  intermediate state / n-gram context);
* GDN: the gated delta rule runs each request's L tokens in order on its own
  register-resident state and writes the state / conv window after every token
  into scratch row r;
* QSA: per-row causal sparse attention inside its request (row t sees its request's
  positions <= pos_t, its own top-512 blocks); index-key ring / compression per
  request; two-level exact top-512 per row;
* MoE: per-row top-10, the union of the routed experts of all rows (+ shared) is
  computed once; only the routed (row, expert) pairs are handed to the down
  projection.

Every stage is a list of tasks; a task runs on CTA (base + t) % 256 and every CTA
walks the stages in order. There is no grid-wide barrier: dependencies only point
to earlier stages and all CTAs are co-resident, so every wait makes progress.
Stages hand values over through tagged-pair mailboxes in a global scratch buffer,
tagged ``epoch * LAUNCH_SLOTS + launch + 1``; each wait runs under the guard
(``common/guard.py``): past its deadline it stops the launch (TP 1: a failure stays
on this GPU), so no step runs on with words that never arrived.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf2_f32,
    bf16_pair,
    butterfly,
    fresh,
    global_load,
    load_ptr64,
)
from sgl_amd_mono.common.ops import bf16_round as bf_round
from sgl_amd_mono.common.ops import fp8_roundtrip as _fp8_roundtrip
from sgl_amd_mono.common.ops import hw_exp as _exp
from sgl_amd_mono.common.ops import hw_rcp as _rcp
from sgl_amd_mono.common.ops import hw_rsq as _rsq
from sgl_amd_mono.common.ops import mxfp4_to_bf16x8 as _mxfp4_to_bf16x8
from sgl_amd_mono.common.ops import rsrc as _rsrc
from sgl_amd_mono.common.ops import uniform as _uniform
from sgl_amd_mono.common.ops import wave_umax as _wave_umax
from sgl_amd_mono.common.ops import xred as _xred
from sgl_amd_mono.common.ops import xshfl as _xshfl
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, kernel_sources, key_tuple

from . import layout as LY
from .device import spin
from .config import (
    EPS,
    GDN_DK,
    GDN_HV,
    GDN_KEY_DIM,
    GDN_VAL_DIM,
    HC,
    HC_DIM,
    HC_LOWRANK,
    HIDDEN,
    MAX_R,
    MAX_T,
    MOE_INTER,
    N_EXPERTS,
    TOPK,
    MonoBuild,
)
from .layout import scratch_layout

SOURCES = kernel_sources(__name__)

''' + s[start:]

# ---- 2. module-level helpers now in common/ops (identical code: fresh, global_load)
a = s.index("def _opaque_i32(v):")
b = s.index("def _ordered_i32(v):")
s = s[:a] + s[b:]
a = s.index("def _gload_v4(addr):")
b = s.index("def _log2(v):")
s = s[:a] + s[b:]
sub1("TL_LABELS = []\n\n\n", "")

# ---- 3. the builder: one MonoBuild key, keyed with the sources digest
a = s.index("@functools.cache\ndef build_qwen4_exp_monokernel(")
b = s.index("    R, L = nreq, ntok\n")
s = s[:a] + '''def build_qwen38_flash_next_kernel(key: MonoBuild):
    """Return the ``@flyc.jit`` launcher of one launch (``MonoBuild``): the layers
    of ``key.plan`` for R = ``key.nreq`` requests x L = ``key.ntok`` tokens. The
    pointer table has one row per layer plus a global row (and the final mixer's
    row after it in a ``final_mix`` launch)."""
    keyed = key_tuple(key, SOURCES)
    plan, nreq, ntok = key.layers, key.nreq, key.ntok
    rtt_stride, pos_stride, map_i64 = key.rtt_stride, key.pos_stride, key.map_i64
    ple_first, final_mix, prelude = key.ple_first, key.final_mix, key.prelude
    eos_token_id, first_src, ssm_bf16 = key.eos_token_id, key.first_src, key.ssm_bf16
    issm_row, ple_ist_row, ictx_row = key.issm_row, key.ple_ist_row, key.ictx_row
    iconv_strides = (key.iconv.step, key.iconv.channel, key.iconv.window, key.iconv.row)
    fault_site, fault_after = key.fault_site, key.fault_after
''' + s[b:]
sub1("    assert 1 <= R <= LY.MAX_R and 1 <= NT <= LY.MAX_T\n", "    assert 1 <= R <= MAX_R and 1 <= NT <= MAX_T\n")

# ---- 4. the kernel: symbol, keyed, guard scope over the whole body
a = s.index("    @flyc.kernel(known_block_size=[THREADS, 1, 1])\n    def qwen4_exp_monokernel(")
b = s.index("        tid = fx.thread_idx.x\n        bid = fx.block_idx.x\n", a)
c = s.index("    @flyc.jit\n    def launch(")
body = s[b:c]
body = "".join(("    " + ln if ln.strip() else ln) for ln in body.splitlines(keepends=True))
s = s[:a] + '''    kernel_name = kernel_symbol(
        "qwen38_flash_next_layers",
        s=NT,
        r=R,
        p="".join(k[0] for k in plan),
        f=int(final_mix),
        b=int(ssm_bf16),
    )

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def qwen38_flash_next_kernel(
        h_in: Int64,
        h_out: Int64,
        ptab: Int64,
        scratch: Int64,
        epoch: Int64,
        launch_id: Int32,
    ):
        _ = keyed
        # every wait of this launch runs under the guard (deadline, error words);
        # TP 1: no peers, a failure stops this GPU's launch
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
''' + body + s[c:]

# ---- 5. the tag: word 0 of the guard block is the step epoch
sub1(
    "            step_value = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32))\n"
    "            tag = step_value * LY.LAUNCH_SLOTS + launch_id + 1\n",
    "            # this launch's tag: word 0 of the guard block is the step epoch, which\n"
    "            # the host bumps after a step's last launch\n"
    "            step_value = _uniform(\n"
    "                bo.buffer_load(_rsrc(epoch), 0, vec_width=1, dtype=T.i32)\n"
    "            )\n"
    "            tag = step_value * LY.LAUNCH_SLOTS + launch_id + 1\n",
)
sub1("            r_ptab = _rsrc(ptab)\n", "")
a = s.index("            def ptr(layer, slot):\n")
b = s.index("            def rsrc_if(addr, live, nbytes):\n")
s = s[:a] + '''            def ptr(layer, slot):
                return load_ptr64(ptab, layer * LY.N_PTR + slot)

''' + s[b:]

sub1(
    "                return bo.create_buffer_resource_from_addr(\n"
    "                    addr, num_records_bytes=live.select(fx.Int32(nbytes), fx.Int32(0))\n"
    "                )\n",
    "                # aiter's descriptor takes an i64 byte count (every size here is < 2^31)\n"
    "                return _rsrc(addr, fx.Int64(fx.Int32(live.select(fx.Int32(nbytes), fx.Int32(0)))))\n",
)

# ---- 6. in-kernel helpers now in common/ops (identical code)
a = s.index("            def bf16_pair(a, b):\n")
b = s.index("            def sigmoid(v):\n")
s = s[:a] + s[b:]
sub1(
    "                v = load_all()\n"
    "                while pending(v):\n"
    "                    spin_pause()\n"
    "                    v = load_all()\n",
    "                # under the guard: a deadline, checked every CHECK_POLLS polls\n"
    "                v = spin(load_all, pending)\n",
)
sub1(
    "            def wave_sum(v):\n"
    "                for sh in range_constexpr(6):\n"
    "                    v = _xred(v, 32 >> sh, lambda a, b: a + b)\n"
    "                return v\n",
    "            def wave_sum(v):\n"
    "                return butterfly(v, (32, 16, 8, 4, 2, 1))\n",
)

# ---- 7. the debug timeline (``timeline`` was False in every build): gone
a = s.index("            tl_count = [0]\n")
b = s.index("            def rtt_at(rtt_row, tok):\n")
s = s[:a] + s[b:]
s, n = re.subn(r"\n[ ]*stamp\((?:f?\"[^\"]*\")?\)", "", s)
print("stamps removed", n)
assert "stamp(" not in s, [ln for ln in s.splitlines() if "stamp(" in ln]

# ---- 8. helpers renamed
n_op = s.count("_opaque_i32(")
s = s.replace("_opaque_i32(", "fresh(")
print("fresh", n_op)
n = 0
while "_gload_v4(" in s:
    a = s.index("_gload_v4(")
    i, depth = a + len("_gload_v4("), 1
    while depth:
        depth += {"(": 1, ")": -1}.get(s[i], 0)
        i += 1
    s = s[:a] + "fx.Vector(global_load(" + s[a + len("_gload_v4("): i - 1] + ", 4))" + s[i:]
    n += 1
print("global_load", n)
assert "_gload_v4" not in s

# ---- 9. the launcher
a = s.index("    @flyc.jit\n    def launch(")
s = s[:a] + '''    @flyc.jit
    def launch(
        h_in: Int64,
        h_out: Int64,
        ptab: Int64,
        scratch: Int64,
        epoch: Int64,
        launch_id: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        qwen38_flash_next_kernel(h_in, h_out, ptab, scratch, epoch, launch_id).launch(
            grid=(G,), block=(THREADS,), stream=stream
        )

    return launch
'''
for gone in ("spin_pause", "mem_realtime", "TL_LABELS", "G_TIMELINE", "timeline", "\bstep\b"):
    if re.search(gone, s):
        print("left:", gone, [ln for ln in s.splitlines() if re.search(gone, ln)][:5])
open(DST, "w").write(s)
