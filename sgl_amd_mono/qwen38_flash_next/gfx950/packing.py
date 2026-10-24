# SPDX-License-Identifier: Apache-2.0
"""Host-side packing and pointer tables of the Qwen3.8-Flash-Next launches.

The MXFP4 experts and the FP8 attention / GDN projections are SGLang's own params
(AITER layouts), read in place; only the small bf16 tensors SGLang does not keep in
the kernel's tile layout (hyper-connection mixers, router, PLE, MTP input fusion)
are packed here. Plain torch: the kernel module does not import it, so it stays out
of the kernels' cache key."""

import torch

from . import layout as LY
from .config import HIDDEN, N_EXPERTS


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """Pack BF16 ``[N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""
    rows, k = w.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"BF16 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w16 = w.view(torch.int16).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)


def check_moe_params(m):
    """Shape / dtype / layout asserts on SGLang's MXFP4 MoE params the kernel reads."""
    E, IP = N_EXPERTS + 1, LY.MOE_INTER_PAD
    shapes = {
        "gu": (E, 2 * IP, HIDDEN // 2),
        "dn": (E, HIDDEN, IP // 2),
        "gu_s": (E, 2 * IP, LY.GU_SGROUPS),
        "dn_s": (E, HIDDEN, LY.DN_SGROUPS),
    }
    for k, shape in shapes.items():
        assert tuple(m[k].shape) == shape, (k, m[k].shape)
        assert m[k].is_contiguous() and m[k].element_size() == 1, k
    assert m["gu"].dtype == torch.uint8 and m["dn"].dtype == torch.uint8


def pack_hc_down(w_down, w_inj=None):
    """Mix down weights [320, HC_DIM] + one extra 16-row tile whose rows 0..3 are the
    combine-gate (inject) rows: the down GEMM of a K-slice also yields the gate
    partials."""
    extra = torch.zeros(16, w_down.shape[1], dtype=w_down.dtype, device=w_down.device)
    if w_inj is not None:
        extra[: w_inj.shape[0]] = w_inj
    return pack_bf16(torch.cat([w_down, extra], dim=0).contiguous())


def pack_mixer(norm, down, up, inj=None) -> dict:
    """A hyper-connection mixer's norm, down (+ combine gate) and up weights."""
    out = {
        "norm": norm.contiguous(),
        "down": pack_hc_down(down, inj),
        "up": pack_bf16(up.contiguous()),
    }
    if inj is not None:
        out["inj"] = inj.contiguous()
    return out


def _pack_hc(lw, name):
    names = ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")  # fmt: skip
    return pack_mixer(*(lw[f"{name}.{w}.weight"] for w in names))


def _pack_common(lw, moe, device) -> dict:
    """What both layer kinds pack: the two mixers, the router (+ the shared expert's
    gate and 15 zero rows) and SGLang's FusedMoE params (read in place)."""
    router = torch.cat(
        [
            lw["mlp.gate.weight"],
            lw["mlp.shared_expert_gate.weight"],
            torch.zeros(15, HIDDEN, dtype=torch.bfloat16, device=device),
        ],
        dim=0,
    )
    return {
        "hca": _pack_hc(lw, "attn_hyper_connection"),
        "hcm": _pack_hc(lw, "mlp_hyper_connection"),
        "router": pack_bf16(router.contiguous()),
        "moe": moe,
    }


def pack_gdn_layer(lw, moe, proj):
    """moe / proj: SGLang's FusedMoE params / FP8 projections {"ip", "out", "ba":
    (weight, scale)} of the layer (AITER layouts, read in place); the b/a rows stay
    row-major FP8 (the GDN core dots them with the per-token FP8 activation)."""
    a = "linear_attn."
    t = _pack_common(lw, moe, proj["ip"][0].device)
    for k in ("ip", "out", "ba"):
        t[k], t[k + "_s"] = proj[k]
    t["conv_w"] = lw[a + "conv1d.weight"].squeeze(1).contiguous()
    t["neg_a"] = (-torch.exp(lw[a + "A_log"].float())).contiguous()
    t["dt_bias"] = lw[a + "dt_bias"].float().contiguous()
    t["norm_w"] = lw[a + "norm.weight"].float().contiguous()
    return t


def pack_qsa_layer(lw, moe, proj):
    """proj: SGLang's FP8 params {"qp": qkv_proj, "qp2": indexer.index_qk_proj, "out":
    o_proj} as (weight, scale)."""
    a = "self_attn."
    t = _pack_common(lw, moe, proj["qp"][0].device)
    for k in ("qp", "qp2", "out"):
        t[k], t[k + "_s"] = proj[k]
    for k, name in (
        ("qnorm", "q_norm"),
        ("knorm", "k_norm"),
        ("iqnorm", "indexer.q_layernorm"),
        ("iknorm", "indexer.k_layernorm"),
    ):
        t[k] = lw[f"{a}{name}.weight"].contiguous()
    return t


def pack_ple(lw):
    """PLE weights of layer 1: packed [key_proj; value_proj] + norms + dilated conv weight."""
    p = "ple."
    w = torch.cat([lw[p + "key_proj.weight"], lw[p + "value_proj.weight"]], dim=0)
    return {
        "w": pack_bf16(w.contiguous()),
        "nk": lw[p + "norm_key.weight"].contiguous(),
        "nq": lw[p + "norm_query.weight"].contiguous(),
        "nc": lw[p + "norm_conv.weight"].contiguous(),
        "cw": lw[p + "conv1d.weight"].squeeze(1).contiguous(),
    }


def pack_fusion(lw):
    """The MTP draft's input fusion: the two pre-fc norms and the fc projections."""
    return {
        "pe": lw["pre_fc_norm_embedding.weight"].contiguous(),
        "ph": lw["pre_fc_norm_hidden.weight"].contiguous(),
        "fce": pack_bf16(lw["fc_embedding.weight"].contiguous()),
        "fch": pack_bf16(lw["fc_hidden.weight"].contiguous()),
    }


# ---------------------------------------------------------------- pointer rows
def _set(row, t, slots):
    for slot, name in slots:
        row[slot] = t[name].data_ptr()


_MIXER = (("norm", 0), ("down", 1), ("up", 2), ("inj", 3))
_MOE = ((LY.P_GU, "gu"), (LY.P_GU_S, "gu_s"), (LY.P_DN, "dn"), (LY.P_DN_S, "dn_s"))


def _common_row(t) -> list:
    """A layer row's mixers, router and experts."""
    row = [0] * LY.N_PTR
    _set(row, t["hca"], [(LY.P_HCA_NORM + i, k) for k, i in _MIXER])
    _set(row, t["hcm"], [(LY.P_HCM_NORM + i, k) for k, i in _MIXER])
    _set(row, t, [(LY.P_ROUTER, "router")])
    _set(row, t["moe"], _MOE)
    return row


def gdn_ptr_row(t, conv_state, ssm_state):
    """Pointer table row for one GDN layer; conv/ssm state = base of the [slots, ...] pools of this layer."""
    row = _common_row(t)
    _set(row, t, [(LY.P_IP, "ip"), (LY.P_CONV_W, "conv_w"), (LY.P_NEG_A, "neg_a")])
    _set(
        row, t, [(LY.P_DT_BIAS, "dt_bias"), (LY.P_NORM_W, "norm_w"), (LY.P_OUT, "out")]
    )
    _set(row, t, [(LY.P_BA, "ba"), (LY.P_IP_S, "ip_s"), (LY.P_OUT_S, "out_s")])
    _set(row, t, [(LY.P_BA_S, "ba_s")])
    row[LY.P_CONV_STATE] = conv_state.data_ptr()
    row[LY.P_SSM_STATE] = ssm_state.data_ptr()
    return row


def qsa_ptr_row(t, k_buf, v_buf, ring, comp):
    """Pointer table row for one QSA layer (this layer's KV / index-ring / compressed buffers)."""
    for b in (k_buf, v_buf):  # addressed with 64-bit pointers (any pool size)
        assert b.is_contiguous() and b.shape[-2:] == (2, 256)
    for b in (ring, comp):
        assert b.is_contiguous() and b.numel() * b.element_size() < 2**32, (
            "buffer offsets are 32-bit"
        )
    row = _common_row(t)
    _set(row, t, [(LY.P_IP, "qp"), (LY.P_OUT, "out"), (LY.P_IP_S, "qp_s")])
    _set(row, t, [(LY.P_IP2, "qp2"), (LY.P_IP2_S, "qp2_s"), (LY.P_OUT_S, "out_s")])
    _set(row, t, [(LY.P_QNORM, "qnorm"), (LY.P_KNORM, "knorm")])
    _set(row, t, [(LY.P_IQNORM, "iqnorm"), (LY.P_IKNORM, "iknorm")])
    for slot, b in ((LY.P_KBUF, k_buf), (LY.P_VBUF, v_buf), (LY.P_RING, ring), (LY.P_COMP, comp)):  # fmt: skip
        row[slot] = b.data_ptr()
    return row


def final_row(mixer, inj) -> list:
    """The model-level mixer's row (``inj``: any layer's combine gate, unused)."""
    row = [0] * LY.N_PTR
    _set(row, mixer, [(LY.P_HCA_NORM + i, k) for k, i in _MIXER[:3]])
    row[LY.P_HCA_INJ] = inj.data_ptr()
    return row


def global_ptr_row(fb, req_to_token, cos_sin, rope_pos, mamba_map):
    """The global row's step tensors (forward batch ``fb``) and SGLang's tables."""
    assert cos_sin.dtype == torch.bfloat16 and req_to_token.dtype == torch.int32
    row = [0] * LY.N_PTR
    row[LY.G_MAMBA_MAP] = mamba_map.data_ptr()
    row[LY.G_POS] = fb.positions.data_ptr()
    row[LY.G_LOC] = fb.out_cache_loc.data_ptr()
    row[LY.G_REQ] = fb.req_pool_indices.data_ptr()
    row[LY.G_SEQ] = fb.seq_lens.data_ptr()
    row[LY.G_RTT] = req_to_token.data_ptr()
    row[LY.G_COS] = cos_sin.data_ptr()
    row[LY.G_ROPEPOS] = rope_pos.data_ptr()
    return row


def add_ple_ptrs(row, t, state):
    """Add PLE pointers (and the [slots, 10240, 9] short-conv state base) to a layer row."""
    assert (
        state.dtype == torch.bfloat16
        and state.is_contiguous()
        and state[0].numel() * 2 == LY.PLE_STATE_SLOT_BYTES
    )
    slots = (LY.P_PLE_W, LY.P_PLE_NK, LY.P_PLE_NQ, LY.P_PLE_NC, LY.P_PLE_CW)
    _set(row, t, zip(slots, ("w", "nk", "nq", "nc", "cw")))
    row[LY.P_PLE_STATE] = state.data_ptr()
    return row


def mtp_rows(m, k_buf, v_buf, ring, comp, grow):
    """Pointer rows of a draft launch: [QSA layer row, global row (+ fusion pointers), final-mixer row]."""
    row = qsa_ptr_row(m["layer"], k_buf, v_buf, ring, comp)
    _set(grow, m, ((LY.G_MTP_PE, "pe"), (LY.G_MTP_PH, "ph")))
    _set(grow, m, ((LY.G_MTP_FCE, "fce"), (LY.G_MTP_FCH, "fch")))
    return [row, grow, final_row(m["final"], m["layer"]["hca"]["inj"])]


# ---------------------------------------------------------------- check mode
def unshuffle16(w):
    """AITER shuffle_weight(w, (16, 16)) inverse for 1-byte elements: [N, K] storage -> row-major [N, K] bytes."""
    w = w.view(torch.uint8)
    rows, kb = w.shape
    w = w.reshape(rows // 16, kb // 32, 2, 16, 16).permute(0, 3, 1, 2, 4)
    return w.reshape(rows, kb)


def _unshuffle_e8m0(s):
    """e8m0_shuffle inverse: [a, d, f, c, e, b] -> [32 a + 16 b + c, 8 d + 4 e + f]."""
    rows, g = s.shape
    s = s.reshape(rows // 32, g // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2)
    return s.reshape(rows, g)


def unshuffle_moe_expert(m, e):
    """Expert e of the AITER-layout MoE params back to the checkpoint layout, real rows / K only:
    -> (gate [640, 1280], up [640, 1280], down [2560, 320], gate_s [640, 80], up_s [640, 80], down_s [2560, 20])."""
    I, IP = LY.MOE_INTER, LY.MOE_INTER_PAD
    gu, gus = unshuffle16(m["gu"][e]), _unshuffle_e8m0(m["gu_s"][e])
    dn, dns = unshuffle16(m["dn"][e]), _unshuffle_e8m0(m["dn_s"][e])
    return gu[:I], gu[IP : IP + I], dn[:, : I // 2], gus[:I], gus[IP : IP + I], dns[:, : I // 32]  # fmt: skip
