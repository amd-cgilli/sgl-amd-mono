# SPDX-License-Identifier: Apache-2.0
"""The first 12 layers (launches [0], [1-3] with the PLE, [4-7], [8-11] with the
final mixer) and the MTP head of the Qwen3.8-Flash-Next checkpoint
(QWEN38_FLASH_NEXT_MODEL_PATH, a directory or a Hugging Face id in the local cache)
in SGLang's post-load gfx950 layouts, and pools of SGLang's shapes with random state.
The PLE n-gram table (51 GB) is one random FP8 table every head shares."""

import json
import os
import re
from functools import lru_cache
from types import SimpleNamespace

import torch

from sgl_amd_mono.qwen38_flash_next.gfx950 import layout as LY
from sgl_amd_mono.qwen38_flash_next.gfx950 import packing
from sgl_amd_mono.qwen38_flash_next.gfx950.config import (
    GDN_CONV_DIM,
    HC_DIM,
    MOE_INTER,
    N_EXPERTS,
)

MODEL = os.environ.get(
    "QWEN38_FLASH_NEXT_MODEL_PATH", "amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8"
)
PFX = "model.language_model."
L = 4  # verify rows a request (3 draft steps + 1)
MAX_R = 8
# request r's context: every context % 4, past one 8192-block index segment, and a
# step crossing the first segment boundary
CONTEXTS = (60, 1203, 9000, 70001, 32766, 2048, 140003, 60)
# request r's KV slots start at BASES[r]
BASES = [64 + sum((c + L + 127) // 64 * 64 for c in CONTEXTS[:r]) for r in range(MAX_R)]
DEV = "cuda"


def gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(seed)


@lru_cache(maxsize=1)
def checkpoint_dir():
    """The checkpoint's local directory, or None (nothing is downloaded)."""
    if os.path.isdir(MODEL):
        return MODEL
    from huggingface_hub import try_to_load_from_cache

    index = try_to_load_from_cache(MODEL, "model.safetensors.index.json")
    return os.path.dirname(index) if isinstance(index, str) else None


@lru_cache(maxsize=1)
def _index() -> dict:
    with open(os.path.join(checkpoint_dir(), "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


@lru_cache(maxsize=1)
def eos_token_id() -> int:
    with open(os.path.join(checkpoint_dir(), "config.json")) as f:
        cfg = json.load(f)
    return int(cfg.get("text_config", cfg)["eos_token_id"])


def load(names) -> dict:
    from safetensors import safe_open

    by_file = {}
    for n in names:
        by_file.setdefault(_index()[n], []).append(n)
    out = {}
    for fname, ns in by_file.items():
        with safe_open(os.path.join(checkpoint_dir(), fname), "pt", device="cpu") as f:
            out |= {n: f.get_tensor(n).to(DEV) for n in ns}
    return out


# the experts, the attention projections and the n-gram table
BIG = re.compile(
    r"\.experts\.|\.shared_expert\.|ple_embedding\.|attn\.(indexer\.)?\w*proj"
)


def small(p: str) -> dict:
    """The tensors under ``p`` but the big ones, by their names under ``p``."""
    names = [k for k in _index() if k.startswith(p) and not BIG.search(k)]
    return {k[len(p) :]: v for k, v in load(names).items()}


def sglang_moe(p: str) -> dict:
    """FusedMoE's MXFP4 params as SGLang keeps them on gfx950 (AITER): the routed
    experts and the shared one (index 512) stacked, the intermediate dim padded 640
    -> 768, ``shuffle_weight(w, (16, 16))`` and ``e8m0_shuffle``'d scales."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    I, IP = MOE_INTER, LY.MOE_INTER_PAD
    srcs = [f"{p}mlp.experts.{e}." for e in range(N_EXPERTS)] + [f"{p}mlp.shared_expert."]  # fmt: skip
    projs = ("gate_proj", "up_proj", "down_proj")
    raw = load([s + f"{j}.{w}" for s in srcs for j in projs for w in ("weight", "weight_scale")])  # fmt: skip

    def stack(proj, w, pad):
        t = torch.stack([raw[s + f"{proj}.{w}"].view(torch.uint8) for s in srcs])
        return torch.nn.functional.pad(t, pad)

    def gate_up(w):  # gate rows 0 .. 639, up rows 768 .. 1407
        pad = (0, 0, 0, IP - I)
        return torch.cat([stack("gate_proj", w, pad), stack("up_proj", w, pad)], 1)

    def down(w, n):
        return stack("down_proj", w, (0, n - raw[srcs[0] + f"down_proj.{w}"].shape[-1]))

    def scales(s):
        e, r, _ = s.shape
        return e8m0_shuffle(s.reshape(e * r, -1).contiguous()).view(e, r, -1)

    return {
        "gu": shuffle_weight(gate_up("weight"), (16, 16)),
        "gu_s": scales(gate_up("weight_scale")),
        "dn": shuffle_weight(down("weight", IP // 2), (16, 16)),
        "dn_s": scales(down("weight_scale", IP // 32)),
    }


def sglang_fp8(p: str, names, rows: int):
    """A merged Quark FP8 projection as SGLang keeps it: ``rows`` rows (zero-padded,
    scale 1), (16, 16)-shuffled, f32 per-channel scales."""
    from aiter.ops.shuffle import shuffle_weight

    raw = load([f"{p}{n}.{w}" for n in names for w in ("weight", "weight_scale")])
    w = torch.cat([raw[f"{p}{n}.weight"] for n in names])
    s = torch.cat([raw[f"{p}{n}.weight_scale"].float().reshape(-1) for n in names])
    pad = rows - w.shape[0]
    w = torch.cat([w, torch.zeros(pad, w.shape[1], dtype=w.dtype, device=DEV)])
    s = torch.cat([s, torch.ones(pad, device=DEV)])
    return shuffle_weight(w.contiguous(), (16, 16)), s


# SGLang's merged FP8 projections: (checkpoint names, padded rows)
# fmt: off
PROJ = {
    "gdn": {"ip": ("in_proj_qkv in_proj_z", 16384), "out": ("out_proj", 2560),
            "ba": ("in_proj_b in_proj_a", LY.BA_ROWS)},
    "qsa": {"qp": ("q_proj k_proj v_proj", 13312), "qp2": ("indexer.index_qk_proj", 640),
            "out": ("o_proj", 2560)},
}
# fmt: on


def sglang_proj(p: str, kind: str) -> dict:
    a = p + ("linear_attn." if kind == "gdn" else "self_attn.")
    return {k: sglang_fp8(a, n.split(), rows) for k, (n, rows) in PROJ[kind].items()}


def ngram_tables() -> dict:
    p = f"{PFX}layers.1.ple.ple_embedding."
    raw = load([p + n for n in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_embedding.weight_scale")])  # fmt: skip
    vocab = raw[p + "ngram_heads_vocab_sizes"].long()
    meta = torch.cat([raw[p + "layer_multipliers"].long(), vocab, torch.zeros_like(vocab)])  # fmt: skip
    table = torch.randn(int(vocab.max()), LY.NGRAM_DIM, generator=gen(7), device=DEV)
    return {
        "table": (table * 32).to(torch.float8_e4m3fn),
        "scale": raw[p + "ngram_embedding.weight_scale"].bfloat16().contiguous(),
        "meta": meta.contiguous(),
    }


@lru_cache(maxsize=1)
def cos_sin() -> torch.Tensor:
    """SGLang's bf16 RoPE table [262144, 64] (theta 1e7, 64 rotated dims)."""
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, device=DEV).float() / 64))
    f = torch.outer(torch.arange(262144, device=DEV, dtype=torch.float), inv)
    return torch.cat([f.cos(), f.sin()], -1).bfloat16().contiguous()


@lru_cache(maxsize=1)
def embedding() -> torch.Tensor:
    return load([f"{PFX}embed_tokens.weight"])[f"{PFX}embed_tokens.weight"]


def target() -> SimpleNamespace:
    """Layers 0-11 and the model-level mixer, and a ``VerifyRunner`` on them;
    ``small``: each layer's small checkpoint tensors by their names in the layer."""
    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import VerifyRunner

    kinds = ["qsa" if i % 4 == 3 else "gdn" for i in range(12)]
    layers, small_w = {}, {}
    for i, kind in enumerate(kinds):
        p = f"{PFX}layers.{i}."
        lw = small_w[i] = small(p)
        pack = packing.pack_gdn_layer if kind == "gdn" else packing.pack_qsa_layer
        layers[i] = pack(lw, moe=sglang_moe(p), proj=sglang_proj(p, kind))
        if i == 1:
            layers[i]["ple"] = packing.pack_ple(lw)
    mix = small(f"{PFX}hyper_connection_mixer.")
    final = {
        "norm": mix["hc_norm.weight"].contiguous(),
        "down": packing.pack_hc_down(mix["input_mix_weight_down.weight"]),
        "up": packing.pack_bf16(mix["input_mix_weight_up.weight"].contiguous()),
    }
    ngram = ngram_tables()
    runner = VerifyRunner(kinds=kinds, layers=layers, final=final, ngram=ngram,
                          embed_w=embedding(), cos_sin=cos_sin, ntok=L, max_reqs=MAX_R,
                          eos_token_id=eos_token_id(), device=DEV)  # fmt: skip
    runner.set_deadline(30.0)
    return SimpleNamespace(kinds=kinds, layers=layers, small=small_w, ngram=ngram, runner=runner)  # fmt: skip


def draft() -> SimpleNamespace:
    """The MTP head (``mtp.*``) and a ``DraftRunner`` on it; ``small``: its small
    checkpoint tensors by their names under ``mtp.``."""
    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import DraftRunner

    lw, p = small("mtp."), "mtp.layers.0."
    views = packing.pack_mtp(lw, moe=sglang_moe(p), proj=sglang_proj(p, "qsa"))
    runner = DraftRunner(views=views, embed_w=embedding(), cos_sin=cos_sin, layer_id=0,
                         ntok_extend=L, max_reqs=MAX_R, eos_token_id=eos_token_id(),
                         device=DEV)  # fmt: skip
    runner.set_deadline(30.0)
    return SimpleNamespace(views=views, small=lw, runner=runner)


def _compile(key) -> None:
    from sgl_amd_mono.qwen38_flash_next.gfx950 import kernel, runner

    runner.ensure_writable_flydsl_cache()
    runner.compile_only(kernel.build_qwen38_flash_next_kernel(key))


def precompile(keys) -> None:
    """Compile ``keys`` in 16 processes (FlyDSL's compile-only mode, to its cache)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    keys = list(dict.fromkeys(keys))
    with ProcessPoolExecutor(min(16, len(keys)), mp_context=get_context("spawn")) as ex:
        list(ex.map(_compile, keys))


class Pools:
    """SGLang's req-to-token and KV pools in one, for ``MAX_R`` requests over
    ``CONTEXTS``: request r at request slot 2 r + 3, mamba slot 5 MAX_R - 2 r and KV
    slots BASES[r] + position; the verify scratch rows (request r owns row r) and the
    deduplicated intermediate conv window SGLang allocates on ROCm."""

    def __init__(self, kinds, seed: int = 0):
        g = gen(seed)
        R = MAX_R
        self.reqs = [2 * r + 3 for r in range(R)]
        n_req, n_slot, n_scr = max(self.reqs) + 2, 5 * R + 2, R + 1
        off = BASES[-1] + (CONTEXTS[-1] + L + 127) // 64 * 64
        stride = (max(CONTEXTS) + L + 64 + 255) // 256 * 256
        self.req_to_token = torch.zeros(n_req, stride, dtype=torch.int32, device=DEV)
        self.req_index_to_mamba_index_mapping = torch.zeros(n_req, dtype=torch.int32, device=DEV)  # fmt: skip
        for r, c in enumerate(CONTEXTS):
            self.req_to_token[self.reqs[r], : c + L] = BASES[r] + torch.arange(c + L, device=DEV)  # fmt: skip
            self.req_index_to_mamba_index_mapping[self.reqs[r]] = 5 * R - 2 * r

        def rnd(*shape, scale=1.0):
            return (torch.randn(*shape, generator=g, device=DEV) * scale).bfloat16()

        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(*shape, dtype=dtype, device=DEV)

        self.mamba, self.kv, self.iconv_phys = {}, {}, {}
        for i, kind in enumerate(kinds):
            if kind == "gdn":
                # draft token j's conv window: the physical row's columns j .. j + 2
                phys = self.iconv_phys[i] = zeros(n_scr, GDN_CONV_DIM, L + 2)
                s = phys.stride()
                icw = phys.as_strided((n_scr, L, GDN_CONV_DIM, 3), (s[0], s[2], s[1], s[2]))  # fmt: skip
                self.mamba[i] = SimpleNamespace(
                    conv=[rnd(n_slot, GDN_CONV_DIM, 3, scale=0.5)],
                    temporal=rnd(n_slot, 48, 128, 128, scale=0.02),
                    intermediate_ssm=zeros(n_scr, L, 48, 128, 128),
                    intermediate_conv_window=[icw],
                )
                continue
            k, v, comp = zeros(off + 64, 2, 256), zeros(off + 64, 2, 256), zeros(off // 4 + 32, 1, 128)  # fmt: skip
            for b, c in zip(BASES, CONTEXTS):
                k[b : b + c], v[b : b + c] = rnd(c, 2, 256, scale=0.5), rnd(c, 2, 256, scale=0.5)  # fmt: skip
                comp[b // 4 : (b + c) // 4] = rnd((b + c) // 4 - b // 4, 1, 128, scale=0.5)  # fmt: skip
            self.kv[i] = SimpleNamespace(k=k, v=v, comp=comp, ring=rnd(n_req * 4, 1, 128, scale=0.5))  # fmt: skip
        # the RoPE positions of each request's pending keys c - 4 .. c - 1
        self.qsa_rope_position_buffer = zeros(n_req * 4, 3, dtype=torch.int64)
        for r, c in enumerate(CONTEXTS):
            for q in range(4):
                self.qsa_rope_position_buffer[self.reqs[r] * 4 + q] = c - 1 - (c - 1 - q) % 4  # fmt: skip
        self.short_conv = rnd(n_slot, HC_DIM, LY.PLE_STATE_LEN, scale=0.5)
        self.short_conv_ist = zeros(n_scr, L, HC_DIM, LY.PLE_STATE_LEN)
        ctx = torch.randint(0, 248000, (n_slot, 2), generator=g, device=DEV)
        ictx = torch.full((n_scr, L, 2), eos_token_id(), dtype=torch.int64, device=DEV)
        self.ngram_pool = SimpleNamespace(context=ctx, intermediate_context=ictx)
        # the accessors the runners call
        self.mamba2_layer_cache = self.mamba.__getitem__
        self.short_conv_layer_cache = lambda i: self.short_conv
        self.short_conv_layer_intermediate_cache = lambda i: self.short_conv_ist
        self.get_key_buffer = lambda i: self.kv[i].k
        self.get_value_buffer = lambda i: self.kv[i].v
        self.get_qsa_key_state_buffer = lambda i: self.kv[i].ring
        self.get_qsa_compressed_k_buffer = lambda i: self.kv[i].comp

    def install(self) -> None:
        """Make these the pools SGLang's forward context hands the runners."""
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            set_forward_context,
        )

        backend = SimpleNamespace(req_to_token_pool=self, token_to_kv_pool=self)
        set_forward_context(ForwardContext(attn_backend=backend))

    def tensors(self) -> dict:
        """Every pool tensor by name (the conv windows' physical rows)."""
        ngram = self.ngram_pool
        out = {"rope_pos": self.qsa_rope_position_buffer, "short_conv": self.short_conv,
               "short_conv_ist": self.short_conv_ist, "ngram_ctx": ngram.context,
               "ngram_ictx": ngram.intermediate_context}  # fmt: skip
        for i, m in self.mamba.items():
            out |= {f"L{i}.conv": m.conv[0], f"L{i}.ssm": m.temporal,
                    f"L{i}.issm": m.intermediate_ssm, f"L{i}.iconv": self.iconv_phys[i]}  # fmt: skip
        for i, kv in self.kv.items():
            out |= {f"L{i}.{n}": getattr(kv, n) for n in ("k", "v", "ring", "comp")}
        return out

    def snapshot(self) -> dict:
        return {k: t.clone() for k, t in self.tensors().items()}

    def restore(self, snap: dict) -> None:
        for k, t in self.tensors().items():
            t.copy_(snap[k])

    def allowed_rows(self, b) -> dict:
        """Pool tensor -> the rows step ``b`` may write (none if absent): its KV slots
        and compressed groups, its requests' ring and RoPE rows, its scratch rows."""
        locs = b.out_cache_loc.unique()
        ring = (b.req_pool_indices[:, None] * 4 + torch.arange(4, device=DEV)).flatten()
        scratch = torch.arange(b.batch_size, device=DEV)
        out = {"rope_pos": ring, "short_conv_ist": scratch, "ngram_ictx": scratch}
        for i in self.mamba:
            out |= {f"L{i}.issm": scratch, f"L{i}.iconv": scratch}
        for i in self.kv:
            out |= {f"L{i}.k": locs, f"L{i}.v": locs, f"L{i}.ring": ring, f"L{i}.comp": (locs // 4).unique()}  # fmt: skip
        return out


def step_batch(R: int, rows: int = L, seed: int = 1, pads: int = 0) -> SimpleNamespace:
    """SGLang's forward batch fields of a step of the first R requests, ``rows`` rows
    each at their next positions, and ``pads`` pad requests as SGLang pads a graph's
    batch (request slot 0, KV slot 0, position 0, a sequence of 1)."""
    ctx, pad = CONTEXTS[:R], [0] * pads * rows
    ids = torch.randint(
        0, 248000, ((R + pads) * rows,), generator=gen(seed), device=DEV
    )
    ids[1:2] = eos_token_id()  # an n-gram across an EOS
    pos = [c + j for c in ctx for j in range(rows)]
    locs = [b + c + j for b, c in zip(BASES, ctx) for j in range(rows)]
    reqs = [2 * r + 3 for r in range(R)] + [0] * pads
    pos = torch.tensor(pos + pad, device=DEV)
    # the MRoPE positions [3, S]: a view of SGLang's [3, max tokens] graph buffer
    mrope = torch.zeros(3, MAX_R * L, dtype=torch.int64, device=DEV)[:, : pos.numel()]
    return SimpleNamespace(
        batch_size=R + pads,
        input_ids=ids,
        positions=pos,
        mrope_positions=mrope.copy_(pos.expand(3, -1)),
        out_cache_loc=torch.tensor(locs + pad, device=DEV),
        req_pool_indices=torch.tensor(reqs, device=DEV),
        seq_lens=torch.tensor(list(ctx) + [1] * pads, device=DEV),
    )
