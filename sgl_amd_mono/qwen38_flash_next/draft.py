# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang#45353 at 6ce98069c3 (Apache-2.0), by salykova:
# python/sglang/srt/models/qwen4_exp_monokernel.py (Qwen4ExpMTPMonoRunner) and the
# hook of python/sglang/srt/models/qwen4_exp_mtp.py
"""Qwen3.8-Flash-Next's MTP draft on gfx950: its spec and forward (RFC #43949).

NOT REGISTERED (``sgl_amd_mono.SPECS``): monokernel API version 1 routes a draft
model's steps by the target's key, ("target_verify", R, draft tokens), while the
NEXTN draft runs its steps as DRAFT_EXTEND_V2 (which ``shape_key`` does not
route) and as DECODE steps of one token a request. Registered now, the spec would
build launches that never run. It waits for SGLang to route a draft's own step
modes; until then the draft runs SGLang's forward.

A draft step is one launch (``gfx950/runner.DraftRunner``): the MTP input fusion
(the token embedding and the target's hyper-connection hidden states, each
normalized and projected), the QSA decoder layer and the draft's mixer. A draft
extend runs R requests x 4 rows, a draft decode step R x 1, and ``_Draft.forward``
returns what ``Qwen4ExpForCausalLMMTP.forward`` does. The weights are SGLang's,
read in place, except the copies logged at startup: the hyper-connection mixers,
router and input-fusion projections packed for the MFMA tiles, and the FP8
projections SGLang keeps row-major, zero-padded in AITER's shuffle.
"""

from __future__ import annotations

from sglang.srt.monokernels.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Requirements,
)

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.qwen38_flash_next import spec as target

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/monokernels/spec.py": ("DeviceLimits", "Feature", "MonoSpec", "Requirements"),
    "srt/models/qwen4_exp_mtp.py": (
        "Qwen4ExpForCausalLMMTP",
        "model",
        "lm_head",
        "logits_processor",
        "_set_hc_logits_hidden_states",
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
    ),
    "srt/models/qwen3_5.py": ("layers", "embed_tokens", "rotary_emb"),
    "srt/layers/radix_attention.py": ("layer_id",),
    "srt/model_executor/forward_batch_info.py": ("spec_info",),
    "srt/speculative/*.py": ("hidden_states",),
    "srt/runtime_context.py": ("get_parallel", '"tp_size"'),
}


def _tokens_per_request(model) -> int:
    """A draft extend's rows a request: the target's verify rows."""
    return target.TOKENS


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.qwen38_flash_next.gfx950 import weights

    reasons = []
    if get_parallel().tp_size != 1:
        reasons.append("the draft launch runs the MTP layer on one GPU")
    return reasons + weights.mtp_reasons(model)


class _Draft:
    """The draft's launch and its forward."""

    def __init__(self, lm, runner):
        self.lm, self.runner = lm, runner

    def forward(
        self, plan, input_ids, positions, forward_batch, input_embeds=None, *_, **kw
    ):
        """``Qwen4ExpForCausalLMMTP.forward`` for a routed draft step."""
        from sgl_amd_mono.qwen38_flash_next.gfx950.runner import draft_eligible

        lm, r = self.lm, self.runner
        spec = forward_batch.spec_info
        hidden = None if spec is None else spec.hidden_states
        if input_embeds is not None or not draft_eligible(
            forward_batch, hidden, r.L_ext, r.max_reqs
        ):
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, input_embeds, **kw
            )
        x, hc = r.launch(forward_batch, hidden)
        out = lm.logits_processor(input_ids, x, lm.lm_head, forward_batch)
        lm._set_hc_logits_hidden_states(out, hc, forward_batch)
        return out


def _build(model, hf_config, ctx) -> _Draft:
    import torch

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.qwen38_flash_next.gfx950 import weights
    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import DraftRunner

    params = weights.param_ptrs(model)
    views = weights.mtp_views(model, ctx.copies, params)
    layer = model.model.layers[0]
    rope = layer.rotary_emb
    runner = DraftRunner(
        views=views,
        embed_w=model.model.embed_tokens.weight,
        cos_sin=lambda: rope.cos_sin_cache_as(torch.bfloat16),
        layer_id=layer.attn.layer_id,
        ntok_extend=target.TOKENS,
        max_reqs=target.MAX_R,
        eos_token_id=int(target._text(hf_config).eos_token_id),
        device=model.model.embed_tokens.weight.device,
        fault=guard.fault_injection(0),
    )
    runner.set_deadline(ctx.deadline_s)
    return _Draft(model, runner)


SPEC = MonoSpec(
    name="qwen38_flash_next_mtp_draft_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, target.MAX_R * target.TOKENS + 1)),
            tp_sizes=(target.TP,),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    # the launcher checks the target's requirements; the capture check, this
    requires=Requirements(attention_backends=target.SPEC.requires.attention_backends),
    numerics=target.NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    tokens_per_request=_tokens_per_request,
)
