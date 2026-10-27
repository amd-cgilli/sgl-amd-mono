# SPDX-License-Identifier: Apache-2.0
"""Each Qwen3.8-Flash-Next launch kind on one gfx950 GPU, on the checkpoint's first 12
layers and MTP head (``qwen38_flash_next/prefix.py``): verify steps (prelude, PLE,
middle, final-mixer launches, MRoPE positions) of 1, 8 and 7 + 1 pad requests, MTP
draft extend and decode steps of 1 and 8. Every step leaves the error words zero,
writes only its own pool rows (a pad request: KV slot 0, request slot 0's ring, its
scratch rows), and a CUDA graph replays it bitwise (but the rows its pads race on).
The prelude's n-gram embedding and context are bitwise a torch reference's. Every MoE
block's router logits match an f32 router, its output an a16w4 torch MoE on the
launch's input and routes. The GDN and QSA layers are checked against SGLang's
forward by check mode (``test/e2e``)."""

import contextlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from aiter.utility.fp4_utils import mxfp4_to_f32

pytest.importorskip("flydsl")
if (
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
):
    pytest.skip("gfx950 only", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).with_name("qwen38_flash_next")))
import prefix as P  # noqa: E402

from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.qwen38_flash_next.gfx950 import layout as LY  # noqa: E402
from sgl_amd_mono.qwen38_flash_next.gfx950 import packing  # noqa: E402
from sgl_amd_mono.qwen38_flash_next.gfx950.config import (  # noqa: E402
    HC_DIM,
    HIDDEN,
    N_EXPERTS,
    TOPK,
)

if P.checkpoint_dir() is None:
    pytest.skip(f"no local checkpoint {P.MODEL}", allow_module_level=True)


REQUESTS = (1, 8)  # 7 + 1 pad builds the launches of 8


def _router(small: dict, p: str = "") -> torch.Tensor:
    """The 512 routed experts' and the shared expert's gate rows, f32."""
    gate = [small[p + "mlp.gate.weight"], small[p + "mlp.shared_expert_gate.weight"]]
    return torch.cat(gate).float()


@pytest.fixture(scope="module")
def target():
    t = P.target()
    t.pools = P.Pools(t.kinds)
    t.pools.install()
    steps = [P.step_batch(R) for R in REQUESTS]
    P.precompile(k for b in steps for k in t.runner.prepare_step(b, b.mrope_positions)[1])  # fmt: skip
    t.snap = t.pools.snapshot()
    t.blocks = {i: (_router(t.small[i]), t.layers[i]["moe"]) for i in t.layers}
    return t


@pytest.fixture(scope="module")
def draft():
    d = P.draft()
    d.pools = P.Pools(["qsa"])
    d.pools.install()
    steps = [P.step_batch(R, rows) for R in REQUESTS for rows in (P.L, 1)]
    hs = torch.zeros(P.MAX_R * P.L, HC_DIM, dtype=torch.bfloat16, device="cuda")
    P.precompile(d.runner.prepare_step(b, hs[: b.input_ids.numel()])[1] for b in steps)
    d.snap = d.pools.snapshot()
    d.blocks = [(_router(d.small, "layers.0."), d.views["layer"]["moe"])]
    return d


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _mailbox(runner, key, layer: int, name: str, n: int, tag: int, whole=True):
    """The first n (value, tag) pairs of mailbox ``name`` of the launch's layer:
    (values int32 [n], tagged by this launch bool [n])."""
    S = key.nreq * key.ntok
    w0 = LY.scratch_layout(layer + 1, S, key.nreq)[0][f"L{layer}.{name}"] // 4
    pairs = runner.scratch[w0 : w0 + 2 * n].view(n, 2)
    ok = pairs[:, 1] == tag
    assert not whole or bool(ok.all()), f"layer {layer} {name}: stale pairs"
    return pairs[:, 0].contiguous(), ok


def _bf16(words: torch.Tensor, S: int) -> torch.Tensor:
    return words.view(torch.bfloat16).view(S, -1)


@contextlib.contextmanager
def _after_each_launch(runner, check):
    """``check(key, tag)`` after each launch of the runner's eager steps; its first
    failure is raised once the step is done (an unfinished step leaves its tags)."""
    build, failed = runner.kernel, []

    def kernel(key):
        launch = build(key)

        def run(*args, stream):
            launch(*args, stream=stream)
            torch.cuda.synchronize()
            try:
                check(key, int(runner.epoch[0]) * LY.LAUNCH_SLOTS + args[5] + 1)
            except AssertionError as e:
                failed.append(e)

        return run

    runner.kernel = kernel
    try:
        yield
    finally:
        del runner.kernel
    if failed:
        raise failed[0]


def _error_words(runner) -> list:
    return runner.epoch[list(guard.ERROR_WORDS)].tolist()


def _stray_rows(pools, snap: dict, b) -> list:
    """The pool rows the step wrote outside its own, and the states it owns but
    left unwritten."""
    allowed, bad = pools.allowed_rows(b), []
    for name, t in pools.tensors().items():
        ne = (_bits(t) != _bits(snap[name])).view(t.shape[0], -1).any(1)
        rows, ok = ne.nonzero().flatten(), allowed.get(name)
        stray = rows if ok is None else rows[~torch.isin(rows, ok)]
        if stray.numel():
            bad.append(f"{name} rows {stray[:4].tolist()}")
        if name.endswith((".k", ".v", ".issm", ".iconv")) and not rows.numel():
            bad.append(f"{name} unwritten")
    return bad


def _pad_rows(name: str, R: int) -> slice:
    """The rows of pool ``name`` the pads of a step of R requests write."""
    if name.endswith(("ring", "rope_pos")):
        return slice(0, 4)
    if name.endswith((".k", ".v", ".comp")):
        return slice(0, 1)
    return slice(R, None) if name.endswith(("conv_ist", "ictx", "issm", "iconv")) else slice(0, 0)  # fmt: skip


def _run_step(m, b, run, check, pads: int = 0) -> None:
    """Step ``b`` (``run()`` -> its outputs) eagerly with ``check`` after each launch,
    then from a CUDA graph, each from the same pools."""
    pools, runner = m.pools, m.runner
    pools.install()
    pools.restore(m.snap)
    with _after_each_launch(runner, check):
        eager = [t.clone() for t in run()]
    assert not any(err := _error_words(runner)), err
    assert not (bad := _stray_rows(pools, m.snap, b)), bad
    after = pools.snapshot()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run()
    pools.restore(m.snap)
    graph.replay()
    torch.cuda.synchronize()
    assert not any(err := _error_words(runner)), err
    assert not (bad := _stray_rows(pools, m.snap, b)), bad
    R = b.batch_size - pads
    S = R * b.input_ids.numel() // b.batch_size
    assert all(_same(o[:S], e[:S]) for o, e in zip(outs, eager)), "graph outputs"
    diff = []
    for n, t in pools.tensors().items():
        ne = (_bits(t) != _bits(after[n])).view(t.shape[0], -1).any(1)
        ne[_pad_rows(n, R) if pads else slice(0, 0)] = False
        diff += [n] if ne.any() else []
    assert not diff, f"graph pools differ: {diff}"
    pools.restore(m.snap)


def _check_prelude(t, b, key, tag: int) -> None:
    """The n-gram embedding and context as ``Qwen4ExpNGramEmbedding`` computes them,
    bitwise."""
    R, S = b.batch_size, b.batch_size * P.L
    pools, ngram, eos = t.pools, t.ngram, P.eos_token_id()
    slots = pools.req_index_to_mamba_index_mapping[b.req_pool_indices].long()
    seq = torch.cat([pools.ngram_pool.context[slots], b.input_ids.view(R, -1)], 1)
    t0, t1, t2 = seq.unfold(1, 3, 1).reshape(S, 3).unbind(1)
    mult, vocab = ngram["meta"][:3], ngram["meta"][3:19]
    p2 = torch.where((t0 == eos) | (t1 == eos), eos, t0)
    bigram = (t2 * mult[0]) ^ (t1 * mult[1])
    mix = torch.stack([bigram, bigram ^ (p2 * mult[2])], 1).repeat_interleave(8, 1)
    rows = ngram["table"][torch.remainder(mix, vocab).flatten()]
    want = (rows.bfloat16() * ngram["scale"]).view(S, -1)
    assert _same(t.runner.emb_buf[:S], want), "PLE n-gram embedding"
    ictx = pools.ngram_pool.intermediate_context[:R]
    assert torch.equal(ictx, seq.unfold(1, 2, 1)[:, 1:]), "n-gram context"


def _deq(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    v = mxfp4_to_f32(w.contiguous()).unflatten(-1, (-1, 32))
    return (v * torch.exp2(s.float() - 127)[..., None]).flatten(-2)


def _torch_moe(x: torch.Tensor, moe: dict, logits: torch.Tensor) -> torch.Tensor:
    """The MXFP4-weight, bf16-activation MoE on the top-10 of softmax(logits)
    renormalized plus the shared expert (sigmoid gate), f32 [S, HIDDEN]."""
    p = logits[:, :N_EXPERTS].softmax(-1)
    wts, ids = p.topk(TOPK, -1)
    wts = torch.cat([wts / wts.sum(-1, keepdim=True), logits[:, N_EXPERTS:].sigmoid()], 1)  # fmt: skip
    ids = torch.cat([ids, torch.full_like(ids[:, :1], N_EXPERTS)], 1)
    y = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for e in ids.unique().tolist():
        g, u, d, gs, us, ds = packing.unshuffle_moe_expert(moe, e)
        rows, k = (ids == e).nonzero(as_tuple=True)
        xe = x[rows].float()
        mid = (F.silu(xe @ _deq(g, gs).T) * (xe @ _deq(u, us).T)).bfloat16().float()
        y.index_add_(0, rows, wts[rows, k, None] * (mid @ _deq(d, ds).T))
    return y


def _check_moe(runner, key, tag: int, blocks: list) -> None:
    """The launch's MoE blocks (``blocks``: (router, experts) a layer) from its
    mailboxes: the input x, the router logits, the down partials of every group."""
    S = key.nreq * key.ntok
    for li, (router, moe) in enumerate(blocks):
        x = _bf16(_mailbox(runner, key, li, "m.mixx", S * HIDDEN // 2, tag)[0], S)
        logits = _mailbox(runner, key, li, "logit", S * LY.ROUTER_ROWS, tag)[0]
        logits = logits.view(torch.float32).view(S, -1)[:, : N_EXPERTS + 1]
        want = x.float() @ router.T
        rel = (logits - want).norm() / want.norm()
        assert rel < 1e-5, f"launch layer {li}: router logits rel {rel}"
        groups = LY.max_groups(S, LY.dn_sg(S))
        words, ok = _mailbox(runner, key, li, "dpart", groups * S * HIDDEN, tag, False)
        live = ok.view(groups, -1).all(1)
        y = words.view(torch.float32).view(groups, S, HIDDEN)[live].sum(0)
        cos = F.cosine_similarity(y, _torch_moe(x, moe, logits), dim=1)
        assert cos.min() > 0.9999, f"launch layer {li}: MoE cosine {cos.min()}"


@pytest.mark.parametrize(
    "requests,pads", [(1, 0), (8, 0), (7, 1)], ids=["R=1", "R=8", "R=7+1 pad"]
)
def test_verify_step(target, requests, pads):
    b = P.step_batch(requests, seed=requests, pads=pads)
    plans = target.runner.plans

    def check(key, tag):
        i = (tag - 1) % LY.LAUNCH_SLOTS
        _check_moe(target.runner, key, tag, [target.blocks[li] for li in plans[i][1]])
        if i == 0 and not pads:
            _check_prelude(target, b, key, tag)

    run = lambda: target.runner.launch(b, b.mrope_positions)  # noqa: E731
    _run_step(target, b, run, check, pads)


@pytest.mark.parametrize("rows", [P.L, 1], ids=["extend", "decode"])
@pytest.mark.parametrize("requests", REQUESTS, ids=["R=1", "R=8"])
def test_draft_step(draft, requests, rows):
    b = P.step_batch(requests, rows, seed=10 + requests)
    hs = torch.randn(requests * rows, HC_DIM, generator=P.gen(11), device="cuda")
    hs = hs.bfloat16()

    def check(key, tag):
        _check_moe(draft.runner, key, tag, draft.blocks)

    _run_step(draft, b, lambda: draft.runner.launch(b, hs), check)
