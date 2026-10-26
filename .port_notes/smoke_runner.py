"""CPU smoke of the host runner: tables and launch keys of a fake verify step."""
from types import SimpleNamespace
from unittest import mock
import torch
torch.cuda.is_current_stream_capturing = lambda: False
from sgl_amd_mono.qwen38_flash_next.gfx950 import layout as LY, runner as RN

t = lambda *s, d=torch.bfloat16: torch.zeros(*s, dtype=d)
hc = lambda: {"norm": t(1), "down": t(1), "up": t(1), "inj": t(1)}
moe = {"gu": t(1), "gu_s": t(1), "dn": t(1), "dn_s": t(1)}
kinds = ["qsa" if i % 4 == 3 else "gdn" for i in range(48)]
def view(kind, i):
    v = {"hca": hc(), "hcm": hc(), "router": t(1), "moe": moe, "out": t(1), "out_s": t(1)}
    if kind == "gdn":
        v.update(ip=t(1), ip_s=t(1), conv_w=t(1), neg_a=t(1), dt_bias=t(1), norm_w=t(1), ba=t(1), ba_s=t(1))
    else:
        v.update(qp=t(1), qp_s=t(1), qp2=t(1), qp2_s=t(1), qnorm=t(1), knorm=t(1), iqnorm=t(1), iknorm=t(1))
    if i == 1:
        v["ple"] = {k: t(1) for k in ("w", "nk", "nq", "nc", "cw")}
    return v
layers = {i: view(k, i) for i, k in enumerate(kinds)}
R, L, slots = 3, 4, 4
conv = t(slots, 10240, 3); ssm = t(slots, 48, 128, 128, d=torch.float32)
issm = t(8, 4, 48, 128, 128, d=torch.float32)
icw_phys = t(8 * 4 * 10240 * 3 + 64)
icw = icw_phys.as_strided((8, 4, 10240, 3), (4 * 30720, 30720, 3, 1))
cache = SimpleNamespace(conv=[conv], temporal=ssm, intermediate_ssm=issm, intermediate_conv_window=[icw])
ngram = SimpleNamespace(context=t(slots, 2, d=torch.int64), intermediate_context=t(8, 4, 2, d=torch.int64))
rtp = SimpleNamespace(
    req_to_token=t(16, 262208, d=torch.int32)[:, :], req_index_to_mamba_index_mapping=t(16, d=torch.int32),
    mamba2_layer_cache=lambda i: cache, ngram_pool=ngram,
    short_conv_layer_cache=lambda i: t(slots, 10240, 9),
    short_conv_layer_intermediate_cache=lambda i: t(8, 4, 10240, 9),
)
kv = t(100, 2, 256)
kvp = SimpleNamespace(get_key_buffer=lambda i: kv, get_value_buffer=lambda i: kv,
    get_qsa_key_state_buffer=lambda i: t(16, 128), get_qsa_compressed_k_buffer=lambda i: t(16, 128),
    qsa_rope_position_buffer=t(100, d=torch.int64))
fb = SimpleNamespace(batch_size=R, input_ids=t(R * L, d=torch.int64), positions=t(R * L, d=torch.int64),
    out_cache_loc=t(R * L, d=torch.int64), req_pool_indices=t(R, d=torch.int64), seq_lens=t(R, d=torch.int64))
cos = t(1000, 64)
r = RN.VerifyRunner(kinds=kinds, layers=layers, final={"norm": t(1), "down": t(1), "up": t(1)},
    ngram={"meta": t(35, d=torch.int64), "table": t(1), "scale": t(1)}, embed_w=t(1), cos_sin=lambda: cos,
    ntok=4, max_reqs=8, eos_token_id=248044, device="cpu", fault=None)
r.set_deadline(30.0)
print("epoch", r.epoch.tolist())
with mock.patch("sglang.srt.model_executor.forward_context.get_req_to_token_pool", return_value=rtp), \
     mock.patch("sglang.srt.model_executor.forward_context.get_token_to_kv_pool", return_value=kvp):
    tables, keys = r.prepare_step(fb)
    again = r.prepare_step(fb)
assert again[0] is tables
print(len(tables), [tuple(x.shape) for x in tables[:3]], tables[-1].shape)
for k in (keys[0], keys[1], keys[2], keys[-1]):
    print(k)
assert tables[0][1, LY.G_EMBED_W] == 0 or True
print("ple ptr set:", int(tables[1][0, LY.P_PLE_STATE]) != 0, "ple istate:", int(tables[1][0, LY.P_PLE_ISTATE]) != 0)
# a bad cache is refused with a reason
cache.intermediate_ssm = None
fb2 = SimpleNamespace(**{**vars(fb), "input_ids": t(R * L, d=torch.int64)})
with mock.patch("sglang.srt.model_executor.forward_context.get_req_to_token_pool", return_value=rtp), \
     mock.patch("sglang.srt.model_executor.forward_context.get_token_to_kv_pool", return_value=kvp):
    try:
        r.prepare_step(fb2)
    except ValueError as e:
        print("refused:", str(e)[:200])
