exec(open("smoke_runner.py").read().split("r = RN.VerifyRunner")[0])
views = {"layer": view("qsa", 3), "pe": t(1), "ph": t(1), "fce": t(1), "fch": t(1), "final": {"norm": t(1), "down": t(1), "up": t(1)}}
d = RN.DraftRunner(views=views, embed_w=t(1), cos_sin=lambda: cos, layer_id=0, ntok_extend=4, max_reqs=8, eos_token_id=248044, device="cpu")
hs = t(R * L, 10240)
with mock.patch("sglang.srt.model_executor.forward_context.get_req_to_token_pool", return_value=rtp), \
     mock.patch("sglang.srt.model_executor.forward_context.get_token_to_kv_pool", return_value=kvp):
    table, build = d.prepare_step(fb, hs)
print(table.shape, build)
fb.forward_mode = SimpleNamespace(is_draft_extend_v2=lambda: True, is_decode=lambda: False)
print("eligible", RN.draft_eligible(fb, hs, 4, 8))
