"""
Stage 4 checks: DKA inside the model code (Block / Backbone / LanguageModel, the notebook constructors, the
presets, training logs and diagnostics), torch.compile, gradient checkpointing and bf16, and the RFA
innovation presets.

    python tests/test_dka_integration.py
"""

import copy
import math
import dataclasses
import os
import sys
import time

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dka import MultiheadIsotropicDKA, MultiheadIsotropicRFA, DKAConfig
from dka.init import init_spectrally_coupled_rope
from dka.model.baselines import MultiHeadAttentionLayer, RoPE, XPos
from dka.model import (Backbone, BackboneConfig, LanguageModel, RFATransformerNetwork, print_layer_diagnostics,
                       get_model_sde_snapshots)
from dka.training import single_epoch_lm
from dka.training.presets import make_args, build_backbone, build_language_model

DKA_PRESETS = ("D1", "D2", "D3", "D2.Z", "D3.Z", "D2.S", "D3.S")


def small(preset, **kw):
    return make_args(preset, **dict(dict(d_e=32, d_k_total=32, d_v_total=32, n_heads=4, num_blocks=2, seq_len=64,
                                         device="cpu"), **kw))


def attn_layers(model):
    return [m for m in model.modules() if isinstance(m, (MultiheadIsotropicDKA, MultiheadIsotropicRFA))]


def test_presets_build_dka():
    """The D presets build DKA layers with the intended settings: no zero-decay heads, the per-token prior,
    calibration over the training length, SC decays mu = b omega_max, inflation in the .Z variants and evidence
    saturation in the .S variants."""
    for preset in DKA_PRESETS:
        args = small(preset)
        net = build_backbone(args)
        layers = attn_layers(net)
        assert len(layers) == 2 and all(isinstance(m, MultiheadIsotropicDKA) for m in layers), preset
        c = layers[0].config
        v_inf = layers[0]._param_summary(layers[0].sde_params())["sigma_tilde_sq_v"]
        assert torch.allclose(v_inf, torch.full_like(v_inf, args.target_ss_process_var), rtol=1e-4), (preset, v_inf)
        assert c.zero_decay_frac == 0 and c.prior == "tokens" and c.calibration_length == 64, preset
        assert (c.inflation == 0.5) == preset.endswith(".Z") and (layers[0].theta is not None) == preset.endswith(".Z")
        assert (c.saturation == 1.0) == preset.endswith(".S") and (layers[0].log_n_sat is not None) == preset.endswith(".S")
        if preset != "D1":
            _, mu = init_spectrally_coupled_rope(4, 4, b=args.damping, zero_frac=0.0)
            assert torch.allclose(layers[0].mu, mu) and c.frequency_init == "spectral", preset
        assert (layers[0].mu > 0).all(), preset
        out, infos = net(torch.randn(2, 10, 32))
        assert out.shape == (2, 10, 32) and torch.isfinite(out).all(), preset
        assert infos[0]["prior_weight"].shape == (4,) and infos[0]["step_mean"].shape == (4,), preset


def test_spectrum_presets():
    """D3.b02 / D3.b05 (paper size): mu_h = b omega_h,max on the RoPE spectrum with the preset's base, the slowest
    head's horizon 1/mu is D3's 625 tokens, and the frequency range is 2 or 2.5 decades narrower than RoPE's 4; the
    RFA twins (M3.b02 / M3.b05) have the same omega and mu with no zero-decay heads, and B1.b02 / B1.b05 use the
    same base."""
    for b, (d, m, r) in {0.2: ("D3.b02", "M3.b02", "B1.b02"), 0.5: ("D3.b05", "M3.b05", "B1.b05")}.items():
        args = make_args(d, device="cpu")
        dka = attn_layers(build_backbone(args))[0]
        rfa = attn_layers(build_backbone(make_args(m, device="cpu")))[0]
        assert isinstance(dka, MultiheadIsotropicDKA) and isinstance(rfa, MultiheadIsotropicRFA)
        assert args.damping == b and abs(1 / dka.mu[0].item() - 625) < 1e-2, (d, 1 / dka.mu)
        assert dka.config.temperature and dka.log_tau is not None, d
        assert not attn_layers(build_backbone(make_args(d + ".noT", device="cpu")))[0].config.temperature, d
        omega, mu = init_spectrally_coupled_rope(8, 16, b=b, base=args.rope_base)
        assert torch.allclose(dka.omega, omega) and torch.allclose(dka.mu, mu), d
        assert torch.allclose(rfa.omega, omega) and torch.allclose(rfa.mu, mu) and (rfa.mu > 0).all(), m
        assert abs(math.log10(dka.omega.max() / dka.omega.min()) - math.log10(args.rope_base) * 127 / 128) < 1e-4
        rope = _baseline(r).rope
        D = 2 * rope.theta.numel()
        assert torch.allclose(rope.theta, 1.0 / args.rope_base ** (torch.arange(0, D, 2).float() / D)), r
    for b in (0.2, 0.5):                         # the slowest-horizon sweep, and the fastest head at 1/b
        for h, hv in (("312", 312.5), ("1250", 1250.0)):
            mu = attn_layers(build_backbone(make_args(f"D3.b0{int(b * 10)}.h{h}", device="cpu")))[0].mu
            assert abs(1 / mu[0].item() - hv) < 1e-2 * hv and abs(1 / mu[-1].item() - 1 / b) < 1e-4, (b, h)
    # the default base is unchanged, and SC baselines take the base too
    assert torch.allclose(attn_layers(build_backbone(small("D3")))[0].omega, init_spectrally_coupled_rope(4, 4)[0])
    layer = [x for x in build_backbone(small("B5", rope_base=100.0)).modules() if isinstance(x, MultiHeadAttentionLayer)][0]
    n_freqs = layer.rope.theta.shape[1]
    assert torch.allclose(layer.rope.theta, init_spectrally_coupled_rope(4, n_freqs, base=100.0)[0])
    assert torch.allclose(layer.bias_slopes.flatten(), 2 * init_spectrally_coupled_rope(4, n_freqs, b=5.0, base=100.0)[1])


def test_dka_temperature():
    """D3.T: a learned inverse temperature per head on the evidence logits, initialized at tau = 1 (so D3.T starts
    as D3 with the same weights); the null slot is not tempered, and tau gets gradients."""
    torch.manual_seed(0)
    net_t, net = build_backbone(small("D3.T")), build_backbone(small("D3"))
    layer_t = attn_layers(net_t)[0]
    assert layer_t.config.temperature and torch.equal(layer_t.log_tau.data, torch.zeros(4))
    assert attn_layers(net)[0].log_tau is None
    net.load_state_dict({k: v for k, v in net_t.state_dict().items() if "log_tau" not in k})
    x = torch.randn(2, 10, 32)
    assert torch.allclose(net_t(x)[0], net(x)[0], atol=1e-6)
    net_t(x)[0].square().mean().backward()
    assert all(m.log_tau.grad is not None and m.log_tau.grad.abs().sum() > 0 for m in attn_layers(net_t))
    with torch.no_grad():                        # tau -> 0: every key's weight -> 1, the null slot keeps its weight
        for m in attn_layers(net_t):
            m.log_tau.fill_(-30.0)
    assert torch.isfinite(net_t(x)[0]).all()


def _baseline(preset):
    net = build_backbone(small(preset)).eval()
    return [m for m in net.modules() if isinstance(m, MultiHeadAttentionLayer)][0]


def test_b7_xpos():
    """xPos: each frequency pair's contribution to q_n . k_m is RoPE's times zeta_k^((n - m) / 512), with
    zeta_k = (2k + 0.4 D) / (1.4 D), whatever the sequence length (the centering cancels)."""
    layer = _baseline("B7")
    assert layer.config.rope == "xpos" and isinstance(layer.rope, XPos)
    D = 8
    xpos, rope = XPos(D), RoPE(D)
    zeta = (torch.arange(0, D, 2).double() + 0.4 * D) / (1.4 * D)
    for L in (7, 12):
        q, k = torch.randn(1, L, 2, D), torch.randn(1, L, 2, D)
        xq, xk = xpos(q).view(1, L, 2, D // 2, 2), xpos(k, key=True).view(1, L, 2, D // 2, 2)
        rq, rk = rope(q).view(1, L, 2, D // 2, 2), rope(k).view(1, L, 2, D // 2, 2)
        x_pair = torch.einsum("bnhkc,bmhkc->bnmhk", xq, xk)
        r_pair = torch.einsum("bnhkc,bmhkc->bnmhk", rq, rk)
        lag = (torch.arange(L)[:, None] - torch.arange(L)[None, :]).double()
        decay = zeta[None, None, :] ** (lag[:, :, None] / 512)                        # [L, L, D/2]
        assert torch.allclose(x_pair.double(), r_pair.double() * decay[None, :, :, None, :], atol=1e-5)
    out, info = build_backbone(small("B7"))(torch.randn(2, 10, 32))
    assert torch.isfinite(out).all()


def test_b8_nope():
    """NoPE: no positional encoding and no bias; permuting the earlier tokens leaves the last output unchanged."""
    layer = _baseline("B8")
    c = layer.config
    assert c.rope is None and layer.rope is None and not (c.alibi or c.kerple_log or c.softmax_plus_one)
    assert c.bias_damping is None and c.decay_damping is None
    x = torch.randn(1, 9, 32)
    perm = torch.cat([torch.randperm(8), torch.tensor([8])])
    with torch.no_grad():
        assert torch.allclose(layer(x)[0][0, -1], layer(x[:, perm])[0][0, -1], atol=1e-6)


def test_b10_b5_plus_null():
    """B10 = B5 (SC-RoPE + coupled linear bias, slopes 2 mu_h) with softmax plus one: rows sum to at most one."""
    layer = _baseline("B10")
    c = layer.config
    assert c.rope == "sc_rope" and c.bias_damping == 5.0 and c.softmax_plus_one and c.decay_damping is None
    with torch.no_grad():
        a = layer(torch.randn(2, 10, 32))[1]["attn_mat"]
    assert torch.isfinite(a).all() and (a.sum(-1) <= 1 + 1e-6).all()


def test_b9_softmax_plus_one():
    """RoPE + softmax plus one: the weights are a softmax over the keys and one extra zero logit, so each row sums
    to at most one, and they stay finite for very large or very negative scores."""
    layer = _baseline("B9")
    assert layer.config.softmax_plus_one and layer.config.rope == "rope"
    B, L, H, D = 2, 10, layer.num_heads, layer.head_dim
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    for scale in (1.0, 30.0, -30.0):
        x = torch.randn(B, L, 32)
        with torch.no_grad():
            if scale != 1.0:
                layer.query_projection.weight.mul_(scale)
            a = layer(x)[1]["attn_mat"]
            Q = layer.rope(layer.query_projection(x).view(B, L, H, D)).transpose(1, 2)
            K = layer.rope(layer.key_projection(x).view(B, L, H, D)).transpose(1, 2)
            s = (Q @ K.transpose(-2, -1) / D ** 0.5).masked_fill(~causal, float("-inf"))
            ref = torch.cat([s, torch.zeros(B, H, L, 1)], -1).softmax(-1)[..., :-1]
            if scale != 1.0:
                layer.query_projection.weight.div_(scale)
        assert torch.isfinite(a).all() and (a.sum(-1) <= 1 + 1e-6).all()
        assert torch.allclose(a, ref, atol=1e-6), scale


def test_yarn():
    """YaRN on a RoPE model: unchanged up to the training length; beyond it (s = L / L0) the shortest wavelengths keep
    their frequency, the longest are divided by s, and the logits are scaled by (0.1 ln s + 1)^2."""
    D, L0 = 64, 512
    rope, yarn = RoPE(D), RoPE(D, yarn_original_length=L0)
    assert torch.equal(yarn.frequencies(L0), rope.theta) and yarn.yarn_logit_scale(L0) == 1.0
    f = yarn.frequencies(4 * L0)
    wavelength = 2 * math.pi / rope.theta
    fast, slow = wavelength < L0 / 32, wavelength > L0
    assert torch.allclose(f[fast], rope.theta[fast]) and torch.allclose(f[slow], rope.theta[slow] / 4)
    assert ((f <= rope.theta + 1e-12) & (f >= rope.theta / 4 - 1e-12)).all()
    assert abs(yarn.yarn_logit_scale(4 * L0) - (0.1 * math.log(4) + 1) ** 2) < 1e-12
    # in a model: identical outputs up to L0, different beyond
    torch.manual_seed(0)
    base = build_backbone(small("B1")).eval()
    with_yarn = build_backbone(small("B1", yarn_original_length=16)).eval()
    with_yarn.load_state_dict(base.state_dict())
    with torch.no_grad():
        x = torch.randn(1, 16, 32)
        assert torch.equal(base(x)[0], with_yarn(x)[0])
        x = torch.randn(1, 40, 32)
        assert not torch.allclose(base(x)[0], with_yarn(x)[0])


def test_b6_kerple_log():
    """B6 = KERPLE-log, no rotation: the weights are those of the same layer without the bias, times
    (1 + a_h dt)^-p_h, renormalized; p_h and a_h are learned and clamped at 0.01."""
    torch.manual_seed(0)
    net = build_backbone(small("B6")).eval()
    layer = [m for m in net.modules() if isinstance(m, MultiHeadAttentionLayer)][0]
    c = layer.config
    assert c.kerple_log and c.rope is None and not c.alibi and c.bias_damping is None and c.decay_damping is None
    assert layer.kerple_p.requires_grad and (0 <= layer.kerple_p).all() and (layer.kerple_p <= 2).all()
    assert (0 <= layer.kerple_a).all() and (layer.kerple_a <= 1).all()
    with torch.no_grad():
        layer.kerple_a[0] = -1.0                       # below the clamp: acts as 0.01
    plain = copy.deepcopy(layer)
    plain.config = dataclasses.replace(c, kerple_log=False)
    x = torch.randn(2, 12, 32)
    with torch.no_grad():
        a = layer(x)[1]["attn_mat"]
        a0 = plain(x)[1]["attn_mat"]
    dist = (torch.arange(12)[:, None] - torch.arange(12)[None, :]).abs()
    p = layer.kerple_p.detach().clamp(min=0.01).view(1, -1, 1, 1)
    q = layer.kerple_a.detach().clamp(min=0.01).view(1, -1, 1, 1)
    w = a0 * (1 + q * dist) ** (-p)
    assert torch.allclose(a, w / w.sum(-1, keepdim=True), atol=1e-6)


def test_b5_coupled_linear_bias():
    """B5 = SC-RoPE with the bias -2 mu_h |i - j| before the softmax, mu_h = b omega_h,max (DKA's log h at zeta = 0):
    its weights are those of the same layer without the bias, times exp(-2 mu_h dt), renormalized."""
    args = small("B5")
    net = build_backbone(args).eval()
    layer = [m for m in net.modules() if isinstance(m, MultiHeadAttentionLayer)][0]
    c = layer.config
    assert c.rope == "sc_rope" and c.bias_damping == 5.0 and c.decay_damping is None and not c.alibi
    _, mu = init_spectrally_coupled_rope(4, layer.head_dim // 2, b=5.0)
    assert torch.allclose(layer.bias_slopes.flatten(), 2 * mu)
    plain = copy.deepcopy(layer)
    plain.config = dataclasses.replace(c, bias_damping=None)
    x = torch.randn(2, 12, 32)
    with torch.no_grad():
        a = layer(x)[1]["attn_mat"]
        a0 = plain(x)[1]["attn_mat"]
    pos = torch.arange(12)
    w = a0 * torch.exp(-layer.bias_slopes * (pos[:, None] - pos[None, :]).abs())
    assert torch.allclose(a, w / w.sum(-1, keepdim=True), atol=1e-6)


def test_typed_backbone_matches_notebook_classes():
    """The presets' typed Backbone equals the notebook constructor RFATransformerNetwork for DKA args."""
    for preset in ("D2", "D3.Z"):
        args = small(preset)
        torch.manual_seed(0)
        typed = build_backbone(args)
        torch.manual_seed(0)
        old = RFATransformerNetwork(args, num_blocks=2, n_heads=4, input_dim=32, query_key_dim_total=32,
                                    value_dim_total=32, hidden_dim=128, Norm=nn.LayerNorm)
        sd_typed, sd_old = typed.state_dict(), old.state_dict()
        assert list(sd_typed) == list(sd_old), preset
        assert all(torch.equal(sd_typed[k], sd_old[k]) for k in sd_old), preset
        x = torch.randn(2, 10, 32)
        assert torch.equal(typed(x)[0], old(x)[0]), preset


def test_rfa_only_flags_raise():
    """RFA ablation flags have no DKA counterpart and are refused rather than silently ignored."""
    for flag, value in (("rotate_values", False), ("additive_bias_type", 0), ("zero_rotations", True),
                        ("lambda_real_zero", True), ("use_full_residual_norm", 0)):
        try:
            build_backbone(small("D2", **{flag: value}))
        except ValueError:
            continue
        raise AssertionError(f"{flag}={value} should raise for DKA")
    try:
        build_backbone(small("M2", attention_type="dkaa"))
    except ValueError:
        return
    raise AssertionError("unknown attention_type should raise")


def test_innovation_presets():
    """M*.I are the M* models with the innovation output: same parameters and init, output shifted by -W_o V."""
    for base in ("M1", "M2", "M3"):
        torch.manual_seed(0)
        est = build_backbone(small(base))
        torch.manual_seed(0)
        inn = build_backbone(small(base + ".I"))
        assert attn_layers(est)[0].config.output == "estimate" and attn_layers(inn)[0].config.output == "innovation"
        sd_e, sd_i = est.state_dict(), inn.state_dict()
        assert list(sd_e) == list(sd_i) and all(torch.equal(sd_e[k], sd_i[k]) for k in sd_e), base
        layer_e, layer_i = attn_layers(est)[0].double(), attn_layers(inn)[0].double()
        x = torch.randn(2, 10, 32, dtype=torch.float64)
        V = layer_e._project_in(layer_e.W_v, x).permute(0, 1, 4, 2, 3).reshape(2, 10, -1)
        diff = layer_e(x)[0] - layer_i(x)[0]
        assert torch.allclose(diff, nn.functional.linear(V, layer_e.W_o.weight), atol=1e-12), base


def _lm_grads(lm, ids, use_loss_method):
    lm.zero_grad()
    if use_loss_method:
        loss, _ = lm.loss(ids, ids, causal=True)
    else:
        logits, _ = lm(ids, causal=True)
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in lm.named_parameters() if p.grad is not None}


def test_chunked_loss_and_checkpointing_are_exact():
    """LanguageModel.loss (chunked CE) and block checkpointing give the plain path's loss and gradients."""
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (2, 30))
    for preset in ("D2", "D2.Z"):
        lm = build_language_model(small(preset, vocab_size=100)).double()
        lm.loss_chunk_size = 7
        ref_loss, ref_grads = _lm_grads(lm, ids, use_loss_method=False)
        assert "log_c" in " ".join(ref_grads) and all(torch.isfinite(g).all() for g in ref_grads.values())
        for ckpt in (False, True):
            lm.backbone.grad_checkpointing = ckpt
            loss, grads = _lm_grads(lm, ids, use_loss_method=True)
            assert torch.allclose(loss, ref_loss), (preset, ckpt)
            assert all(torch.allclose(grads[n], ref_grads[n]) for n in ref_grads), (preset, ckpt)


def test_training_logs():
    """single_epoch_lm logs c_h, zeta or n_sat and the mean step of every block for DKA; RFA's history is unchanged."""
    ids = torch.randint(0, 100, (2, 16))
    batches = [{"input_ids": ids, "labels": ids}] * 3
    for preset in ("D2.Z", "D2.S", "M2"):
        args = small(preset, vocab_size=100)
        lm = build_language_model(args)
        opt = torch.optim.Adam(lm.parameters(), lr=1e-3)
        _, hist = single_epoch_lm(lm, batches, {"loss": []}, opt, args, log_every=1)
        assert len(hist["loss"]) == 3 and len(hist["mu"]) == 3 and hist["mu"][0].shape == (4,)
        if preset != "M2":
            assert "tau" not in hist
            extra = "zeta" if preset == "D2.Z" else "saturation"
            assert ("zeta" in hist) == (preset == "D2.Z") and ("saturation" in hist) == (preset == "D2.S")
            for key in ("prior_weight", extra, "step"):
                assert len(hist[key]) == 3 and hist[key][0].shape == (2, 4), key
            assert ((hist["step"][0] > 0) & (hist["step"][0] < 1)).all()
            assert not (hist["prior_weight"][0] == hist["prior_weight"][-1]).all()   # c_h is trained
        else:
            assert hist["tau"][0].shape == (4,) and "prior_weight" not in hist and "step" not in hist


def test_metadata_and_diagnostics():
    net = build_backbone(small("D2.Z")).eval()
    _, infos = net(torch.randn(1, 10, 32), return_metadata=True)
    assert [info["attn_mat"].shape for info in infos] == [(1, 10, 10, 4)] * 2
    assert infos[0]["P_tot"].shape == (1, 10, 4)
    _, infos = net(torch.randn(1, 10, 32))
    assert "attn_mat" not in infos[0]
    print_layer_diagnostics(net)
    snap = get_model_sde_snapshots(net)
    assert len(snap) == 2 and {"prior_weight", "zeta"} <= set(snap["layer_0"]) and "tau" not in snap["layer_0"]


def test_compiled_matches_eager():
    """In-place torch.compile of a DKA backbone: same loss, float32-level gradients, unchanged state_dict keys."""
    if not torch.cuda.is_available():
        print("  (no GPU, skipped)")
        return
    for preset in ("D2", "D2.Z", "D2.S"):
        torch.manual_seed(0)
        eager = build_language_model(small(preset, vocab_size=100)).cuda()
        comp = copy.deepcopy(eager)
        comp.backbone.compile()
        ids = torch.randint(0, 100, (2, 64), device="cuda")
        (le, ge), (lc, gc) = _lm_grads(eager, ids, True), _lm_grads(comp, ids, True)
        assert torch.allclose(le, lc, rtol=1e-4), preset
        assert all((ge[n] - gc[n]).norm() <= 1e-2 * ge[n].norm() + 1e-6 for n in ge), preset
        assert list(comp.state_dict()) == list(eager.state_dict())


def test_bf16_autocast():
    """Under bf16 autocast (not used by train_lm.py, but a common option) values and gradients stay finite
    and close to float32."""
    if not torch.cuda.is_available():
        print("  (no GPU, skipped)")
        return
    torch.manual_seed(0)
    lm = build_language_model(small("D2.Z", vocab_size=100, seq_len=256)).cuda()
    ids = torch.randint(0, 100, (2, 256), device="cuda")
    l32, g32 = _lm_grads(lm, ids, True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l16, g16 = _lm_grads(lm, ids, True)
    assert torch.isfinite(l16) and all(torch.isfinite(g).all() for g in g16.values())
    assert abs(l16 - l32) < 2e-2 * l32, (l16, l32)
    rel = max(((g16[n] - g32[n]).norm() / (g32[n].norm() + 1e-12)).item() for n in g32)
    print(f"  bf16 loss {l16.item():.4f} vs fp32 {l32.item():.4f}; worst relative gradient error {rel:.3f}")
    assert rel < 0.2


def test_gpu_paper_size():
    """Forward/backward at the paper's size (d=256, 8 heads, 6 blocks, L=512, B=16): DKA vs RFA."""
    if not torch.cuda.is_available():
        print("  (no GPU, skipped)")
        return
    x = torch.randn(16, 512, 256, device="cuda")
    for preset in ("M2.I", "D2", "D2.Z"):
        net = build_backbone(make_args(preset)).cuda()
        net(x)[0].square().mean().backward()   # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(3):
            out, _ = net(x)
            out.square().mean().backward()
        torch.cuda.synchronize()
        print(f"  {preset:5s} fwd+bwd: {(time.time() - t0) / 3 * 1000:.0f} ms, "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
        assert torch.isfinite(out).all()
        del net, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[pass]  {name}")
            except Exception as e:
                failed += 1
                print(f"[FAIL]  {name}: {type(e).__name__}: {e}")
    print(f"\n{failed} failing test(s)")
    sys.exit(1 if failed else 0)
