"""
Checks of the DKA attention layer (dka/dka_attention.py): against the literal reference (dka/reference.py),
against RFA in the flat-prior limit, the limits of Section 3.4 and Appendix E, Proposition 3, and numerics.

    python tests/test_dka_layer.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dka import DKAConfig, MultiheadIsotropicDKA, RFAConfig, MultiheadIsotropicRFA
from dka.kernels import rotate, rotation_angles
from dka.reference import dka_update

C128, F64 = torch.complex128, torch.float64
D_MODEL, H, D = 12, 3, 4


def close(a, b, tol):
    a, b = (x if torch.is_tensor(x) else torch.tensor(x, dtype=F64) for x in (a, b))
    err = (a - b).abs().max().item()
    assert err <= tol, f"max abs error {err:.3e} > {tol:.1e}"


def cplx(x):
    """[..., 2] real/imaginary pairs -> complex."""
    return torch.complex(x[..., 0], x[..., 1])


def make_layer(seed=0, mu=(0.08, 0.3, 0.7), **kw):
    """A float64 DKA layer with randomized per-head parameters (so that no default value hides a mistake)."""
    torch.manual_seed(seed)
    layer = MultiheadIsotropicDKA(DKAConfig(d_model=D_MODEL, n_heads=H, head_dim=D, **kw)).double()
    g = torch.Generator().manual_seed(seed + 100)
    rnd = lambda lo, hi: lo + (hi - lo) * torch.rand(H, generator=g, dtype=F64)
    with torch.no_grad():
        layer.mu.copy_(torch.tensor(mu, dtype=F64))
        layer.omega.copy_(torch.rand(H, D // 2, generator=g, dtype=F64) * 1.5)
        layer.sigma_raw.copy_(rnd(-1.5, 1.0))
        layer.eta_raw.copy_(rnd(-1.0, 1.0))
        layer.nu_raw.copy_(rnd(-1.0, 1.5))
        if layer.config.prior != "direct":
            layer.log_c.copy_(rnd(-3.0, 1.0))
        else:
            layer.gamma_raw.copy_(rnd(-0.5, 1.5))
        if layer.theta is not None:
            layer.theta.copy_(rnd(-2.0, 2.0))
    return layer


def inputs(B=2, N=6, seed=1, irregular=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, D_MODEL, generator=g, dtype=F64)
    t = torch.cumsum(0.4 + torch.rand(B, N, generator=g, dtype=F64), 1) if irregular else None
    return x, t


# ---------------------------------------------------------------------------------------------------
# The layer computes the paper's update
# ---------------------------------------------------------------------------------------------------

def test_matches_reference():
    """dV of the layer equals the literal per-pair update of Eqs. 9-10 (dka.reference.dka_update), per head."""
    for kw in (dict(), dict(inflation=0.3), dict(prior="direct"), dict(prior="window"),
               dict(process_noise="steady_state", inflation=0.6)):
        layer = make_layer(**kw)
        x, t = inputs()
        _, meta = layer(x, t_measure=t, return_metadata=True)
        p = layer.sde_params()
        Xq, Xk, Xv = (cplx(layer._project_in(W, x)) for W in (layer.W_q, layer.W_k, layer.W_v))
        for b in range(x.shape[0]):
            for h in range(H):
                mu, s2 = p["mu"][h].item(), p["sigma_sq"][h].item()
                A = torch.diag(torch.complex(torch.full((D,), -mu, dtype=F64), p["omega"][h]))
                I = torch.eye(D, dtype=C128)
                m = 0.0 if p["theta"] is None else s2 / (2 * mu) * math.exp(p["theta"][h].item())
                ref = dka_update(Xq[b, :, h], Xk[b, :, h], Xv[b, :, h], t[b], A, s2 * I, p["eta_sq"][h].item() * I,
                                 math.exp(-p["log_prior_precision"][h].item()) * I, p["nu"][h].item(), inflation=m)
                close(cplx(meta["dV"][b, :, h]), ref["dV"], 1e-10)


def test_output_projection_and_metadata():
    """out = W_o dV; dV = sum_j attn Phi+ V~_j - step V; step + null share = 1; posterior variance = 1 / (gamma^-2 + S)."""
    layer = make_layer(inflation=0.2)
    x, t = inputs()
    out, meta = layer(x, t_measure=t, return_metadata=True)
    B, N = x.shape[:2]
    close(out, layer.W_o(meta["dV"].permute(0, 1, 4, 2, 3).reshape(B, N, -1)), 1e-12)
    cos, sin = rotation_angles(layer.sde_params()["omega"], t)
    V = layer._project_in(layer.W_v, x)
    agg = rotate(torch.einsum("bijh,bjhdc->bihdc", meta["attn_mat"], rotate(V, cos, -sin)), cos, sin)
    close(meta["dV"], agg - meta["step"][..., None, None] * V, 1e-12)
    close(meta["step"] + meta["null_share"], 1.0, 1e-12)
    prior_precision = torch.exp(layer.sde_params()["log_prior_precision"])
    close(meta["posterior_var"], 1 / (prior_precision + torch.exp(meta["log_evidence"])), 1e-10)


def test_prior_counts_tokens():
    """With prior="tokens", one fully agreeing token (the self-pair with W_k = W_q) takes the step 1 / (1 + c_h) in
    every head, whatever its decay, noise, inflation or degrees of freedom; "window" makes slow heads abstain."""
    for kw in (dict(), dict(inflation=0.4), dict(process_noise="steady_state")):
        layer = make_layer(mu=(0.0, 0.003, 0.8), **kw)
        with torch.no_grad():
            layer.W_k.load_state_dict(layer.W_q.state_dict())
        x, _ = inputs(N=1)
        _, meta = layer(x, return_metadata=True)
        c_h = torch.exp(layer.log_c)
        close(meta["step"], (1 / (1 + c_h)).expand_as(meta["step"]), 1e-12)
    # In "window" mode with the same c_h, a single agreeing token moves a slow head far less than a fast one
    layer = make_layer(mu=(0.001, 0.05, 0.8), prior="window")
    with torch.no_grad():
        layer.W_k.load_state_dict(layer.W_q.state_dict())
        layer.log_c.zero_()
    _, meta = layer(inputs(N=1)[0], return_metadata=True)
    step = meta["step"][0, 0]
    assert step[0] < 0.5 * step[2] and step[0] < step[1] < step[2], step


def test_flat_prior_limit_is_rfa():
    """
    Full inflation (zeta -> 1) and a full step (gamma^2 -> inf) give RFA (tau = 1, no floor gamma^2, no cov_eps)
    with the innovation output V_bar - V; with RFA's estimate output the difference is exactly the term -W_o V.
    """
    for robust in (True, False):
        dka = make_layer(prior="direct", process_noise="steady_state", inflation=0.5, robust_weight=robust,
                         mu=(0.05, 0.2, 0.4), epsilon=1e-12)
        with torch.no_grad():
            dka.theta.fill_(80.0)            # zeta = 1 - e^-80; theta / (2 mu) is far beyond every lag
            dka.gamma_raw.fill_(1e12)        # gamma^2 = 1e12
        x, t = inputs(N=10)
        out_dka, _ = dka(x, t_measure=t)
        V = dka._project_in(dka.W_v, x).permute(0, 1, 4, 2, 3).reshape(x.shape[0], x.shape[1], -1)
        for output in ("innovation", "estimate"):
            rfa = MultiheadIsotropicRFA(RFAConfig(d_model=D_MODEL, n_heads=H, head_dim=D, process_noise="steady_state",
                                                  robust_weight=robust, epsilon=1e-12, cov_epsilon=0.0,
                                                  output=output)).double()
            with torch.no_grad():
                for name in ("W_q", "W_k", "W_v", "W_o"):
                    getattr(rfa, name).load_state_dict(getattr(dka, name).state_dict())
                for name in ("omega", "mu", "sigma_raw", "eta_raw", "nu_raw"):
                    getattr(rfa, name).copy_(getattr(dka, name))
                rfa.gamma_raw.fill_(-80.0)       # floor gamma^2 = softplus(-80) ~ 1e-35
                rfa.tau_raw.fill_(math.log(math.expm1(1.0)))   # tau = 1 exactly (inv_softplus adds 1e-6)
            out_rfa, _ = rfa(x, t_measure=t)
            shift = F.linear(V, dka.W_o.weight) if output == "estimate" else 0.0
            close(out_dka + shift, out_rfa, 1e-9)


def test_zero_decay_is_softmax_plus_one():
    """Section 3.4: for zero-decay heads (h = 1) and gamma^2 = 1, the update is a softmax over the innovations
    Phi+_i V~_j - V_i with one extra logit 0 for the null slot, whose innovation is zero."""
    layer = make_layer(prior="direct", mu=(0.0, 0.0, 0.0))
    with torch.no_grad():
        layer.gamma_raw.fill_(math.log(math.expm1(1.0 - layer.config.epsilon)))   # gamma^2 = 1 exactly
    x, t = inputs(N=7)
    _, meta = layer(x, t_measure=t, return_metadata=True)
    p = layer.sde_params()
    cos, sin = rotation_angles(p["omega"], t)
    Q, K, V = (layer._project_in(W, x) for W in (layer.W_q, layer.W_k, layer.W_v))
    Qr, Kr, Vr = (cplx(rotate(X, cos, -sin)) for X in (Q, K, V))
    lag = (t[:, :, None] - t[:, None, :])[..., None]                                   # [B, N, N, 1]
    Sigma_n = p["sigma_sq"] * lag.abs() + p["eta_sq"]
    R_sq = (Kr.abs()**2).sum(-1)[:, None] + (Qr.abs()**2).sum(-1)[:, :, None] - 2 * torch.einsum("bihd,bjhd->bijh", Qr.conj(), Kr).real
    kappa = (p["nu"] + D) / D
    L = -torch.log(Sigma_n) - kappa * torch.log1p(R_sq / (p["nu"] * Sigma_n))
    L = L.masked_fill(lag < 0, float("-inf"))
    weights = torch.softmax(torch.cat([L, torch.zeros_like(L[:, :, :1])], dim=2), dim=2)[:, :, :-1]
    innovations = cplx(rotate(torch.stack([Vr.real, Vr.imag], -1)[:, None].expand(-1, x.shape[1], -1, -1, -1, -1),
                              cos[:, :, None], sin[:, :, None])) - cplx(V)[:, :, None]   # [B, i, j, H, D]
    close(cplx(meta["dV"]), torch.einsum("bijh,bijhd->bihd", weights.to(C128), innovations), 1e-12)


def test_rope_limit():
    """Appendix E: mu = 0, sigma^2 = 0, exponential weight and a full step give rotary softmax attention with
    inverse temperature 2 / (eta^2 nu), a key-norm bias, and rotated values (RoPER)."""
    layer = make_layer(prior="direct", mu=(0.0, 0.0, 0.0), robust_weight=False)
    with torch.no_grad():
        layer.sigma_raw.fill_(-80.0)
        layer.gamma_raw.fill_(1e12)
    x, _ = inputs(N=8)
    _, meta = layer(x, return_metadata=True)
    p = layer.sde_params()
    t = torch.arange(8, dtype=F64)[None]
    cos, sin = rotation_angles(p["omega"], t)
    Q, K, V = (layer._project_in(W, x) for W in (layer.W_q, layer.W_k, layer.W_v))
    Qr, Kr = rotate(Q, cos, -sin), rotate(K, cos, -sin)
    temp = 2 / (p["eta_sq"] * p["nu"])
    logits = temp * torch.einsum("bihdc,bjhdc->bijh", Qr, Kr) - (Kr.pow(2).sum((-1, -2)) / (p["eta_sq"] * p["nu"]))[:, None]
    logits = logits.masked_fill(~torch.ones(8, 8, dtype=torch.bool).tril()[None, :, :, None], float("-inf"))
    est = rotate(torch.einsum("bijh,bjhdc->bihdc", torch.softmax(logits, 2), rotate(V, cos, -sin)), cos, sin)
    close(meta["dV"] + V, est, 1e-9)


# ---------------------------------------------------------------------------------------------------
# Structure: causality, memory window (Proposition 3), zero-decay limit
# ---------------------------------------------------------------------------------------------------

def test_causal():
    layer = make_layer(inflation=0.4)
    x, t = inputs(N=9)
    out, _ = layer(x, t_measure=t)
    x2 = x.clone()
    x2[:, 5:] += torch.randn_like(x2[:, 5:])
    out2, _ = layer(x2, t_measure=t)
    close(out[:, :5], out2[:, :5], 0)
    assert (out[:, 5:] - out2[:, 5:]).abs().max() > 1e-3


def test_memory_window():
    """Proposition 3: dropping tokens older than T changes dV_i by at most C e^{-mu T}, C = B e^-mu theta (3 + theta),
    theta = gamma^2 / (eta^2 (1 - e^-mu)), B the largest value norm; and the change does shrink with T."""
    N = 120
    layer = make_layer(prior="direct", mu=(0.1, 0.25, 0.6))
    x, _ = inputs(B=3, N=N, irregular=False)
    _, meta = layer(x, return_metadata=True)
    p = layer.sde_params()
    V = layer._project_in(layer.W_v, x)
    B_norm = V.pow(2).sum((-1, -2)).sqrt().amax(dim=1)                                # [B, H]
    gamma_sq = torch.exp(-p["log_prior_precision"])
    theta = gamma_sq / (p["eta_sq"] * (1 - torch.exp(-p["mu"])))
    C = B_norm * torch.exp(-p["mu"]) * theta * (3 + theta)
    t = torch.arange(N, dtype=F64)[None].expand(3, -1)
    changes = []
    for T in (2, 5, 10, 20, 40):
        _, meta_T = layer(x[:, N - 1 - T:], t_measure=t[:, N - 1 - T:], return_metadata=True)
        change = (meta["dV"][:, -1] - meta_T["dV"][:, -1]).pow(2).sum((-1, -2)).sqrt()  # [B, H]
        bound = C * torch.exp(-p["mu"] * T)
        assert (change <= bound + 1e-12).all(), (T, change, bound)
        changes.append(change)
    assert (changes[-1] < 0.1 * changes[0]).all()


def test_zero_decay_continuity():
    """mu = 0 heads (sigma^2 learned directly) are the limit of slow heads."""
    x, t = inputs(N=8)
    outs = []
    for mu in (0.0, 1e-9):
        layer = make_layer(mu=(mu, 0.3, mu), inflation=0.3)
        outs.append(layer(x, t_measure=t)[0])
    close(outs[0], outs[1], 1e-6)


# ---------------------------------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------------------------------

def test_gradcheck():
    for kw in (dict(), dict(inflation=0.3, prior="direct")):
        layer = make_layer(**kw)
        x, t = inputs(B=1, N=5)
        x.requires_grad_(True)
        assert torch.autograd.gradcheck(lambda z: layer(z, t_measure=t)[0], (x,), eps=1e-6, atol=1e-6)


def test_float32():
    """float32 matches float64 at ordinary inputs, and stays finite (values and gradients) at extreme ones."""
    layer64 = make_layer(inflation=0.3)
    layer32 = make_layer(inflation=0.3).float()
    x, t = inputs(N=64)
    out64, _ = layer64(x, t_measure=t)
    out32, _ = layer32(x.float(), t_measure=t.float())
    assert ((out32.double() - out64).abs().max() / out64.abs().max()) < 1e-4
    for scale, eta_raw, theta in ((1e3, -12.0, 15.0), (1e-3, 8.0, -15.0), (30.0, 0.0, 0.0)):
        layer = make_layer(inflation=0.3).float()
        with torch.no_grad():
            layer.eta_raw.fill_(eta_raw)
            layer.theta.fill_(theta)
        z = (torch.randn(2, 512, D_MODEL) * scale).requires_grad_(True)
        out, meta = layer(z, return_metadata=True)
        out.square().mean().backward()
        assert torch.isfinite(out).all() and torch.isfinite(z.grad).all(), scale
        for prm in layer.parameters():
            assert prm.grad is None or torch.isfinite(prm.grad).all(), scale


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
