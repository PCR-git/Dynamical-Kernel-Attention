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
        if layer.config.prior in ("tokens", "window"):
            layer.log_c.copy_(rnd(-3.0, 1.0))
        elif layer.config.prior == "direct":
            layer.gamma_raw.copy_(rnd(-0.5, 1.5))
        if layer.theta is not None:
            layer.theta.copy_(rnd(-2.0, 2.0))
        if layer.log_n_sat is not None:
            layer.log_n_sat.copy_(rnd(-1.5, 1.0))
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
               dict(process_noise="steady_state", inflation=0.6), dict(saturation=1.0), dict(saturation=1.0, inflation=0.3)):
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
                s_max = None if p["log_saturation"] is None else math.exp(p["log_saturation"][h].item())
                ref = dka_update(Xq[b, :, h], Xk[b, :, h], Xv[b, :, h], t[b], A, s2 * I, p["eta_sq"][h].item() * I,
                                 math.exp(-p["log_prior_precision"][h].item()) * I, p["nu"][h].item(), inflation=m,
                                 saturation=s_max)
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


def test_evidence_saturation():
    """
    R12: with saturation n_sat the information is S_eff = S / (1 + S eta^2 / n_sat), the estimate is unchanged and
    the step S_eff / (gamma^-2 + S_eff) tends to n_sat / (c_h + n_sat) as agreeing tokens pile up (without saturation
    it tends to 1); a huge n_sat recovers the layer without saturation.
    """
    x, t = inputs(N=8)
    plain = make_layer(inflation=0.3)
    huge = make_layer(inflation=0.3, saturation=1.0)
    with torch.no_grad():
        huge.log_n_sat.fill_(60.0)
    close(huge(x, t_measure=t)[0], plain(x, t_measure=t)[0], 1e-12)

    # N identical tokens in slow heads with no rotation and tiny process noise: every token agrees
    N = 400
    z = torch.randn(1, 1, D_MODEL, dtype=F64).expand(1, N, D_MODEL)
    for sat in (None, 2.0):
        layer = make_layer(mu=(1e-7, 1e-6, 1e-5), saturation=sat)
        with torch.no_grad():
            layer.W_k.load_state_dict(layer.W_q.state_dict())
            layer.omega.zero_()
            layer.sigma_raw.fill_(-30.0)
            if sat is not None:
                layer.log_n_sat.fill_(math.log(sat))
        _, meta = layer(z, return_metadata=True)
        p = layer.sde_params()
        c_h = torch.exp(layer.log_c)
        x_tok = torch.exp(meta["log_evidence"][0]) * p["eta_sq"]              # evidence in agreeing lag-0 tokens
        close(x_tok[-1], torch.full((H,), float(N), dtype=F64), 0.01 * N)
        s_eff = x_tok if sat is None else x_tok / (1 + x_tok / sat)
        close(meta["step"][0], s_eff / (c_h + s_eff), 1e-10)
        limit = 1.0 if sat is None else sat / (c_h + sat)
        assert (meta["step"][0] <= limit + 1e-12).all()
        close(meta["step"][0, -1], limit, 0.01)
        # the step moves toward the same estimate: dV / step does not depend on the saturation
        if sat is None:
            direction = meta["dV"] / meta["step"][..., None, None]
        else:
            close(meta["dV"] / meta["step"][..., None, None], direction, 1e-8)


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


def test_no_null_slot():
    """prior="none" removes the null slot: every step is 1, the update is a full step to the context's estimate, and
    with full inflation (zeta -> 1) as well the layer is exactly RFA with the innovation output (tau = 1, no floor)."""
    layer = make_layer(prior="none", inflation=0.3)
    x, t = inputs(N=8)
    out, meta = layer(x, t_measure=t, return_metadata=True)
    assert torch.isfinite(out).all()
    close(meta["step"], 1.0, 1e-12)
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())

    for robust in (True, False):
        dka = make_layer(prior="none", process_noise="steady_state", inflation=0.5, robust_weight=robust,
                         mu=(0.05, 0.2, 0.4), epsilon=1e-12)
        with torch.no_grad():
            dka.theta.fill_(80.0)
        out_dka, _ = dka(x, t_measure=t)
        rfa = MultiheadIsotropicRFA(RFAConfig(d_model=D_MODEL, n_heads=H, head_dim=D, process_noise="steady_state",
                                              robust_weight=robust, epsilon=1e-12, cov_epsilon=0.0,
                                              output="innovation")).double()
        with torch.no_grad():
            for name in ("W_q", "W_k", "W_v", "W_o"):
                getattr(rfa, name).load_state_dict(getattr(dka, name).state_dict())
            for name in ("omega", "mu", "sigma_raw", "eta_raw", "nu_raw"):
                getattr(rfa, name).copy_(getattr(dka, name))
            rfa.gamma_raw.fill_(-80.0)
            rfa.tau_raw.fill_(math.log(math.expm1(1.0)))
        out_rfa, _ = rfa(x, t_measure=t)
        close(out_dka, out_rfa, 1e-9)


def test_estimate_output():
    """output="estimate" returns W_o (V + dV): it differs from the innovation output by exactly W_o V (no bias)."""
    x, t = inputs(N=8)
    inn = make_layer()
    est = make_layer(output="estimate")
    out_inn, _ = inn(x, t_measure=t)
    out_est, _ = est(x, t_measure=t)
    V = inn._project_in(inn.W_v, x)
    B, N, _ = x.shape
    close(out_est - out_inn, F.linear(V.permute(0, 1, 4, 2, 3).reshape(B, N, -1), inn.W_o.weight), 1e-12)


def test_rotation_ablations():
    """zero_rotations zeroes omega and keeps mu; rotate_values=False changes the output only when omega != 0."""
    layer = MultiheadIsotropicDKA(DKAConfig(d_model=D_MODEL, n_heads=H, head_dim=D, zero_rotations=True))
    ref = MultiheadIsotropicDKA(DKAConfig(d_model=D_MODEL, n_heads=H, head_dim=D))
    assert (layer.omega == 0).all() and torch.equal(layer.mu, ref.mu)
    x, t = inputs(N=8)
    rot, norot = make_layer(), make_layer(rotate_values=False)
    assert (rot(x, t_measure=t)[0] - norot(x, t_measure=t)[0]).abs().max() > 1e-6
    with torch.no_grad():
        rot.omega.zero_()
        norot.omega.zero_()
    close(rot(x, t_measure=t)[0], norot(x, t_measure=t)[0], 1e-12)


def test_qk_norm():
    """qk_norm: the output is invariant to the scale of W_q and W_k, up to the epsilon in the norm (not without it)."""
    x, t = inputs(N=8)
    for qk, invariant in ((True, True), (False, False)):
        a, b = make_layer(qk_norm=qk), make_layer(qk_norm=qk)
        with torch.no_grad():
            b.W_q.weight.mul_(3.0), b.W_q.bias.mul_(3.0), b.W_k.weight.mul_(0.5), b.W_k.bias.mul_(0.5)
        err = (a(x, t_measure=t)[0] - b(x, t_measure=t)[0]).abs().max().item()
        assert (err < 1e-3) == invariant, (qk, err)                     # 1e-3: the epsilon in the norm


def test_mixture_weight():
    """weight="mixture": a_ij Sigma_n = sigmoid(lambda + beta) with lambda the tempered log-likelihood ratio of the
    two Gaussian components (computed here from the densities), and w -> sigmoid(beta) for every key as E -> 0."""
    from dka.kernels import mixture_logits
    g = torch.Generator().manual_seed(5)
    Q, K = torch.randn(1, 4, H, D, 2, generator=g, dtype=F64), torch.randn(1, 4, H, D, 2, generator=g, dtype=F64)
    lags = torch.arange(4, dtype=F64)[None, :, None, None] - torch.arange(4, dtype=F64)[None, None, :, None]
    mu, v_inf, eta_sq = (torch.tensor(v, dtype=F64) for v in ((0.1, 0.4, 0.9), (2.0, 0.5, 1.5), (0.7, 1.2, 0.3)))
    beta, tau = torch.tensor((-1.0, 0.3, 2.0), dtype=F64), torch.tensor((0.5, 1.0, 3.0), dtype=F64)
    E = torch.exp(-mu * lags.abs())
    Sigma, Sigma_inf = v_inf * (1 - E**2) + eta_sq, v_inf + eta_sq
    R_sq = (K[:, None] - E[..., None, None] * Q[:, :, None]).pow(2).sum((-1, -2))
    K_sq = K.pow(2).sum((-1, -2))[:, None]
    # log N(x; m, S I) up to the shared constant, over the d (complex) coordinates as the -d/2 log S convention
    log_p1 = -D / 2 * torch.log(Sigma) - R_sq / (2 * Sigma)
    log_p0 = -D / 2 * torch.log(Sigma_inf) - K_sq / (2 * Sigma_inf)
    content = (log_p1 - log_p0) - D / 2 * torch.log(Sigma_inf / Sigma)          # the quadratic part of the ratio
    lam = torch.log(Sigma_inf / Sigma) + tau * 2 * content / D
    L = mixture_logits(R_sq, K_sq, Sigma, Sigma_inf, D, beta, tau)
    close(torch.exp(L) * Sigma, torch.sigmoid(lam + beta), 1e-12)
    R_far = K_sq.expand_as(R_sq)                                               # E = 0: R = K, Sigma_n = Sigma_inf
    far = mixture_logits(R_far, K_sq, Sigma_inf.expand_as(Sigma), Sigma_inf, D, beta, tau)
    close(torch.exp(far) * Sigma_inf, torch.sigmoid(beta).expand_as(far), 1e-12)
    layer = make_layer(weight="mixture", temperature=True)
    x, t = inputs(N=8)
    out, info = layer(x, t_measure=t)
    out.square().mean().backward()
    assert torch.isfinite(out).all() and layer.beta.grad is not None and "relevance_prior" in info


def test_query_chunk():
    """query_chunk computes the queries in blocks: the same output (irregular times, saturation, mixture, causal or not)."""
    x, t = inputs(N=11)
    for kw in (dict(), dict(saturation=1.0), dict(weight="mixture", temperature=True), dict(inflation=0.3)):
        layer = make_layer(**kw)
        for causal in (True, False):
            ref, info = layer(x, t_measure=t, causal=causal)
            layer.query_chunk = 4
            out, info_c = layer(x, t_measure=t, causal=causal)
            layer.query_chunk = None
            close(out, ref, 1e-12)
            close(info_c["step_mean"], info["step_mean"], 1e-12)


def test_prope():
    """prope_length: heads with horizon above it lose the frequencies with period above it; other heads keep all."""
    L = 512.0
    layer = MultiheadIsotropicDKA(DKAConfig(d_model=D_MODEL, n_heads=8, head_dim=32, damping=1.0, prope_length=L))
    ref = MultiheadIsotropicDKA(DKAConfig(d_model=D_MODEL, n_heads=8, head_dim=32, damping=1.0))
    slow = (1 / ref.mu > L)[:, None]
    low = 2 * math.pi / ref.omega > L
    assert slow.any() and not slow.all()
    assert torch.equal(layer.omega, torch.where(slow & low, torch.zeros_like(ref.omega), ref.omega))


def test_decay_cap():
    """decay_cap: mu_h = min(mu_h, cap); frequencies and the heads below the cap are unchanged."""
    ref = MultiheadIsotropicDKA(DKAConfig(d_model=D_MODEL, n_heads=8, head_dim=32, damping=1.0))
    cap = MultiheadIsotropicDKA(DKAConfig(d_model=D_MODEL, n_heads=8, head_dim=32, damping=1.0, decay_cap=3 / 512))
    assert torch.equal(cap.mu, ref.mu.clamp(max=3 / 512)) and torch.equal(cap.omega, ref.omega)
    assert (ref.mu > 3 / 512).any() and (ref.mu < 3 / 512).any()


def test_polar_residual():
    """residual='polar': E^2 |K|^2 2 (1 - cos); with E = 1 and |Q| = |K| it is ||K - Q||^2, an aligned key matches
    exactly at every lag (the Euclidean residual does not), and the layer runs with finite gradients, with and
    without qk_norm, aggregating each value undivided by E (weight A E^2 on U v_j), so the update is convex."""
    from dka.kernels import polar_residual_norm_sq, dka_residual_norm_sq
    g = torch.Generator().manual_seed(0)
    Q = torch.randn(1, 3, 2, D, 2, generator=g, dtype=F64)
    K = torch.randn(1, 5, 2, D, 2, generator=g, dtype=F64)
    K = K / K.pow(2).sum((-1, -2), keepdim=True).sqrt() * Q[:, :1].pow(2).sum((-1, -2), keepdim=True).sqrt()
    Q = Q / Q.pow(2).sum((-1, -2), keepdim=True).sqrt() * K[:, :1].pow(2).sum((-1, -2), keepdim=True).sqrt()
    ones = torch.ones(1, 3, 5, 2, dtype=F64)
    close(polar_residual_norm_sq(Q, K, ones, eps=0.0), dka_residual_norm_sq(Q, K, ones, ones), 1e-10)
    K_al = 3.0 * Q[:, :1].expand(1, 5, 2, D, 2)                                     # aligned, 3x longer
    for e in (1.0, 0.6, 0.1):
        E = torch.full((1, 1, 5, 2), e, dtype=F64)
        close(polar_residual_norm_sq(Q[:, :1], K_al, E, eps=0.0), 0.0, 1e-9)
    assert dka_residual_norm_sq(Q[:, :1], K_al, E, E * E).min() > 0.1             # Euclidean: best match K = E Q
    x, t = inputs(N=8)
    for qkn in (False, True):
        layer = make_layer(residual="polar", qk_norm=qkn)
        out, _ = layer(x, t_measure=t)
        out.square().mean().backward()
        assert torch.isfinite(out).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())
        # each key's estimate is its value, undivided by E: without the null slot the weights sum to one at every lag
        # (the Euclidean estimator's sum A E / sum A E^2 exceeds one here)
        with torch.no_grad():
            layer.log_c.fill_(-40.0)
        _, meta = layer(x, t_measure=t * 50, return_metadata=True)
        close(meta["attn_mat"].sum(2), 1.0, 1e-10)


def test_flat_precision_and_zero_decay():
    """flat_precision: Sigma_n = eta^2 at every lag, so the forcing sigma^2 has no effect on the output; zero_decay:
    mu = 0 in every head with each head's forcing initialized from its coupled decay (sigma^2 = 2 mu_h v_inf)."""
    x, t = inputs(N=8)
    flat = make_layer(flat_precision=True)
    out, _ = flat(x, t_measure=t * 5)
    with torch.no_grad():
        flat.sigma_raw.add_(3.0)
    close(flat(x, t_measure=t * 5)[0], out, 1e-12)
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in flat.parameters())
    zd = MultiheadIsotropicDKA(DKAConfig(d_model=D_MODEL, n_heads=H, head_dim=D, damping=1.0, zero_decay=True))
    ref = MultiheadIsotropicDKA(DKAConfig(d_model=D_MODEL, n_heads=H, head_dim=D, damping=1.0))
    assert (zd.mu == 0).all() and (ref.mu > 0).all()
    close(zd.sde_params()["sigma_sq"], ref.sde_params()["sigma_sq"], 1e-6)


def test_polar_obs():
    """residual='polar_obs': each key's weight is E^2 kappa^obs w(2 kappa^obs (1 - cos)) with kappa^obs = m_j^2 /
    Sigma_{n,t}, Sigma_{n,t} = eta^2 + rho^2 m_j^2 + forcing (v_inf (1 - E^2) or m_j^2 (1 - E^2) / 2d), the null
    slot c_h m_i^2 / (eta^2 + rho^2 m_i^2); checked against the formula from the layer's own Q and K, for both
    forcings, with finite gradients; without the null slot the weights sum to one (each value undivided by E)."""
    x, t = inputs(N=6)
    for forcing in ("constant", "magnitude"):
        layer = make_layer(residual="polar_obs", polar_forcing=forcing, epsilon=1e-14)   # eps would blur the check
        with torch.no_grad():
            layer.rho_raw.copy_(torch.tensor([-3.0, -1.0, 0.5], dtype=F64))
        out, meta = layer(x, t_measure=t, return_metadata=True)
        out.square().mean().backward()
        assert torch.isfinite(out).all() and all(p.grad is None or torch.isfinite(p.grad).all()
                                                 for p in layer.parameters())
        with torch.no_grad():                                       # the formula, from the layer's Q and K
            p = layer.sde_params()
            cos, sin = rotation_angles(p["omega"], t)
            Q = rotate(layer._project_in(layer.W_q, x), cos, -sin)
            K = rotate(layer._project_in(layer.W_k, x), cos, -sin)
            q_sq, k_sq = Q.pow(2).sum((-1, -2)), K.pow(2).sum((-1, -2))           # [B, N, H]
            cosine = torch.einsum("bihdc,bjhdc->bijh", Q, K) / (q_sq[:, :, None] * k_sq[:, None]).sqrt()
            lags = (t[:, :, None] - t[:, None, :]).abs()[..., None]
            E2 = torch.exp(-2 * p["mu"] * lags)
            m2 = k_sq[:, None]
            force = (m2 * (1 - E2) / (2 * D) if forcing == "magnitude"
                     else p["sigma_sq"] / (2 * p["mu"]) * (1 - E2))
            kobs = m2 / (p["eta_sq"] + p["rho_sq"] * m2 + force)
            tau = torch.exp(layer.log_tau) if layer.log_tau is not None else 1.0
            w = (1 + 2 * kobs * (1 - cosine) / p["nu"]).pow(-tau * (p["nu"] + D) / D)
            weight = (E2 * kobs * w).masked_fill(~torch.ones(6, 6, dtype=torch.bool).tril()[None, :, :, None], 0)
            null = torch.exp(layer.log_c) * k_sq / (p["eta_sq"] + p["rho_sq"] * k_sq)   # kappa_ii uses m_i = |K_i|
            close(meta["attn_mat"], weight / (null[:, :, None] + weight.sum(2, keepdim=True)), 1e-10)
            layer.log_c.fill_(-40.0)
        _, meta = layer(x, t_measure=t * 50, return_metadata=True)
        close(meta["attn_mat"].sum(2), 1.0, 1e-10)


def test_dot_residual():
    """residual='dot': the exponential-weight logits differ from the Euclidean ones by the key-only |K|^2 and
    query-only E^2 |Q|^2 terms; the layer needs the exponential weight."""
    from dka.kernels import dot_residual, dka_residual_norm_sq
    torch.manual_seed(0)
    Q, K = torch.randn(1, 4, H, D // 2, 2, dtype=F64), torch.randn(1, 4, H, D // 2, 2, dtype=F64)
    E = torch.rand(1, 4, 4, H, dtype=F64)
    q_sq = Q.pow(2).sum(dim=(-1, -2))[:, :, None, :]
    k_sq = K.pow(2).sum(dim=(-1, -2))[:, None, :, :]
    close(dot_residual(Q, K, E), dka_residual_norm_sq(Q, K, E, E * E) - k_sq - E * E * q_sq, 1e-10)
    try:
        make_layer(residual="dot")                                  # needs the exponential weight
        raise AssertionError("residual='dot' with the Student-t weight was accepted")
    except ValueError:
        pass
    x, t = inputs(N=8)
    layer = make_layer(residual="dot", robust_weight=False)
    out, info = layer(x, t_measure=t, return_metadata=True)
    out.square().mean().backward()
    assert torch.isfinite(out).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())


def test_rfa_null_slot():
    """RFA with the proximal prior (prior="tokens"): a vanishing c_h recovers RFA's innovation output, and with full
    inflation (zeta -> 1, h = 1) DKA with the same null slot equals it exactly (tau = 1, no floor)."""
    x, t = inputs(N=8)
    cfg = dict(d_model=D_MODEL, n_heads=H, head_dim=D, output="innovation")
    flat = MultiheadIsotropicRFA(RFAConfig(**cfg)).double()
    null = MultiheadIsotropicRFA(RFAConfig(**cfg, prior="tokens")).double()
    null.load_state_dict(flat.state_dict(), strict=False)
    with torch.no_grad():
        null.log_c.fill_(-60.0)
    close(null(x, t_measure=t)[0], flat(x, t_measure=t)[0], 1e-10)
    with torch.no_grad():
        null.log_c.fill_(0.5)
    out, meta = null(x, t_measure=t, return_metadata=True)
    assert torch.isfinite(out).all() and (meta["attn_mat"].sum(2) < 1).all()     # the null slot takes some mass
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in null.parameters())

    for robust in (True, False):
        dka = make_layer(prior="tokens", process_noise="steady_state", inflation=0.5, robust_weight=robust,
                         mu=(0.05, 0.2, 0.4), epsilon=1e-12)
        with torch.no_grad():
            dka.theta.fill_(80.0)
        out_dka, _ = dka(x, t_measure=t)
        rfa = MultiheadIsotropicRFA(RFAConfig(d_model=D_MODEL, n_heads=H, head_dim=D, process_noise="steady_state",
                                              robust_weight=robust, epsilon=1e-12, cov_epsilon=0.0,
                                              output="innovation", prior="tokens")).double()
        with torch.no_grad():
            for name in ("W_q", "W_k", "W_v", "W_o"):
                getattr(rfa, name).load_state_dict(getattr(dka, name).state_dict())
            for name in ("omega", "mu", "sigma_raw", "eta_raw", "nu_raw", "log_c"):
                getattr(rfa, name).copy_(getattr(dka, name))
            rfa.gamma_raw.fill_(-80.0)
            rfa.tau_raw.fill_(math.log(math.expm1(1.0)))
        out_rfa, _ = rfa(x, t_measure=t)
        close(out_dka, out_rfa, 1e-9)


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
    for kw in (dict(), dict(inflation=0.3, prior="direct"), dict(saturation=1.0)):
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



def test_baseline_innovation_and_null():
    """Baselines: output='innovation' returns W_o (sum_j w_j V_j - V_i); learned_null adds a null slot holding V_i,
    so the output is W_o sum_j w_j (V_j - V_i) with the weights a softmax over the keys and the logit b_h."""
    from dka.model.baselines.attention import MultiHeadAttentionLayer
    from dka.model.config import AttentionConfig
    torch.manual_seed(0)
    x = torch.randn(2, 9, 16, dtype=F64)
    est = MultiHeadAttentionLayer(AttentionConfig(d_model=16, n_heads=2, width=16)).double()
    inn = MultiHeadAttentionLayer(AttentionConfig(d_model=16, n_heads=2, width=16, output="innovation")).double()
    nul = MultiHeadAttentionLayer(AttentionConfig(d_model=16, n_heads=2, width=16, output="innovation",
                                                  learned_null=True)).double()
    with torch.no_grad():
        for m in (inn, nul):
            m.load_state_dict(est.state_dict(), strict=False)
        nul.null_logit.copy_(torch.tensor([0.7, -1.3]).view(1, 2, 1, 1))
    V = est.value_projection(x)
    W = est.output_projection
    close(inn(x)[0], est(x)[0] - (V @ W.weight.T), 1e-12)        # W_o is affine: W_o(c - V_i) = W_o c - W V_i

    out, meta = nul(x)
    Q, K = (nul.rope(P(x).view(2, 9, 2, 8)).transpose(1, 2) for P in (nul.query_projection, nul.key_projection))
    Vh = V.view(2, 9, 2, 8).transpose(1, 2)
    scores = (Q @ K.transpose(-2, -1) / math.sqrt(8)).masked_fill(torch.ones(9, 9).triu(1).bool(), float("-inf"))
    b = nul.null_logit.expand(2, 2, 9, 1)
    w = torch.softmax(torch.cat([scores, b], -1), -1)[..., :-1]
    ref = torch.einsum("bhij,bhijd->bhid", w, Vh[:, :, None] - Vh[:, :, :, None])
    close(out, W(ref.transpose(1, 2).reshape(2, 9, 16)), 1e-12)
    try:
        AttentionConfig(d_model=16, n_heads=2, width=16, learned_null=True)
        raise AssertionError("learned_null without the innovation output was accepted")
    except ValueError:
        pass

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
