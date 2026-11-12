"""
DKA's update as two attention passes: the form a fused kernel (FlexAttention) computes.

With logits L_ij (the robust logits of the key-frame residual) and E_ij = e^{-mu lag}, the update

    dV_i = (Phi+_i * sum_j a_ij E_ij V~_j - S_i V_i) / (gamma^-2 + S_i),   a_ij = e^{L_ij},  S_i = sum_j a_ij E_ij^2,

needs two sums with different weights on the same keys, which two softmax passes with their log-sum-exps give:

    pass 1, logits L + log E, values V~:  o1 = sum softmax(L + log E) V~,   lse1 = log sum a E
    pass 2, logits L + 2 log E:           lse2 = log sum a E^2 = log S

    dV_i = (w1 Phi+_i * o1 - s V_i) / (n + s),   (w1, s, n) = exp((lse1, lse2, log gamma^-2) - max of the three).

dka_logit is the elementwise logit, written so that a FlexAttention score_mod can call it unchanged; the lag must
be clamped at zero there, since partly masked blocks evaluate future pairs too. two_pass_update materializes the
weights (the reference for the fused kernel, checked against the layer's own computation).
"""
import torch
import torch._dynamo

from .kernels import cross_dot, robust_logits, rotate


def _one_minus_exp_ratio(x):
    """(1 - e^{-x}) / x, with its limit 1 at x = 0 (as kernels._one_minus_exp_ratio)."""
    small = x < 1e-4
    x_safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 - x / 2 + x * x / 6, -torch.expm1(-x_safe) / x_safe)


def dka_logit(cross, lag, q_sq, k_sq, mu, sigma_sq, eta_sq, nu, tau, d, robust_weight):
    """
    Robust logit L of one query-key pair and log E (broadcastable tensors), as the layer computes them:
        R^2 = |K|^2 + E^2 |Q|^2 - 2 E Re<Q~, K~>  (clamped at 0),   Sigma_n = sigma^2 lag (1 - e^{-x}) / x + eta^2,
        L = -log Sigma_n + tau * content,   content = -kappa log(1 + R^2 / (nu Sigma_n)) (Student-t, kappa =
        (nu + d) / d) or -R^2 / (nu Sigma_n) (exponential).  lag >= 0.
    """
    log_E = -mu * lag
    E = torch.exp(log_E)
    R_sq = (k_sq + E * E * q_sq - 2 * E * cross).clamp_min(0)
    Sigma = sigma_sq * lag * _one_minus_exp_ratio(2 * mu * lag) + eta_sq
    if robust_weight:
        content = -(nu + d) / d * torch.log(1 + R_sq / (nu * Sigma))
    else:
        content = -R_sq / (nu * Sigma)
    if tau is not None:
        content = tau * content
    return -torch.log(Sigma) + content, log_E


def combine(o1, lse1, lse2, null_logit, V, cos=None, sin=None):
    """dV and the step S / (gamma^-2 + S) from the two passes; o1, V: [B, N, H, D, 2]; lse*, null: [B, N, H] | [H]."""
    m = torch.maximum(torch.maximum(lse1, lse2), null_logit)
    w1, s, n = torch.exp(lse1 - m), torch.exp(lse2 - m), torch.exp(null_logit - m)
    M = w1[..., None, None] * o1
    if cos is not None:                                           # counter-rotation Phi+
        M = rotate(M, cos, sin)
    return (M - s[..., None, None] * V) / (n + s)[..., None, None], s / (n + s)


_FLEX, _MASKS = {}, {}
KERNEL_OPTIONS = None        # FlexAttention kernel_options (block sizes, warps, stages); None = its defaults


def _flex_fn(key, device_type="cuda"):
    """
    flex_attention compiled once per pass (eager flex_attention is a slow, unfused reference). Each pass has its own
    compiled function, so that neither sees the other's score_mod or value width: alternating them in one function
    recompiles at every call and, past torch._dynamo's recompile limit, silently falls back to the eager reference.
    """
    from torch.nn.attention.flex_attention import flex_attention
    if device_type != "cuda":                 # CPU: the unfused reference (slow; for tests)
        return flex_attention
    if key not in _FLEX:
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        _FLEX[key] = torch.compile(flex_attention, dynamic=False)
    return _FLEX[key]


def _causal_mask(N, device):
    key = (N, str(device))
    if key not in _MASKS:
        from torch.nn.attention.flex_attention import create_block_mask
        _MASKS[key] = create_block_mask(lambda b, h, qi, kj: qi >= kj, None, None, N, N, device=device)
    return _MASKS[key]


@torch.compiler.disable      # inside a compiled model the flex call does not lower (inductor); it is compiled on its own
def _two_flex_passes(q, k, v, tpos, q_sq, k_sq, mu, v_inf, s_inf, inv_nu, coef, robust, causal):
    """The two FlexAttention passes: (o1, lse1, lse2), [B, H, N, ...]. Captured tensors are [B, H, N] (per query row)."""
    N = q.shape[2]

    def score_mod(k_pow):
        def mod(score, b, h, qi, kj):
            lag = tpos[b, qi] - tpos[b, kj]
            lag = lag.clamp_min(0) if causal else lag.abs()       # partly masked blocks evaluate future pairs too
            log_E = -mu[b, h, qi] * lag
            E = torch.exp(log_E)
            E2 = E * E
            Sigma = s_inf[b, h, qi] - v_inf[b, h, qi] * E2
            x = (k_sq[b, h, kj] + E2 * q_sq[b, h, qi] - 2 * E * score.float()).clamp_min(0) * inv_nu[b, h, qi] / Sigma
            content = -coef[b, h, qi] * (torch.log1p(x) if robust else x)
            return content - torch.log(Sigma) + k_pow * log_E
        return mod

    dev = q.device.type
    mask = _causal_mask(N, q.device) if causal else None
    opts = {} if KERNEL_OPTIONS is None else {"kernel_options": KERNEL_OPTIONS}
    with torch.autocast(device_type=dev, enabled=False):
        o1, lse1 = _flex_fn(1, dev)(q, k, v, score_mod=score_mod(1), block_mask=mask, scale=1.0, return_lse=True,
                                    **opts)
        _, lse2 = _flex_fn(2, dev)(q, k, v[..., :16].contiguous(), score_mod=score_mod(2), block_mask=mask, scale=1.0,
                                   return_lse=True, **opts)
    return o1, lse1, lse2


@torch.compiler.disable
def biased_attention(q, k, v, slopes, scale, causal=True):
    """
    Softmax attention with a linear distance bias, scale q.k - slope_h |i - j| (ALiBi and the coupled bias of the
    baselines), through FlexAttention: (out, lse), [B, H, N, D] and [B, H, N]. slopes: [H] (no gradient).
    """
    def mod(score, b, h, qi, kj):
        return score - slopes[h] * (qi - kj).abs()

    dev = q.device.type
    mask = _causal_mask(q.shape[2], q.device) if causal else None
    with torch.autocast(device_type=dev, enabled=False):
        return _flex_fn("bias", dev)(q, k, v, score_mod=mod, block_mask=mask, scale=scale, return_lse=True)


def flex_update(p, t, Q, K, V, V_rot, cos, sin, config, tau, causal=True):
    """
    The same two passes through FlexAttention: the weights are never materialized (memory linear in the length).
    Q, K, V_rot are flattened to [B, H, N, 2D] real vectors (Re<Q~, K~> is their dot product). Under bf16 autocast
    q, k, v enter the kernel in bf16, and the squared norms are taken from the same rounded vectors, so that
    R^2 = |k - E q|^2 stays consistent (no cancellation between rounded and unrounded terms). Only the two kernel
    calls are excluded from an enclosing torch.compile; the rest of the update compiles with the model.
    """
    B, N, H, D, _ = Q.shape
    low = torch.is_autocast_enabled(Q.device.type)
    dtype = torch.get_autocast_dtype(Q.device.type) if low else Q.dtype
    flat = lambda X: X.flatten(-2).transpose(1, 2).to(dtype).contiguous()          # [B, H, N, 2D]
    q, k, v = flat(Q), flat(K), flat(V_rot)
    q_sq, k_sq = q.float().pow(2).sum(-1), k.float().pow(2).sum(-1)               # [B, H, N], from the kernel's q, k
    tpos = t.to(torch.float32).expand(B, N).contiguous()
    # per-head constants: Sigma_n = (v_inf + eta^2) - v_inf E^2 (mu > 0), content = -coef log(1 + R^2 / (nu Sigma_n))
    mu, eta, nu = (p[n].float() for n in ("mu", "eta_sq", "nu"))
    v_inf = p["sigma_sq"].float() / (2 * mu)
    s_inf, inv_nu = v_inf + eta, 1 / nu
    coef = (nu + config.head_dim) / config.head_dim if config.robust_weight else torch.ones_like(nu)
    if tau is not None:
        coef = coef * tau.float()
    # one copy of each per-head scalar per query row, [B, H, N]: the kernel's backward adds every pair's gradient
    # into the captured tensor, and with one value per head those additions all collide (7x slower per layer);
    # per-row copies spread them, and autograd sums the copies back to one value per head
    rows = lambda s: s.view(1, H, 1).expand(B, H, N).contiguous()
    o1, lse1, lse2 = _two_flex_passes(q, k, v, tpos, q_sq, k_sq, *(rows(s) for s in (mu, v_inf, s_inf, inv_nu, coef)),
                                      config.robust_weight, causal)
    o1 = o1.float().transpose(1, 2).reshape(B, N, H, D, 2)
    lse1, lse2 = lse1.float().transpose(1, 2), lse2.float().transpose(1, 2)              # [B, N, H]
    rot = (cos, sin) if config.rotate_values else (None, None)
    return combine(o1, lse1, lse2, p["log_prior_precision"].float(), V.float(), *rot)


def two_pass_update(p, t, Q, K, V, V_rot, cos, sin, config, tau, causal=True):
    """
    The layer's dV (and step) through the two passes, with the weights materialized. p: sde_params(); t: [B|1, N]
    positions; Q, K: rotated [B, N, H, D, 2]; V, V_rot: values and rotated values; cos, sin: the rotation, applied to
    the aggregate when config.rotate_values (else None).
    """
    lag = t[:, :, None] - t[:, None, :]                          # [B|1, N, N]
    future = lag < 0 if causal else torch.zeros_like(lag, dtype=torch.bool)
    lag = lag.abs()[..., None]                                   # non-causal attention uses |t_i - t_j|
    q_sq = Q.pow(2).sum(dim=(-1, -2))[:, :, None]                # [B, N, 1, H]
    k_sq = K.pow(2).sum(dim=(-1, -2))[:, None]                   # [B, 1, N, H]
    L, log_E = dka_logit(cross_dot(Q, K), lag, q_sq, k_sq, p["mu"], p["sigma_sq"], p["eta_sq"], p["nu"], tau,
                         config.head_dim, config.robust_weight)
    L = L.masked_fill(future[..., None], float("-inf"))
    l1, l2 = L + log_E, L + 2 * log_E
    lse1, lse2 = torch.logsumexp(l1, dim=2), torch.logsumexp(l2, dim=2)   # [B, N, H]
    o1 = torch.einsum("bijh,bjhdc->bihdc", torch.softmax(l1, dim=2), V_rot)
    rot = (cos, sin) if config.rotate_values else (None, None)
    return combine(o1, lse1, lse2, p["log_prior_precision"], V, *rot)


# ----------------------------------------------------------------------------------------------------------------
# RFA. Its estimate sum_j softmax(L)_ij E_ij V~_j decays the weights after the softmax, so it is normalized by
# sum a, not sum a E: pass 1 (logits L + log E, values V~) gives o1 and lse1, pass 2 (logits L) gives lse0, and
#     est = e^{lse1 - lse0} Phi+ o1                                                    (flat prior)
#     dV = (w1 Phi+ o1 - s V) / (n + s),  (w1, s, n) = exp((lse1, lse0, null) - max)   (null slot: combine)
# i.e. DKA's two passes with the powers (1, 0) of E instead of (1, 2).
# ----------------------------------------------------------------------------------------------------------------

def rfa_logit(cross, lag, q_sq, k_sq, mu, process_var, eta_sq, gamma_sq, nu, tau, config):
    """
    RFA's logit L of one query-key pair and log E (broadcastable tensors), as the layer computes them for mu > 0:
        log E = clamp(-mu lag, min_exponent, max_exponent),   R^2 = |q|^2 + E^2 |k|^2 - 2 E Re<q, k>
        (-E Re<q, k> without the full norm),   Sigma = v (1 - E^2) + eta^2 E^2 + gamma^2 + cov_eps,
        v = sigma^2 / (2 mu) ("variance") or sigma~^2 ("steady_state"),   L = tau * robust_logits(R^2, Sigma).
    Sigma is evaluated as (v + gamma^2 + cov_eps) + (eta^2 - v) E^2, the form a score_mod uses. lag >= 0.
    """
    c = config
    log_E = torch.clamp(-mu * lag, min=c.min_exponent, max=c.max_exponent)
    E = torch.exp(log_E)
    E2 = E * E
    v = process_var / (2 * mu) if c.process_noise == "variance" else process_var
    Sigma = (v + gamma_sq + c.cov_epsilon) + (eta_sq - v) * E2
    R_sq = q_sq + E2 * k_sq - 2 * E * cross if c.full_residual_norm else -E * cross
    return tau * robust_logits(R_sq, Sigma, nu, c.head_dim, c.robust_weight, c.dle_bias, c.dle_gate), log_E


def rfa_combine(o1, lse1, lse0, null_logit, V, cos, sin, output):
    """RFA's output from the two passes: the innovation or the estimate, flat prior (null_logit None) or null slot."""
    if null_logit is None:
        est = torch.exp(lse1 - lse0)[..., None, None] * o1
        if cos is not None:
            est = rotate(est, cos, sin)
        return est - V if output == "innovation" else est
    dV, _ = combine(o1, lse1, lse0, null_logit, V, cos, sin)
    return dV if output == "innovation" else V + dV


def rfa_two_pass_update(p, t, Q, K, V, V_agg, cos, sin, config, null_logit, causal=True):
    """
    RFA's delta through the two passes, with the weights materialized (the reference for the fused kernel).
    p: sde_params(); t: [B|1, N]; Q, K: rotated [B, N, H, D, 2]; V, V_agg: values and aggregated (rotated) values;
    cos, sin: the counter-rotation when config.rotate_values (else None); null_logit: [H] (prior="tokens") or None.
    """
    lag = t[:, :, None] - t[:, None, :]
    future = lag < 0 if causal else torch.zeros_like(lag, dtype=torch.bool)
    lag = lag.abs()[..., None]
    q_sq = Q.pow(2).sum(dim=(-1, -2))[:, :, None]
    k_sq = K.pow(2).sum(dim=(-1, -2))[:, None]
    L, log_E = rfa_logit(cross_dot(Q, K), lag, q_sq, k_sq, p["mu"], p["process_var"], p["eta_sq"], p["gamma_sq"],
                         p["nu"], p["tau"], config)
    L = L.masked_fill(future[..., None], float("-inf"))
    l1 = L + log_E
    lse1, lse0 = torch.logsumexp(l1, dim=2), torch.logsumexp(L, dim=2)
    o1 = torch.einsum("bijh,bjhdc->bihdc", torch.softmax(l1, dim=2), V_agg)
    return rfa_combine(o1, lse1, lse0, null_logit, V, cos, sin, config.output)


@torch.compiler.disable      # as _two_flex_passes
def _rfa_flex_passes(q, k, v, tpos, q_sq, k_sq, mu, s_inf, d_eta, inv_nu, tau, coef, config, causal):
    """RFA's two FlexAttention passes: (o1, lse1, lse0), [B, H, N, ...]. Captured tensors are [B, H, N]."""
    c = config
    N, f32 = q.shape[2], lse_dtype(q.dtype)
    lo, hi, inv_sqrt_d = c.min_exponent, c.max_exponent, 1.0 / c.head_dim ** 0.5

    def score_mod(k_pow):
        def mod(score, b, h, qi, kj):
            lag = tpos[b, qi] - tpos[b, kj]
            lag = lag.clamp_min(0) if causal else lag.abs()       # partly masked blocks evaluate future pairs too
            log_E = (-mu[b, h, qi] * lag).clamp(lo, hi)
            E = torch.exp(log_E)
            E2 = E * E
            Sigma = s_inf[b, h, qi] + d_eta[b, h, qi] * E2
            cross = score.to(f32)
            if c.full_residual_norm:
                R_sq = (q_sq[b, h, qi] + E2 * k_sq[b, h, kj] - 2 * E * cross).clamp_min(0)
            else:
                R_sq = -E * cross
            x = R_sq * (inv_nu[b, h, qi] / Sigma if c.dle_gate else inv_sqrt_d)
            L = -coef[b, h, qi] * (torch.log1p(x) if c.robust_weight else x)
            if c.dle_bias:
                L = L - tau[b, h, qi] * torch.log(Sigma)
            return L + log_E if k_pow == 1 else L
        return mod

    dev = q.device.type
    mask = _causal_mask(N, q.device) if causal else None
    opts = {} if KERNEL_OPTIONS is None else {"kernel_options": KERNEL_OPTIONS}
    flags = ("rfa", c.full_residual_norm, c.dle_gate, c.robust_weight, c.dle_bias)
    with torch.autocast(device_type=dev, enabled=False):
        o1, lse1 = _flex_fn(flags + (1,), dev)(q, k, v, score_mod=score_mod(1), block_mask=mask, scale=1.0,
                                               return_lse=True, **opts)
        _, lse0 = _flex_fn(flags + (0,), dev)(q, k, v[..., :16].contiguous(), score_mod=score_mod(0), block_mask=mask,
                                              scale=1.0, return_lse=True, **opts)
    return o1, lse1, lse0


def lse_dtype(dtype):
    """The precision of the logits and log-sum-exps: at least float32 (float64 stays float64, for tests)."""
    return torch.promote_types(dtype, torch.float32)


def rfa_flex_update(p, t, Q, K, V, V_agg, cos, sin, config, null_logit, causal=True):
    """
    rfa_two_pass_update through FlexAttention (weights never materialized). As flex_update: q, k, v in the autocast
    dtype, squared norms from the kernel's own q, k, per-head scalars copied per query row, everything elementwise in
    at least float32.
    """
    c = config
    B, N, H, D, _ = Q.shape
    low = torch.is_autocast_enabled(Q.device.type)
    dtype = torch.get_autocast_dtype(Q.device.type) if low else Q.dtype
    f32 = lse_dtype(dtype)
    flat = lambda X: X.flatten(-2).transpose(1, 2).to(dtype).contiguous()          # [B, H, N, 2D]
    q, k, v = flat(Q), flat(K), flat(V_agg)
    q_sq, k_sq = q.to(f32).pow(2).sum(-1), k.to(f32).pow(2).sum(-1)
    tpos = t.to(f32).expand(B, N).contiguous()
    # per-head constants: Sigma = (v + gamma^2 + cov_eps) + (eta^2 - v) E^2, L = tau (-log Sigma - kappa log(1 + x))
    mu, var, eta, gamma, nu, tau = (p[n].to(f32) for n in ("mu", "process_var", "eta_sq", "gamma_sq", "nu", "tau"))
    v_ss = var / (2 * mu) if c.process_noise == "variance" else var
    s_inf, d_eta, inv_nu = v_ss + gamma + c.cov_epsilon, eta - v_ss, 1 / nu
    coef = tau * (nu + c.head_dim) / c.head_dim if c.robust_weight else tau
    rows = lambda s: s.view(1, H, 1).expand(B, H, N).contiguous()                    # see flex_update
    o1, lse1, lse0 = _rfa_flex_passes(q, k, v, tpos, q_sq, k_sq, *(rows(s) for s in (mu, s_inf, d_eta, inv_nu, tau, coef)),
                                      c, causal)
    o1 = o1.to(f32).transpose(1, 2).reshape(B, N, H, D, 2)
    lse1, lse0 = lse1.to(f32).transpose(1, 2), lse0.to(f32).transpose(1, 2)            # [B, N, H]
    null = None if null_logit is None else null_logit.to(f32)
    return rfa_combine(o1, lse1, lse0, null, V.to(f32), cos, sin, c.output)
