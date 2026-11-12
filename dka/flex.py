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

from .kernels import cross_dot, rotate


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
