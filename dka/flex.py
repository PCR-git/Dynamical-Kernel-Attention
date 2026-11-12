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


def _flex_fn(k_pow):
    """
    flex_attention compiled once per pass (eager flex_attention is a slow, unfused reference). Each pass has its own
    compiled function, so that neither sees the other's score_mod or value width: alternating them in one function
    recompiles at every call and, past torch._dynamo's recompile limit, silently falls back to the eager reference.
    """
    if k_pow not in _FLEX:
        import torch._dynamo
        from torch.nn.attention.flex_attention import flex_attention
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        _FLEX[k_pow] = torch.compile(flex_attention, dynamic=False)
    return _FLEX[k_pow]


def _causal_mask(N, device):
    key = (N, str(device))
    if key not in _MASKS:
        from torch.nn.attention.flex_attention import create_block_mask
        _MASKS[key] = create_block_mask(lambda b, h, qi, kj: qi >= kj, None, None, N, N, device=device)
    return _MASKS[key]


def flex_update(p, t, Q, K, V, V_rot, cos, sin, config, tau, causal=True):
    """
    The same two passes through FlexAttention: the weights are never materialized (memory linear in the length).
    Q, K, V_rot are flattened to [B, H, N, 2D] real vectors (Re<Q~, K~> is their dot product). Under bf16 autocast
    q, k, v enter the kernel in bf16, and the squared norms are taken from the same rounded vectors, so that
    R^2 = |k - E q|^2 stays consistent (no cancellation between rounded and unrounded terms).
    """
    B, N, H, D, _ = Q.shape
    low = torch.is_autocast_enabled(Q.device.type)
    dtype = torch.get_autocast_dtype(Q.device.type) if low else Q.dtype
    flat = lambda X: X.flatten(-2).transpose(1, 2).to(dtype).contiguous()          # [B, H, N, 2D]
    q, k, v = flat(Q), flat(K), flat(V_rot)
    q_sq, k_sq = q.float().pow(2).sum(-1), k.float().pow(2).sum(-1)               # [B, H, N], from the kernel's q, k
    tpos = t.to(torch.float32).expand(B, N).contiguous()
    mu, sig, eta, nu = (p[n].float() for n in ("mu", "sigma_sq", "eta_sq", "nu"))
    tau_h = tau.float() if tau is not None else None
    d, robust = config.head_dim, config.robust_weight

    def score_mod(k_pow):
        def mod(score, b, h, qi, kj):
            lag = tpos[b, qi] - tpos[b, kj]
            lag = lag.clamp_min(0) if causal else lag.abs()       # partly masked blocks evaluate future pairs too
            L, log_E = dka_logit(score.float(), lag, q_sq[b, h, qi], k_sq[b, h, kj], mu[h], sig[h], eta[h], nu[h],
                                 None if tau_h is None else tau_h[h], d, robust)
            return L + k_pow * log_E
        return mod

    mask = _causal_mask(N, Q.device) if causal else None
    with torch.autocast(device_type=Q.device.type, enabled=False):
        o1, lse1 = _flex_fn(1)(q, k, v, score_mod=score_mod(1), block_mask=mask, scale=1.0, return_lse=True)
        _, lse2 = _flex_fn(2)(q, k, v[..., :16].contiguous(), score_mod=score_mod(2), block_mask=mask, scale=1.0,
                              return_lse=True)
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
