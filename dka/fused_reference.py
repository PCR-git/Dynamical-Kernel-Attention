"""
Fused DKA kernel, stage 0: the specification, and a blocked PyTorch reference that computes exactly what the kernel
will, tile by tile, with a hand-written backward pass.

Per batch element and head, with q, k: [N, 2D] (the rotated queries and keys flattened to real vectors, so that
s_ij = q_i . k_j = Re<Q~_i, K~_j>), v: [N, Dv] (the rotated values), positions t, and per-head constants
mu, v_inf = sigma^2 / (2 mu), s_inf = v_inf + eta^2, inv_nu = 1 / nu and coef (kappa tau for the Student-t weight,
tau for the exponential one), each pair j (j <= i if causal) has

    lag  = t_i - t_j  (|t_i - t_j| if not causal),   log E = -mu lag,   E = e^{log E}
    Sigma = s_inf - v_inf E^2                         (Sigma_n for mu > 0)
    R^2  = max(|k_j|^2 + E^2 |q_i|^2 - 2 E s_ij, 0)
    x    = R^2 inv_nu / Sigma,   phi = log(1 + x) (Student-t) or x (exponential)
    L    = -coef phi - log Sigma,   w = L + log E.

Forward. One pass over the keys keeps, per query, a running maximum m of w and three running sums,

    p_ij = e^{w_ij - m_i},   l1 = sum_j p,   l2 = sum_j p E,   acc = sum_j p v_j,

rescaled by e^{m_old - m_new} when the maximum grows, as in FlashAttention. The outputs are

    o1 = acc / l1 = sum_j softmax_j(w) v_j,   lse1 = m + log l1,   lse2 = m + log l2 = logsumexp_j(w + log E),

the two log-sum-exps of the two-pass form (dka/flex.py): lse1 = log sum_j a E and lse2 = log S_i. Since E <= 1, the
second sum cannot overflow. combine() (null slot, counter-rotation, step) stays outside the kernel.

Backward. With P = softmax_j(w) = e^{w - lse1}, Q = softmax_j(w + log E) = e^{w + log E - lse2}, upstream gradients
dO1 [N, Dv], dlse1, dlse2 [N], and D_i = dO1_i . o1_i (as in FlashAttention):

    dv_j      = sum_i P_ij dO1_i
    G_w       = P (dO1_i . v_j - D_i + dlse1_i) + dlse2_i Q             (gradient of w; also that of L)
    G_x       = -coef phi'(x) G_w,      phi' = 1 / (1 + x) or 1
    G_R2      = G_x inv_nu / Sigma      (0 where R^2 was clamped)
    G_Sigma   = -G_x x / Sigma - G_w / Sigma
    G_logE    = G_w + dlse2_i Q + E (G_R2 (2 E |q_i|^2 - 2 s_ij) - 2 v_inf E G_Sigma)
    G_s       = -2 E G_R2:   dq_i = sum_j G_s k_j,   dk_j = sum_i G_s q_i  (matrix products)
    d|q_i|^2  = sum_j G_R2 E^2,   d|k_j|^2 = sum_i G_R2   (row and column sums; then dq += 2 q d|q|^2, dk += 2 k d|k|^2)
    per head: dcoef = -sum G_w phi,  dinv_nu = sum G_x R^2 / Sigma,  ds_inf = sum G_Sigma,
              dv_inf = -sum G_Sigma E^2,  dmu = -sum G_logE lag.

The kernel computes dK, dV and d|k|^2 looping over query blocks for each key block, and dQ and d|q|^2 looping over
key blocks for each query block, so every sum is a reduction within one program (no atomic additions); the per-head
sums are written as one partial per block and added up afterwards. Masked (future) pairs have P = Q = 0 and so
contribute nothing; their lag is clamped at zero so that nothing overflows.
"""
import torch

from .flex import combine

BLOCK_M, BLOCK_N = 64, 64       # query and key block sizes of the reference (the tests use small odd ones)


def _tile(q, k, t, q_sq, k_sq, consts, i0, i1, j0, j1, robust, causal):
    """Pair terms of queries i0..i1-1 against keys j0..j1-1, [B, H, ni, nj]: w (masked to -inf) and intermediates."""
    mu, v_inf, s_inf, inv_nu, coef = consts
    s = q[:, :, i0:i1] @ k[:, :, j0:j1].transpose(-1, -2)
    lag = (t[:, i0:i1, None] - t[:, None, j0:j1])[:, None]                      # [B|1, 1, ni, nj]
    future = lag < 0 if causal else None
    lag = lag.clamp_min(0) if causal else lag.abs()
    log_E = -mu * lag
    E = torch.exp(log_E)
    Sigma = s_inf - v_inf * E * E
    R2 = k_sq[:, :, None, j0:j1] + E * E * q_sq[:, :, i0:i1, None] - 2 * E * s
    kept = R2 > 0
    R2 = torch.where(kept, R2, torch.zeros_like(R2))
    x = R2 * inv_nu / Sigma
    phi = torch.log1p(x) if robust else x
    w = -coef * phi - torch.log(Sigma) + log_E
    if causal:
        w = w.masked_fill(future, float("-inf"))
    return dict(s=s, lag=lag, log_E=log_E, E=E, Sigma=Sigma, R2=R2, kept=kept, x=x, phi=phi, w=w)


def blocked_forward(q, k, v, t, consts, robust, causal, bm=None, bn=None):
    """The kernel's forward pass, tile by tile: (o1, lse1, lse2). q, k: [B, H, N, 2D]; v: [B, H, N, Dv]; t: [B|1, N]."""
    bm, bn = bm or BLOCK_M, bn or BLOCK_N
    B, H, N, _ = q.shape
    q_sq, k_sq = q.pow(2).sum(-1), k.pow(2).sum(-1)
    o1 = torch.empty(B, H, N, v.shape[-1], dtype=v.dtype, device=v.device)
    lse1, lse2 = (torch.empty(B, H, N, dtype=q.dtype, device=q.device) for _ in range(2))
    for i0 in range(0, N, bm):
        i1 = min(i0 + bm, N)
        m = torch.full((B, H, i1 - i0), float("-inf"), dtype=q.dtype, device=q.device)
        l1, l2 = torch.zeros_like(m), torch.zeros_like(m)
        acc = torch.zeros(B, H, i1 - i0, v.shape[-1], dtype=v.dtype, device=v.device)
        for j0 in range(0, i1 if causal else N, bn):        # causal: key blocks up to the query block's end
            j1 = min(j0 + bn, N)
            tl = _tile(q, k, t, q_sq, k_sq, consts, i0, i1, j0, j1, robust, causal)
            m_new = torch.maximum(m, tl["w"].amax(-1))
            alpha = torch.exp(m - m_new)
            p = torch.exp(tl["w"] - m_new[..., None])
            l1 = alpha * l1 + p.sum(-1)
            l2 = alpha * l2 + (p * tl["E"]).sum(-1)
            acc = alpha[..., None] * acc + p @ v[:, :, j0:j1]
            m = m_new
        o1[:, :, i0:i1] = acc / l1[..., None]
        lse1[:, :, i0:i1] = m + torch.log(l1)
        lse2[:, :, i0:i1] = m + torch.log(l2)
    return o1, lse1, lse2


def blocked_backward(q, k, v, t, consts, robust, causal, o1, lse1, lse2, do1, dlse1, dlse2, bm=None, bn=None):
    """
    The kernel's backward pass from the formulas in the module docstring: (dq, dk, dv, dconsts). The reference
    accumulates everything in one loop; the kernel splits it into a key-major and a query-major pass (see above).
    """
    bm, bn = bm or BLOCK_M, bn or BLOCK_N
    B, H, N, _ = q.shape
    mu, v_inf, s_inf, inv_nu, coef = consts
    q_sq, k_sq = q.pow(2).sum(-1), k.pow(2).sum(-1)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    dq_sq, dk_sq = torch.zeros_like(q_sq), torch.zeros_like(k_sq)
    dmu, dv_inf, ds_inf, dinv_nu, dcoef = (torch.zeros(H, dtype=q.dtype, device=q.device) for _ in range(5))
    D = (do1 * o1).sum(-1)
    head_sum = lambda X: X.sum(dim=(0, 2, 3))
    for j0 in range(0, N, bn):
        j1 = min(j0 + bn, N)
        for i0 in range(j0 // bm * bm if causal else 0, N, bm):     # causal: query blocks from the key block on
            i1 = min(i0 + bm, N)
            tl = _tile(q, k, t, q_sq, k_sq, consts, i0, i1, j0, j1, robust, causal)
            E = tl["E"]
            P = torch.exp(tl["w"] - lse1[:, :, i0:i1, None])
            Qw = torch.exp(tl["w"] + tl["log_E"] - lse2[:, :, i0:i1, None])
            dP = do1[:, :, i0:i1] @ v[:, :, j0:j1].transpose(-1, -2)
            G_w = (P * (dP - D[:, :, i0:i1, None] + dlse1[:, :, i0:i1, None])
                   + dlse2[:, :, i0:i1, None] * Qw)
            dv[:, :, j0:j1] += P.transpose(-1, -2) @ do1[:, :, i0:i1]
            dphi = 1 / (1 + tl["x"]) if robust else torch.ones_like(tl["x"])
            G_x = -coef * dphi * G_w
            G_R2 = torch.where(tl["kept"], G_x * inv_nu / tl["Sigma"], torch.zeros_like(G_x))
            G_Sigma = -(G_x * tl["x"] + G_w) / tl["Sigma"]
            G_logE = (G_w + dlse2[:, :, i0:i1, None] * Qw
                      + E * (G_R2 * (2 * E * q_sq[:, :, i0:i1, None] - 2 * tl["s"]) - 2 * v_inf * E * G_Sigma))
            G_s = -2 * E * G_R2
            dq[:, :, i0:i1] += G_s @ k[:, :, j0:j1]
            dk[:, :, j0:j1] += G_s.transpose(-1, -2) @ q[:, :, i0:i1]
            dq_sq[:, :, i0:i1] += (G_R2 * E * E).sum(-1)
            dk_sq[:, :, j0:j1] += G_R2.sum(-2)
            dcoef -= head_sum(G_w * tl["phi"])
            dinv_nu += head_sum(G_x * tl["R2"] / tl["Sigma"])
            ds_inf += head_sum(G_Sigma)
            dv_inf -= head_sum(G_Sigma * E * E)
            dmu -= head_sum(G_logE * tl["lag"])
    dq += 2 * q * dq_sq[..., None]
    dk += 2 * k * dk_sq[..., None]
    return dq, dk, dv, (dmu, dv_inf, ds_inf, dinv_nu, dcoef)


class BlockedDKA(torch.autograd.Function):
    """(o1, lse1, lse2) with the blocked forward and the hand-written backward (the fused kernel's contract)."""

    @staticmethod
    def forward(ctx, q, k, v, t, mu, v_inf, s_inf, inv_nu, coef, robust, causal, bm, bn):
        consts = tuple(c.view(1, -1, 1, 1) for c in (mu, v_inf, s_inf, inv_nu, coef))
        o1, lse1, lse2 = blocked_forward(q, k, v, t, consts, robust, causal, bm, bn)
        ctx.save_for_backward(q, k, v, t, mu, v_inf, s_inf, inv_nu, coef, o1, lse1, lse2)
        ctx.flags = (robust, causal, bm, bn)
        return o1, lse1, lse2

    @staticmethod
    def backward(ctx, do1, dlse1, dlse2):
        q, k, v, t, mu, v_inf, s_inf, inv_nu, coef, o1, lse1, lse2 = ctx.saved_tensors
        robust, causal, bm, bn = ctx.flags
        do1 = torch.zeros_like(o1) if do1 is None else do1
        dlse1 = torch.zeros_like(lse1) if dlse1 is None else dlse1
        dlse2 = torch.zeros_like(lse2) if dlse2 is None else dlse2
        consts = tuple(c.view(1, -1, 1, 1) for c in (mu, v_inf, s_inf, inv_nu, coef))
        dq, dk, dv, dconsts = blocked_backward(q, k, v, t, consts, robust, causal, o1, lse1, lse2, do1, dlse1, dlse2,
                                               bm, bn)
        return (dq, dk, dv, None, *dconsts, None, None, None, None)


def blocked_update(p, t, Q, K, V, V_rot, cos, sin, config, tau, causal=True):
    """The layer's dV (and step) through BlockedDKA: flex_update's interface, with the per-head constants passed once
    per head (the kernel reduces their gradients per block, so no per-row copies are needed)."""
    B, N, H, D, _ = Q.shape
    f32 = torch.promote_types(Q.dtype, torch.float32)
    flat = lambda X: X.flatten(-2).transpose(1, 2).to(f32).contiguous()            # [B, H, N, 2D]
    q, k, v = flat(Q), flat(K), flat(V_rot)
    mu, eta, nu = (p[n].to(f32) for n in ("mu", "eta_sq", "nu"))
    if config.flat_precision:                                                    # Sigma_n = eta^2 at every lag
        v_inf, s_inf = torch.zeros_like(eta), eta
    else:
        v_inf = p["sigma_sq"].to(f32) / (2 * mu)
        s_inf = v_inf + eta
    inv_nu = 1 / nu
    coef = (nu + config.head_dim) / config.head_dim if config.robust_weight else torch.ones_like(nu)
    if tau is not None:
        coef = coef * tau.to(f32)
    o1, lse1, lse2 = BlockedDKA.apply(q, k, v, t.to(f32), mu, v_inf, s_inf, inv_nu, coef, config.robust_weight,
                                      causal, BLOCK_M, BLOCK_N)
    o1 = o1.transpose(1, 2).reshape(B, N, H, D, 2)
    lse1, lse2 = lse1.transpose(1, 2), lse2.transpose(1, 2)                          # [B, N, H]
    rot = (cos, sin) if config.rotate_values else (None, None)
    return combine(o1, lse1, lse2, p["log_prior_precision"].to(f32), V.to(f32), *rot)



# =====================================================================================================================
# Lean RFA (flat precision, exponential weight): the fused kernel's specification and blocked reference.
#
# With Sigma_0 = eta^2 + gamma^2 + cov_eps at every lag, RFA's logit is tau (-log Sigma_0 - R^2 / (nu Sigma_0)) with
# R^2 = |q_i|^2 + E^2 |k_j|^2 - 2 E s_ij (the key transported to the query) and log E = clamp(-mu lag, lo, hi). The
# kernel computes L = -kappa R^2, kappa = tau / (nu Sigma_0); the constant -tau log Sigma_0 is added to the
# log-sum-exps outside. RFA's estimate decays the weights after the softmax, sum_j softmax_j(L) E V~_j, so the one
# pass keeps the running maximum m of L (not L + log E) and
#     p = e^{L - m},   l0 = sum p,   l1 = sum p E,   acc = sum p E V~,
# giving o1 = acc / l1, lse1 = m + log l1 = logsumexp(L + log E) and lse0 = m + log l0 = logsumexp(L): the outputs of
# rfa_two_pass_update's two passes, combined outside (rfa_combine: flat prior or null slot, estimate or innovation).
#
# Backward. With P1 = softmax_j(L + log E) = e^{L + log E - lse1} and P0 = softmax_j(L) = e^{L - lse0}
# (P1 = P0 E e^{lse0 - lse1}: one exponential per pair), upstream dO1, dlse1, dlse0 and D_i = dO1_i . o1_i:
#     dv_j   = sum_i P1 dO1_i
#     G_w1   = P1 (dO1_i . v_j - D_i + dlse1_i),   G_L = G_w1 + dlse0_i P0          (log E enters only w1 = L + log E)
#     G_R2   = -kappa G_L (0 where R^2 was clamped),   G_s = -2 E G_R2,   dq = sum_j G_s k_j,   dk = sum_i G_s q_i
#     d|q_i|^2 = sum_j G_R2,   d|k_j|^2 = sum_i G_R2 E^2
#     G_logE = G_w1 + E G_R2 (2 E |k_j|^2 - 2 s_ij),   dmu = -sum G_logE lag (0 where log E is clamped)
#     dkappa = -sum G_L R^2.
# =====================================================================================================================

def _rfa_tile(q, k, t, q_sq, k_sq, mu, kappa, lo, hi, i0, i1, j0, j1, causal):
    """Lean RFA pair terms of queries i0..i1-1 against keys j0..j1-1, [B, H, ni, nj]: L (masked) and intermediates."""
    s = q[:, :, i0:i1] @ k[:, :, j0:j1].transpose(-1, -2)
    lag = (t[:, i0:i1, None] - t[:, None, j0:j1])[:, None]
    future = lag < 0 if causal else None
    lag = lag.clamp_min(0) if causal else lag.abs()
    raw = -mu * lag
    log_E = raw.clamp(lo, hi)
    free = (raw >= lo) & (raw <= hi)                         # log E not on its clamp (gradient flows to mu)
    E = torch.exp(log_E)
    R2 = q_sq[:, :, i0:i1, None] + E * E * k_sq[:, :, None, j0:j1] - 2 * E * s
    kept = R2 > 0
    R2 = torch.where(kept, R2, torch.zeros_like(R2))
    L = -kappa * R2
    if causal:
        L = L.masked_fill(future, float("-inf"))
    return dict(s=s, lag=lag, log_E=log_E, free=free, E=E, R2=R2, kept=kept, L=L)


def rfa_blocked_forward(q, k, v, t, mu, kappa, lo, hi, causal, bm=None, bn=None):
    """The lean RFA kernel's forward, tile by tile: (o1, lse1, lse0). mu, kappa: [1, H, 1, 1]; lo, hi: the clamp."""
    bm, bn = bm or BLOCK_M, bn or BLOCK_N
    B, H, N, _ = q.shape
    q_sq, k_sq = q.pow(2).sum(-1), k.pow(2).sum(-1)
    o1 = torch.empty(B, H, N, v.shape[-1], dtype=v.dtype, device=v.device)
    lse1, lse0 = (torch.empty(B, H, N, dtype=q.dtype, device=q.device) for _ in range(2))
    for i0 in range(0, N, bm):
        i1 = min(i0 + bm, N)
        m = torch.full((B, H, i1 - i0), float("-inf"), dtype=q.dtype, device=q.device)
        l0, l1 = torch.zeros_like(m), torch.zeros_like(m)
        acc = torch.zeros(B, H, i1 - i0, v.shape[-1], dtype=v.dtype, device=v.device)
        for j0 in range(0, i1 if causal else N, bn):
            j1 = min(j0 + bn, N)
            tl = _rfa_tile(q, k, t, q_sq, k_sq, mu, kappa, lo, hi, i0, i1, j0, j1, causal)
            m_new = torch.maximum(m, tl["L"].amax(-1))
            alpha = torch.exp(m - m_new)
            p = torch.exp(tl["L"] - m_new[..., None])
            pe = p * tl["E"]
            l0 = alpha * l0 + p.sum(-1)
            l1 = alpha * l1 + pe.sum(-1)
            acc = alpha[..., None] * acc + pe @ v[:, :, j0:j1]
            m = m_new
        o1[:, :, i0:i1] = acc / l1[..., None]
        lse1[:, :, i0:i1] = m + torch.log(l1)
        lse0[:, :, i0:i1] = m + torch.log(l0)
    return o1, lse1, lse0


def rfa_blocked_backward(q, k, v, t, mu, kappa, lo, hi, causal, o1, lse1, lse0, do1, dlse1, dlse0, bm=None, bn=None):
    """The lean RFA kernel's backward from the formulas above: (dq, dk, dv, dmu, dkappa)."""
    bm, bn = bm or BLOCK_M, bn or BLOCK_N
    B, H, N, _ = q.shape
    q_sq, k_sq = q.pow(2).sum(-1), k.pow(2).sum(-1)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    dq_sq, dk_sq = torch.zeros_like(q_sq), torch.zeros_like(k_sq)
    dmu, dkappa = (torch.zeros(H, dtype=q.dtype, device=q.device) for _ in range(2))
    D = (do1 * o1).sum(-1)
    head_sum = lambda X: X.sum(dim=(0, 2, 3))
    for j0 in range(0, N, bn):
        j1 = min(j0 + bn, N)
        for i0 in range(j0 // bm * bm if causal else 0, N, bm):
            i1 = min(i0 + bm, N)
            tl = _rfa_tile(q, k, t, q_sq, k_sq, mu, kappa, lo, hi, i0, i1, j0, j1, causal)
            E = tl["E"]
            P0 = torch.exp(tl["L"] - lse0[:, :, i0:i1, None])
            P1 = P0 * E * torch.exp(lse0 - lse1)[:, :, i0:i1, None]
            dP = do1[:, :, i0:i1] @ v[:, :, j0:j1].transpose(-1, -2)
            G_w1 = P1 * (dP - D[:, :, i0:i1, None] + dlse1[:, :, i0:i1, None])
            G_L = G_w1 + dlse0[:, :, i0:i1, None] * P0
            dv[:, :, j0:j1] += P1.transpose(-1, -2) @ do1[:, :, i0:i1]
            G_R2 = torch.where(tl["kept"], -kappa * G_L, torch.zeros_like(G_L))
            G_s = -2 * E * G_R2
            dq[:, :, i0:i1] += G_s @ k[:, :, j0:j1]
            dk[:, :, j0:j1] += G_s.transpose(-1, -2) @ q[:, :, i0:i1]
            dq_sq[:, :, i0:i1] += G_R2.sum(-1)
            dk_sq[:, :, j0:j1] += (G_R2 * E * E).sum(-2)
            G_logE = G_w1 + E * G_R2 * (2 * E * k_sq[:, :, None, j0:j1] - 2 * tl["s"])
            dmu -= head_sum(torch.where(tl["free"], G_logE * tl["lag"], torch.zeros_like(G_logE)))
            dkappa -= head_sum(G_L * tl["R2"])
    dq += 2 * q * dq_sq[..., None]
    dk += 2 * k * dk_sq[..., None]
    return dq, dk, dv, dmu, dkappa


class BlockedRFALean(torch.autograd.Function):
    """(o1, lse1, lse0) of the lean RFA with the blocked forward and the hand-written backward (the kernel's
    contract). mu, kappa: [H]; lo, hi: the clamp of log E."""

    @staticmethod
    def forward(ctx, q, k, v, t, mu, kappa, lo, hi, causal, bm, bn):
        o1, lse1, lse0 = rfa_blocked_forward(q, k, v, t, mu.view(1, -1, 1, 1), kappa.view(1, -1, 1, 1), lo, hi, causal,
                                             bm, bn)
        ctx.save_for_backward(q, k, v, t, mu, kappa, o1, lse1, lse0)
        ctx.flags = (lo, hi, causal, bm, bn)
        return o1, lse1, lse0

    @staticmethod
    def backward(ctx, do1, dlse1, dlse0):
        q, k, v, t, mu, kappa, o1, lse1, lse0 = ctx.saved_tensors
        lo, hi, causal, bm, bn = ctx.flags
        do1 = torch.zeros_like(o1) if do1 is None else do1
        dlse1 = torch.zeros_like(lse1) if dlse1 is None else dlse1
        dlse0 = torch.zeros_like(lse0) if dlse0 is None else dlse0
        dq, dk, dv, dmu, dkappa = rfa_blocked_backward(q, k, v, t, mu.view(1, -1, 1, 1), kappa.view(1, -1, 1, 1), lo,
                                                       hi, causal, o1, lse1, lse0, do1, dlse1, dlse0, bm, bn)
        return dq, dk, dv, None, dmu, dkappa, None, None, None, None, None


def rfa_blocked_update(p, t, Q, K, V, V_agg, cos, sin, config, null_logit, causal=True):
    """The lean RFA's delta through BlockedRFALean: rfa_two_pass_update's interface. kappa = tau / (nu Sigma_0) and
    the constant -tau log Sigma_0 is added to both log-sum-exps (Sigma_0 = eta^2 + gamma^2 + cov_eps)."""
    from .flex import rfa_combine
    B, N, H, D, _ = Q.shape
    f32 = torch.promote_types(Q.dtype, torch.float32)
    flat = lambda X: X.flatten(-2).transpose(1, 2).to(f32).contiguous()            # [B, H, N, 2D]
    q, k, v = flat(Q), flat(K), flat(V_agg)
    sigma0 = (p["eta_sq"] + p["gamma_sq"] + config.cov_epsilon).to(f32)
    tau = p["tau"].to(f32)
    kappa = tau / (p["nu"].to(f32) * sigma0)
    o1, lse1, lse0 = BlockedRFALean.apply(q, k, v, t.to(f32), p["mu"].to(f32), kappa, config.min_exponent,
                                          config.max_exponent, causal, BLOCK_M, BLOCK_N)
    shift = -tau * torch.log(sigma0)                                                 # [H]
    o1 = o1.transpose(1, 2).reshape(B, N, H, D, 2)
    lse1, lse0 = lse1.transpose(1, 2) + shift, lse0.transpose(1, 2) + shift          # [B, N, H]
    null = None if null_logit is None else null_logit.to(f32)
    return rfa_combine(o1, lse1, lse0, null, V.to(f32), cos, sin, config.output)
