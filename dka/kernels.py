"""
Pure functions implementing the Isotropic RFA and DKA computations.

Layout conventions:
    complex vectors     [..., D, 2]   (last dim = real, imaginary)
    pairwise tensors    [B, N, N, H]  (query i, key j, head h; B may be 1 for shared time grids)
"""

import torch
import torch.nn.functional as F


def rotate(x, cos, sin):
    """Complex multiplication x * (cos + i sin). x: [..., D, 2]; cos, sin: broadcastable to [..., D]."""
    re = x[..., 0] * cos - x[..., 1] * sin
    im = x[..., 0] * sin + x[..., 1] * cos
    return torch.stack([re, im], dim=-1)


def rotation_angles(omega, t):
    """omega [H, D], t [B, N] -> (cos, sin), each [B, N, H, D]."""
    theta = omega * t[..., None, None]
    return torch.cos(theta), torch.sin(theta)


def pairwise_lags(t):
    """t [B, N] -> |t_i - t_j| as [B, N, N, 1]."""
    return (t[:, :, None] - t[:, None, :]).abs()[..., None]


def lag_covariance(mu, lags, decay_exp, process_var, eta_sq, gamma_sq, process_noise, mu_eps=1e-5, cov_eps=1e-2):
    """
    Isotropic DLE covariance of a transported key as a function of lag:

        Sigma(dt) = Var_process(dt) + eta^2 e^{-2 mu dt} + gamma^2 + cov_eps

    with Var_process(dt) = sigma^2 (1 - e^{-2 mu dt}) / (2 mu)   ("variance")
                     or  = sigma~^2 (1 - e^{-2 mu dt})            ("steady_state"),
    and the Brownian limit sigma^2 * dt for mu -> 0 in both cases.

    mu, process_var, eta_sq, gamma_sq: [H]. lags: [B, N, N, 1]. decay_exp = clamp(-mu * lags): [B, N, N, H].
    """
    E = torch.exp(decay_exp)
    unitary = mu.abs() < mu_eps
    one_minus_E2 = -torch.expm1(2 * decay_exp)
    if process_noise == "variance":
        denom = torch.where(unitary, torch.ones_like(mu), 2 * mu)
        var_process = process_var * torch.where(unitary, lags, one_minus_E2 / denom)
    elif process_noise == "steady_state":
        var_process = torch.where(unitary, process_var * lags, process_var * one_minus_E2)
    else:
        raise ValueError(f"Unknown process_noise {process_noise!r}")
    return var_process + eta_sq * E**2 + gamma_sq + cov_eps


def residual_norm_sq(Q_rot, K_rot, E, full=True):
    """
    ||q_i - E_ij k_j||^2 = ||q_i||^2 + E_ij^2 ||k_j||^2 - 2 E_ij Re<q_i, k_j>,  in the rotated frame.

    Q_rot, K_rot: [B, N, H, D, 2]; E: [B, N, N, H]. Returns [B, N, N, H].
    With full=False only the cross term -E_ij Re<q_i, k_j> is returned (dot-product attention).
    """
    cross = 2 * E * torch.einsum("bihdc,bjhdc->bijh", Q_rot, K_rot)
    if not full:
        return -cross / 2
    q_sq = Q_rot.pow(2).sum(dim=(-1, -2))[:, :, None, :]
    k_sq = K_rot.pow(2).sum(dim=(-1, -2))[:, None, :, :]
    return q_sq + E**2 * k_sq - cross


def robust_logits(R_sq, Sigma, nu, d, robust_weight=True, dle_bias=True, dle_gate=True, kappa=None):
    """
    Attention logits from the Student-t (robust_weight=True) or exponential likelihood
    of each transported key:

        robust:      log P - kappa * log(1 + R^2 / (nu Sigma)),   kappa = (nu + d) / d
        exponential: log P - R^2 / (nu Sigma)

    where P = 1 / Sigma. dle_bias=False drops log P; dle_gate=False replaces 1/(nu Sigma) by 1/sqrt(d).
    Pass kappa=1 for the unsharpened log-likelihood (used for the total-precision diagnostic).
    """
    bias = -torch.log(Sigma) if dle_bias else 0.0
    gate = 1.0 / (nu * Sigma) if dle_gate else 1.0 / d**0.5
    if robust_weight:
        if kappa is None:
            kappa = (nu + d) / d
        return bias - kappa * torch.log(1 + gate * R_sq)
    return bias - gate * R_sq


def mixture_logits(R_sq, K_sq, Sigma, Sigma_inf, d, beta, tau=None):
    """
    Logits L = log a = -log Sigma_n + log w of the contamination model (Appendix: attention weights from a
    contamination model): a key is relevant, N(E Q~_i, Sigma_n), or background, N(0, Sigma_inf), and

        w = P(relevant | K~_j) = sigmoid(lambda + beta),
        lambda = log(Sigma_inf / Sigma_n) + tau (|K_j|^2 / Sigma_inf - R^2 / Sigma_n) / d,

    the log-likelihood ratio tempered by 2 / d (and tau on its content term). log sigmoid(x) = -softplus(-x), so
    w <= 1 and a <= 1 / Sigma_n. R_sq, Sigma: [B|1, N, N, H]; K_sq: [B, 1, N, H]; Sigma_inf, beta, tau: [H].
    """
    content = K_sq / Sigma_inf - R_sq / Sigma
    if tau is not None:
        content = tau * content
    lam = torch.log(Sigma_inf / Sigma) + content / d
    return -torch.log(Sigma) - F.softplus(-(lam + beta))


def temper_content(logits, Sigma, tau):
    """
    Robust logits L = -log Sigma + content with only the content term multiplied by tau:

        -log Sigma + tau * (L + log Sigma)  =  -log Sigma - tau kappa log(1 + R^2 / (nu Sigma))

    In agreement (R = 0) the content term is 0, so a key's weight is still 1/Sigma whatever tau: S stays the
    information, the null slot keeps its calibration and a <= 1/Sigma. tau frees the exponent from nu (a Pearson
    type VII likelihood, of which the Student-t is the case tau = 1). Masked logits (-inf) stay -inf for tau > 0.
    """
    log_sigma = torch.log(Sigma)
    return tau * (logits + log_sigma) - log_sigma


# ---------------------------------------------------------------------------------------------------
# DKA (Dynamical_Kernel_Attention.tex)
# ---------------------------------------------------------------------------------------------------

def _one_minus_exp_ratio(x):
    """(1 - e^{-x}) / x for x >= 0, with its limit 1 at x = 0."""
    small = x < 1e-4
    x_safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 - x / 2 + x * x / 6, -torch.expm1(-x_safe) / x_safe)


def forcing_variance(mu, sigma_sq, lags):
    """
    Variance of the forcing accumulated over a lag (the DLE solution, Eq. 3):

        V(dt) = sigma^2 (1 - e^{-2 mu dt}) / (2 mu) = v_inf (1 - E^2),

    computed as sigma^2 dt (1 - e^{-x}) / x with x = 2 mu dt, which is stable for all mu >= 0 and tends to
    the Brownian sigma^2 dt as mu -> 0. mu, sigma_sq: [H]; lags: [..., 1]. Returns [..., H].
    """
    return sigma_sq * lags * _one_minus_exp_ratio(2 * mu * lags)


def inflation_terms(theta, mu, lags):
    """
    Observation coefficient h and noise coefficient c of a transported token (Appendix F).

    The past state's prior variance is inflated from v_inf to v_inf + m, with zeta = m / (v_inf + m) =
    sigmoid(theta):
        log h = log sigmoid(theta - 2 mu dt) - log sigmoid(theta),   c = sigmoid(-theta) + sigmoid(theta) E^2,
    so h = E^2 / c. theta=None is DKA's stationary prior (zeta = 0): log h = -2 mu dt and c = 1.
    theta -> +inf gives RFA's flat prior (h = 1, c = E^2).

    theta, mu: [H]; lags: [..., 1]. Returns (log_h [..., H], c [..., H] or the float 1.0).
    """
    two_mu_dt = 2 * mu * lags
    if theta is None:
        return -two_mu_dt, 1.0
    log_h = F.logsigmoid(theta - two_mu_dt) - F.logsigmoid(theta)
    c = torch.sigmoid(-theta) + torch.sigmoid(theta) * torch.exp(-two_mu_dt)
    return log_h, c


def dka_covariance(mu, sigma_sq, eta_sq, lags, c=1.0):
    """
    Sigma_n = V(dt) + c eta^2 (Eq. 14; Sigma_zeta of Appendix F when c != 1). The covariance of the
    transported measurement is h Sigma_n, and the factor h cancels from the Mahalanobis distance.
    Unlike RFA's covariance, the measurement noise eta^2 is not decayed and there is no floor gamma^2.
    """
    return forcing_variance(mu, sigma_sq, lags) + c * eta_sq


def dka_residual_norm_sq(Q_rot, K_rot, E, h, c=1.0):
    """
    Squared key-frame residual, |E K_j - h Q_i|^2 / h in the rotated frame:

        c ||K_j||^2 + h ||Q_i||^2 - 2 E Re<Q~_i, K~_j>        (c = E^2 / h),

    which without inflation (h = E^2, c = 1) is ||K~_j - E Q~_i||^2 (Eq. 16): the key compared with the
    backward conditional mean E Q~_i. Divided by Sigma_n it is the squared Mahalanobis distance d_ij^2.
    The query and key swap roles relative to RFA's ||Q~_i - E K~_j||^2.

    Q_rot, K_rot: [B, N, H, D, 2]; E, h, c: [B|1, N, N, H]. Returns [B, N, N, H], clamped at 0 against
    round-off in the expansion.
    """
    cross = 2 * E * torch.einsum("bihdc,bjhdc->bijh", Q_rot, K_rot)
    q_sq = Q_rot.pow(2).sum(dim=(-1, -2))[:, :, None, :]
    k_sq = K_rot.pow(2).sum(dim=(-1, -2))[:, None, :, :]
    return (c * k_sq + h * q_sq - cross).clamp_min(0)


def calibrated_evidence(mu, sigma_sq, eta_sq, length, theta=None):
    """
    S_bar: the evidence S_i a head gathers with unit robust weights when every token of a context of
    `length` unit-spaced tokens agrees with the query (Appendix G), computed exactly as the sum

        S_bar = sum_{dt=0}^{length-1} h(dt) / Sigma_n(dt).

    Over a finite context it is finite for slow and zero-decay heads too. Used by the "window" calibration of
    the proximal prior, gamma^-2 = c_h S_bar. mu, sigma_sq, eta_sq, theta: [H]. Returns [H].
    """
    lags = torch.arange(length, dtype=mu.dtype, device=mu.device)[:, None]
    log_h, c = inflation_terms(theta, mu, lags)
    return (torch.exp(log_h) / dka_covariance(mu, sigma_sq, eta_sq, lags, c)).sum(0)
