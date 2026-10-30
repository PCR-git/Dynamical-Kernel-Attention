import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DKAConfig
from .init import (inv_softplus, init_complex_matrix, init_complex_linear_,
                   init_rope, init_spectrally_coupled_rope, init_decay_per_head)
from .kernels import (rotate, rotation_angles, pairwise_lags, robust_logits, inflation_terms, dka_covariance,
                      dka_residual_norm_sq, calibrated_evidence)


class MultiheadIsotropicDKA(nn.Module):
    """
    Multi-head Isotropic Dynamical Kernel Attention (Algorithm 1 of Dynamical_Kernel_Attention.tex).

    Each head treats past tokens, transported to the query time by the dynamics lambda_k = -mu + i omega_k,
    as measurements of the current state conditioned under the stationary process: a token at lag dt observes
    the current state with coefficient h = E^2 and noise E^2 Sigma_n, Sigma_n = v_inf (1 - E^2) + eta^2.
    The layer takes one proximal robust estimation step from the current estimate V_i:

        dV_i = (Phi+_i * sum_j a_ij E_ij V~_j - S_i V_i) / (gamma^-2 + S_i),   S_i = sum_j a_ij h_ij,

    with unnormalized weights a_ij = w_ij / Sigma_n. The proximal prior acts as a null slot with logit
    -log gamma^2, which is included in the row maximum used to stabilize the exponentials. The output
    W_o dV is added to the residual stream.

    Parameters (per head):
        omega      rotation frequencies [H, head_dim/2] (buffer unless learn_rotations)
        mu         decay rate [H] (fixed buffer)
        sigma_raw  process noise; softplus gives sigma^2 or v_inf depending on config.process_noise
        eta_raw    measurement noise, eta^2 = softplus(eta_raw) + eps
        nu_raw     Student-t degrees of freedom, nu = d * softplus(nu_raw) + 2 + eps
        log_c      prior="tokens": gamma^-2 = exp(log_c) / eta^2; prior="window": gamma^-2 = exp(log_c) * S_bar
        gamma_raw  prior="direct": gamma^2 = softplus(gamma_raw) + eps
        theta      inflation zeta = sigmoid(theta) (only if config.inflation is not None)
    """

    def __init__(self, config: DKAConfig):
        super().__init__()
        self.config = c = config
        H, D = c.n_heads, c.head_dim
        width = H * D

        self.W_q = nn.Linear(c.d_model, 2 * width)
        self.W_k = nn.Linear(c.d_model, 2 * width)
        self.W_v = nn.Linear(c.d_model, 2 * width)
        self.W_o = nn.Linear(2 * width, c.d_model)

        # --- Dynamics (as in RFA) ---
        n_freqs = D // 2
        omega_sc, mu = init_spectrally_coupled_rope(H, n_freqs, b=c.damping, zero_frac=c.zero_decay_frac)
        omega = omega_sc if c.frequency_init == "spectral" else init_rope(H, n_freqs)
        if c.decay_init == "log_linear":
            mu = init_decay_per_head(H, min_decay=0.01, max_decay=0.95, zero_frac=c.zero_decay_frac)
        mu = mu * c.decay_scale
        if c.learn_rotations:
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega)
        self.register_buffer("mu", mu)

        # --- Noise: equal rho = v_inf / eta^2 across heads at initialization ---
        if c.process_noise == "variance":
            sigma_init = inv_softplus(torch.where(mu > 0, 2 * mu * c.init_steady_state_var,
                                                  torch.full_like(mu, c.init_steady_state_var)))
        else:
            sigma_init = inv_softplus(torch.full((H,), c.init_steady_state_var))
        self.sigma_raw = nn.Parameter(sigma_init)
        self.eta_raw = nn.Parameter(inv_softplus(torch.full((H,), c.init_measurement_var)))
        self.nu_raw = nn.Parameter(torch.full((H,), c.init_nu_over_d))

        # --- Proximal prior ---
        if c.prior != "direct":
            self.log_c = nn.Parameter(torch.full((H,), math.log(c.init_prior_weight)))
        else:
            self.gamma_raw = nn.Parameter(inv_softplus(torch.full((H,), c.init_prior_var)))

        # --- Variance inflation ---
        if c.inflation is not None:
            zeta = min(max(c.inflation, 1e-4), 1 - 1e-4)
            theta = torch.full((H,), math.log(zeta / (1 - zeta)))
            if c.learn_inflation:
                self.theta = nn.Parameter(theta)
            else:
                self.register_buffer("theta", theta)
        else:
            self.theta = None

        init_complex_linear_(self.W_q, init_complex_matrix(c.d_model, width), "in")
        init_complex_linear_(self.W_k, init_complex_matrix(c.d_model, width), "in")
        init_complex_linear_(self.W_v, init_complex_matrix(c.d_model, width), "in")
        init_complex_linear_(self.W_o, init_complex_matrix(width, c.d_model), "out")

    # ------------------------------------------------------------------

    def sde_params(self):
        """
        Constrained per-head parameters: dict of [H] tensors (omega is [H, head_dim]). sigma_sq is always the
        forcing per unit time, whichever quantity is learned; log_prior_precision is log gamma^-2.
        """
        c = self.config
        mu = self.mu
        omega = c.omega_rescale * torch.stack([self.omega, -self.omega], dim=-1).flatten(-2)
        learned = F.softplus(self.sigma_raw)
        sigma_sq = learned if c.process_noise == "variance" else torch.where(mu > 0, 2 * mu * learned, learned)
        eta_sq = F.softplus(self.eta_raw) + c.epsilon
        nu = c.head_dim * F.softplus(self.nu_raw) + 2.0 + c.epsilon
        if c.prior == "tokens":
            log_prior_precision = self.log_c - torch.log(eta_sq)     # c_h I(0), I(0) = h(0) / Sigma_n(0) = 1 / eta^2
        elif c.prior == "window":
            S_bar = calibrated_evidence(mu, sigma_sq, eta_sq, c.calibration_length, self.theta)
            log_prior_precision = self.log_c + torch.log(S_bar)
        else:
            log_prior_precision = -torch.log(F.softplus(self.gamma_raw) + c.epsilon)
        return dict(mu=mu, omega=omega, sigma_sq=sigma_sq, eta_sq=eta_sq, nu=nu, theta=self.theta,
                    log_prior_precision=log_prior_precision)

    def _project_in(self, W, Z):
        B, N, _ = Z.shape
        return W(Z).view(B, N, 2, self.config.n_heads, self.config.head_dim).permute(0, 1, 3, 4, 2)

    def forward(self, Z_q, Z_k=None, Z_v=None, t_measure=None, causal=True, return_metadata=False):
        """
        Args:
            Z_q, Z_k, Z_v: real inputs [B, N, d_model] (Z_k, Z_v default to Z_q).
            t_measure:     observation times [B, N]; None means regular positions 0..N-1.
            causal:        restrict attention to j <= i.
            return_metadata: also return attention weights, evidence and other diagnostics.

        Returns:
            out [B, N, d_model] = W_o dV (to be added to the residual stream), and a dict with the per-head
            parameters (plus diagnostics if return_metadata).
        """
        c = self.config
        Z_k = Z_q if Z_k is None else Z_k
        Z_v = Z_q if Z_v is None else Z_v
        B, N, _ = Z_q.shape
        p = self.sde_params()

        if t_measure is None:
            t = torch.arange(N, device=Z_q.device, dtype=p["mu"].dtype)[None]
        else:
            t = t_measure.to(p["mu"].dtype)

        # --- Dynamics and conditioned measurement model (all [B|1, N, N, H]) ---
        cos, sin = rotation_angles(p["omega"], t)
        lags = pairwise_lags(t)
        E = torch.exp(-p["mu"] * lags)
        log_h, c_noise = inflation_terms(p["theta"], p["mu"], lags)
        h = E * E if p["theta"] is None else torch.exp(log_h)
        Sigma = dka_covariance(p["mu"], p["sigma_sq"], p["eta_sq"], lags, c_noise)

        # --- Project and rotate into the stationary frame (Phi^-) ---
        Q = rotate(self._project_in(self.W_q, Z_q), cos, -sin)
        K = rotate(self._project_in(self.W_k, Z_k), cos, -sin)
        V = self._project_in(self.W_v, Z_v)          # current estimate, in its own frame
        V_rot = rotate(V, cos, -sin)

        # --- Robust logits L = -log Sigma_n - kappa log(1 + d^2 / nu) ---
        R_sq = dka_residual_norm_sq(Q, K, E, h, c_noise)
        logits = robust_logits(R_sq, Sigma, p["nu"], c.head_dim, c.robust_weight)
        if causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=Z_q.device).tril()
            logits = logits.masked_fill(~mask[None, :, :, None], float("-inf"))

        # --- Stabilized weights: subtract the max over the keys and the null slot (logit -log gamma^2) ---
        null_logit = p["log_prior_precision"]                                  # [H]
        m = torch.maximum(logits.amax(dim=2), null_logit)                       # [B, N, H]
        A = torch.exp(logits - m[:, :, None, :])
        null = torch.exp(null_logit - m)                                        # gamma^-2 e^{-m}
        S = (A * h).sum(dim=2)                                                  # information e^{-m} S_i

        # --- Aggregate decayed values, counter-rotate (Phi^+), proximal innovation ---
        A_hat = A * E
        M = rotate(torch.einsum("bijh,bjhdc->bihdc", A_hat, V_rot), cos, sin)
        denom = null + S                                                        # e^{-m} (gamma^-2 + S_i)
        dV = (M - S[..., None, None] * V) / denom[..., None, None]

        out = self.W_o(dV.permute(0, 1, 4, 2, 3).reshape(B, N, -1))   # [re heads..., im heads...]

        info = self._param_summary(p)
        if return_metadata:
            info.update(self._metadata(p, A_hat, null, S, m, denom, dV))
        return out, info

    # ------------------------------------------------------------------

    def _param_summary(self, p):
        """Per-head physical parameters (keys as in RFA where they mean the same thing)."""
        mu = p["mu"]
        v_inf = torch.where(mu > 0, p["sigma_sq"] / (2 * mu).clamp_min(1e-12), torch.full_like(mu, float("inf")))
        summary = {
            "mu_v": mu, "sigma_sq_v": p["sigma_sq"], "sigma_tilde_sq_v": v_inf, "eta_sq_v": p["eta_sq"],
            "gamma_sq_v": torch.exp(-p["log_prior_precision"]), "nu_over_d": p["nu"] / self.config.head_dim,
        }
        if p["theta"] is not None:
            summary["zeta"] = torch.sigmoid(p["theta"])
        return summary

    def _metadata(self, p, A_hat, null, S, m, denom, dV):
        """
        Diagnostics. The update is dV_i = sum_j attn[i, j] Phi+_i V~_j - step_i V_i with attn = A_hat / denom
        and step = S / denom; 1 - step is the share of the null slot (abstention).
        """
        D = self.config.head_dim
        H = p["mu"].shape[0]
        eigenvals = torch.stack([-p["mu"][:, None].expand(H, D), p["omega"]]).reshape(2, 1, H * D, 1)
        return {
            "attn_mat": A_hat / denom[:, :, None, :],  # weight of each (counter-rotated) value [B, N, N, H]
            "step": S / denom,                       # S_i / (gamma^-2 + S_i), the gain [B, N, H]
            "null_share": null / denom,              # 1 - step
            "log_evidence": torch.log(S) + m,        # log S_i, the information from the context [B, N, H]
            "posterior_var": torch.exp(-m) / denom,  # 1 / (gamma^-2 + S_i)
            "dV": dV,                                # innovation in the eigenbasis [B, N, H, D, 2]
            "eigenvals": eigenvals,
        }
