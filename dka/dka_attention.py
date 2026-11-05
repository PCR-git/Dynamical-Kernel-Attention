import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DKAConfig
from .init import (inv_softplus_exact as inv_softplus, init_complex_matrix, init_complex_linear_,
                   init_rope, init_spectrally_coupled_rope, init_decay_per_head)
from .kernels import (rotate, rotation_angles, pairwise_lags, robust_logits, temper_content, mixture_logits, inflation_terms, dka_covariance,
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
        gamma_raw  prior="direct": gamma^2 = softplus(gamma_raw) + eps (prior="none": no null slot, gamma^-2 = 0)
        theta      inflation zeta = sigmoid(theta) (only if config.inflation is not None)
        log_n_sat  evidence saturation, S_max = n_sat / eta^2 (only if config.saturation is not None)
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
        omega_sc, mu = init_spectrally_coupled_rope(H, n_freqs, b=c.damping, zero_frac=c.zero_decay_frac,
                                                    base=c.rope_base)
        omega = omega_sc if c.frequency_init == "spectral" else init_rope(H, n_freqs, base=c.rope_base)
        if c.zero_rotations:
            omega = torch.zeros_like(omega)
        if c.prope_length is not None:               # slow heads: no frequencies below one cycle per prope_length
            slow = (mu * c.decay_scale * c.prope_length < 1)[:, None]                # horizon 1/mu > prope_length
            omega = torch.where(slow & (2 * math.pi / omega > c.prope_length), torch.zeros_like(omega), omega)
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
        if c.prior in ("tokens", "window"):
            self.log_c = nn.Parameter(torch.full((H,), math.log(c.init_prior_weight)))
        elif c.prior == "direct":
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

        # --- Evidence saturation (shared measurement error) ---
        if c.saturation is not None:
            log_n_sat = torch.full((H,), math.log(c.saturation))
            if c.learn_saturation:
                self.log_n_sat = nn.Parameter(log_n_sat)
            else:
                self.register_buffer("log_n_sat", log_n_sat)
        else:
            self.log_n_sat = None

        # --- Inverse temperature on the content term of the logits (not on -log Sigma or the null slot) ---
        self.log_tau = nn.Parameter(torch.zeros(H)) if c.temperature else None

        # --- Contamination model: beta_h = logit of the prior probability of relevance ---
        self.beta = nn.Parameter(torch.full((H,), c.init_relevance_logit)) if c.weight == "mixture" else None

        # --- Evaluation memory: process the queries in blocks of this size (None = all at once) ---
        self.query_chunk = None

        # --- QK normalization: |Q_i| = |K_j| = s_h ---
        self.log_qk_scale = (nn.Parameter(torch.full((H,), 0.5 * math.log(c.init_qk_scale_sq * D)))
                             if c.qk_norm else None)

        init_complex_linear_(self.W_q, init_complex_matrix(c.d_model, width), "in")
        init_complex_linear_(self.W_k, init_complex_matrix(c.d_model, width), "in")
        init_complex_linear_(self.W_v, init_complex_matrix(c.d_model, width), "in")
        init_complex_linear_(self.W_o, init_complex_matrix(width, c.d_model), "out")

    # ------------------------------------------------------------------

    def sde_params(self):
        """
        Constrained per-head parameters: dict of [H] tensors (omega is [H, head_dim]). sigma_sq is always the
        forcing per unit time, whichever quantity is learned; log_prior_precision is log gamma^-2;
        log_saturation is log S_max (None without saturation).
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
        elif c.prior == "none":
            log_prior_precision = torch.full_like(eta_sq, float("-inf"))   # null slot weight exp(-inf) = 0
        else:
            log_prior_precision = -torch.log(F.softplus(self.gamma_raw) + c.epsilon)
        log_saturation = None if self.log_n_sat is None else self.log_n_sat - torch.log(eta_sq)
        return dict(mu=mu, omega=omega, sigma_sq=sigma_sq, eta_sq=eta_sq, nu=nu, theta=self.theta,
                    log_prior_precision=log_prior_precision, log_saturation=log_saturation)

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

        # --- Project and rotate into the stationary frame (Phi^-) ---
        cos, sin = rotation_angles(p["omega"], t)
        Q = rotate(self._project_in(self.W_q, Z_q), cos, -sin)
        K = rotate(self._project_in(self.W_k, Z_k), cos, -sin)
        if self.log_qk_scale is not None:            # rotations preserve the norm
            s = torch.exp(self.log_qk_scale)[:, None, None]
            Q = s * Q / Q.pow(2).sum(dim=(-1, -2), keepdim=True).add(c.epsilon).sqrt()
            K = s * K / K.pow(2).sum(dim=(-1, -2), keepdim=True).add(c.epsilon).sqrt()
        V = self._project_in(self.W_v, Z_v)          # current estimate, in its own frame
        V_rot = rotate(V, cos, -sin) if c.rotate_values else V

        # --- Update, over blocks of queries when query_chunk is set (rows are independent; same result) ---
        chunk = self.query_chunk if self.query_chunk and not return_metadata and N > self.query_chunk else N
        rows = [self._rows(p, t, Q, K, V, V_rot, cos, sin, i0, min(i0 + chunk, N), causal)
                for i0 in range(0, N, chunk)]
        dV = torch.cat([r[0] for r in rows], dim=1)
        S_eff = torch.cat([r[4] for r in rows], dim=1)
        denom = torch.cat([r[6] for r in rows], dim=1)

        X = dV + V if c.output == "estimate" else dV                            # new estimate, or innovation
        out = self.W_o(X.permute(0, 1, 4, 2, 3).reshape(B, N, -1))     # [re heads..., im heads...]

        info = self._param_summary(p)
        info["step_mean"] = (S_eff / denom).detach().mean(dim=(0, 1))   # per-head mean step over batch and positions
        if return_metadata:
            _, A_hat, null, S, _, m, _ = rows[0]
            info.update(self._metadata(p, A_hat, null, S, S_eff, m, denom, dV))
        return out, info

    def _rows(self, p, t, Q, K, V, V_rot, cos, sin, i0, i1, causal):
        """The update of queries i0..i1-1 against all keys: (dV, A_hat, null, S, S_eff, m, denom), rows i0..i1-1."""
        c = self.config
        N = K.shape[1]

        # --- Dynamics and conditioned measurement model (all [B|1, nq, N, H]) ---
        lags = (t[:, i0:i1, None] - t[:, None, :]).abs()[..., None]
        E = torch.exp(-p["mu"] * lags)
        log_h, c_noise = inflation_terms(p["theta"], p["mu"], lags)
        h = E * E if p["theta"] is None else torch.exp(log_h)
        Sigma = dka_covariance(p["mu"], p["sigma_sq"], p["eta_sq"], lags, c_noise)

        # --- Robust logits L = -log Sigma_n - kappa log(1 + d^2 / nu) ---
        R_sq = dka_residual_norm_sq(Q[:, i0:i1], K, E, h, c_noise)
        tau = torch.exp(self.log_tau) if self.log_tau is not None else None
        if c.weight == "mixture":                                               # w = P(relevant | key)
            Sigma_inf = p["sigma_sq"] / (2 * p["mu"]) + p["eta_sq"]
            K_sq = K.pow(2).sum(dim=(-1, -2))[:, None]
            logits = mixture_logits(R_sq, K_sq, Sigma, Sigma_inf, c.head_dim, self.beta, tau)
        else:
            logits = robust_logits(R_sq, Sigma, p["nu"], c.head_dim, c.robust_weight)
            if tau is not None:                                                 # content term only (Pearson VII)
                logits = temper_content(logits, Sigma, tau)
        if causal:
            mask = torch.arange(N, device=K.device)[None, :] <= torch.arange(i0, i1, device=K.device)[:, None]
            logits = logits.masked_fill(~mask[None, :, :, None], float("-inf"))

        # --- Stabilized weights: subtract the max over the keys and the null slot (logit -log gamma^2) ---
        null_logit = p["log_prior_precision"]                                  # [H]
        m = torch.maximum(logits.amax(dim=2), null_logit)                       # [B, nq, H]
        A = torch.exp(logits - m[:, :, None, :])
        null = torch.exp(null_logit - m)                                        # gamma^-2 e^{-m}
        S = (A * h).sum(dim=2)                                                  # information e^{-m} S_i
        if p["log_saturation"] is None:
            S_eff = S
        else:                                                                   # S / (1 + S / S_max), scaled by e^{-m}
            S_eff = S / (1 + S * torch.exp(m - p["log_saturation"]))

        # --- Aggregate decayed values, counter-rotate (Phi^+), proximal innovation ---
        A_hat = A * E
        M = torch.einsum("bijh,bjhdc->bihdc", A_hat, V_rot)
        if c.rotate_values:
            M = rotate(M, cos[:, i0:i1], sin[:, i0:i1])
        if p["log_saturation"] is not None:                                     # same estimate M / S, discounted step
            A_hat = A_hat * (S_eff / S)[:, :, None, :]
            M = M * (S_eff / S)[..., None, None]
        denom = null + S_eff                                                    # e^{-m} (gamma^-2 + S_eff_i)
        dV = (M - S_eff[..., None, None] * V[:, i0:i1]) / denom[..., None, None]
        return dV, A_hat, null, S, S_eff, m, denom

    # ------------------------------------------------------------------

    def _param_summary(self, p):
        """Per-head physical parameters (keys as in RFA where they mean the same thing; prior_weight is c_h)."""
        mu = p["mu"]
        v_inf = torch.where(mu > 0, p["sigma_sq"] / (2 * mu).clamp_min(1e-12), torch.full_like(mu, float("inf")))
        summary = {
            "mu_v": mu, "sigma_sq_v": p["sigma_sq"], "sigma_tilde_sq_v": v_inf, "eta_sq_v": p["eta_sq"],
            "gamma_sq_v": torch.exp(-p["log_prior_precision"]), "nu_over_d": p["nu"] / self.config.head_dim,
        }
        if self.config.prior in ("tokens", "window"):
            summary["prior_weight"] = torch.exp(self.log_c)
        if p["theta"] is not None:
            summary["zeta"] = torch.sigmoid(p["theta"])
        if self.log_n_sat is not None:
            summary["saturation"] = torch.exp(self.log_n_sat)
        if self.beta is not None:
            summary["relevance_prior"] = torch.sigmoid(self.beta)
        return summary

    def _metadata(self, p, A_hat, null, S, S_eff, m, denom, dV):
        """
        Diagnostics. The update is dV_i = sum_j attn[i, j] Phi+_i V~_j - step_i V_i with attn = A_hat / denom
        and step = S_eff / denom; 1 - step is the share of the null slot (abstention). Without saturation S_eff = S.
        """
        D = self.config.head_dim
        H = p["mu"].shape[0]
        eigenvals = torch.stack([-p["mu"][:, None].expand(H, D), p["omega"]]).reshape(2, 1, H * D, 1)
        return {
            "attn_mat": A_hat / denom[:, :, None, :],  # weight of each (counter-rotated) value [B, N, N, H]
            "step": S_eff / denom,                   # S_eff_i / (gamma^-2 + S_eff_i), the gain [B, N, H]
            "null_share": null / denom,              # 1 - step
            "log_evidence": torch.log(S) + m,        # log S_i, the information from the context [B, N, H]
            "P_tot": torch.log(S) + m,               # (same, under RFA's key for the plotting code)
            "log_evidence_eff": torch.log(S_eff) + m,  # log S_eff_i, after saturation (= log_evidence without it)
            "posterior_var": torch.exp(-m) / denom,  # 1 / (gamma^-2 + S_eff_i)
            "dV": dV,                                # innovation in the eigenbasis [B, N, H, D, 2]
            "eigenvals": eigenvals,
        }
