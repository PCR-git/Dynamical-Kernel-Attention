"""
Checks of the vectorized DKA kernels in dka/kernels.py against the literal reference (dka/reference.py),
their limits, and their numerical behavior in float32.

    python tests/test_dka_kernels.py
"""

import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dka.kernels import (rotate, rotation_angles, forcing_variance, inflation_terms, dka_covariance,
                         dka_residual_norm_sq, calibrated_evidence)
from dka.reference import measurement_model, dle_covariance

C128, F64 = torch.complex128, torch.float64

# Per-head parameters: decay mu, forcing sigma^2 (per unit time), measurement noise eta^2, inflation theta
MU = torch.tensor([0.02, 0.15, 0.6, 1.5], dtype=F64)
SIGMA_SQ = torch.tensor([0.3, 1.1, 0.5, 2.0], dtype=F64)
ETA_SQ = torch.tensor([0.4, 0.2, 1.3, 0.7], dtype=F64)
THETA = torch.tensor([-1.5, 0.0, 0.8, 2.5], dtype=F64)
LAGS = torch.tensor([0.0, 0.5, 1.0, 3.0, 7.5, 20.0], dtype=F64)[:, None]


def close(a, b, tol=1e-12):
    a, b = (x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float64) for x in (a, b))
    err = (a - b).abs().max().item()
    assert err <= tol, f"max abs error {err:.3e} > {tol:.1e}"


def reference_head(h, omega=(0.7, -0.7)):
    """Isotropic system (A, Q, R) of head h, and its stationary variance v_inf."""
    A = torch.diag(torch.tensor([complex(-MU[h].item(), w) for w in omega], dtype=C128))
    I = torch.eye(len(omega), dtype=C128)
    return A, SIGMA_SQ[h].item() * I, ETA_SQ[h].item() * I, (SIGMA_SQ[h] / (2 * MU[h])).item()


def test_forcing_variance_matches_dle():
    V = forcing_variance(MU, SIGMA_SQ, LAGS)
    for h in range(len(MU)):
        A, Q, _, _ = reference_head(h)
        for k, dt in enumerate(LAGS[:, 0].tolist()):
            close(V[k, h], dle_covariance(A, Q, dt)[0, 0].real)


def test_measurement_model_matches_reference():
    """h and h * Sigma_n are the reference's H and Sigma, with (theta) and without (None) inflation."""
    for theta in (None, THETA):
        log_h, c = inflation_terms(theta, MU, LAGS)
        Sigma_n = dka_covariance(MU, SIGMA_SQ, ETA_SQ, LAGS, c)
        for h in range(len(MU)):
            A, Q, R, v_inf = reference_head(h)
            m = 0.0 if theta is None else v_inf * math.exp(theta[h].item())   # zeta = m / (v_inf + m) = sigmoid(theta)
            for k, dt in enumerate(LAGS[:, 0].tolist()):
                _, H, Sigma = measurement_model(A, Q, R, dt, inflation=m)
                close(torch.exp(log_h[k, h]), H[0, 0].real)
                close(torch.exp(log_h[k, h]) * Sigma_n[k, h], Sigma[0, 0].real)


def test_inflation_limits():
    """theta -> -inf is DKA (no inflation); theta -> +inf is RFA's flat prior: h = 1, Sigma_n = V + eta^2 E^2."""
    lo, c_lo = inflation_terms(torch.full_like(MU, -60.0), MU, LAGS)
    ref, _ = inflation_terms(None, MU, LAGS)
    close(lo, ref, 1e-12)
    close(c_lo, 1.0, 1e-12)
    hi, c_hi = inflation_terms(torch.full_like(MU, 200.0), MU, LAGS)   # theta / (2 mu) far beyond every lag
    E_sq = torch.exp(-2 * MU * LAGS)
    close(hi, 0.0, 1e-12)
    close(dka_covariance(MU, SIGMA_SQ, ETA_SQ, LAGS, c_hi), forcing_variance(MU, SIGMA_SQ, LAGS) + ETA_SQ * E_sq, 1e-12)


def test_zero_decay_limit():
    """mu = 0 gives the Brownian sigma^2 dt + eta^2 and h = 1, continuously in mu."""
    zero = torch.zeros_like(MU)
    close(forcing_variance(zero, SIGMA_SQ, LAGS), SIGMA_SQ * LAGS, 0)
    tiny = torch.full_like(MU, 1e-9)
    close(forcing_variance(tiny, SIGMA_SQ, LAGS), SIGMA_SQ * LAGS, 1e-6)
    for theta in (None, THETA):
        log_h, c = inflation_terms(theta, zero, LAGS)
        close(log_h, 0.0, 0)
        close(torch.as_tensor(c), 1.0, 1e-15)


def test_residual_matches_reference_distance():
    """dka_residual_norm_sq / Sigma_n equals the reference's d^2 = r^dagger Sigma^-1 r with r = Phi K_j - H Q_i."""
    g = torch.Generator().manual_seed(0)
    B, N, D = 2, 5, 4
    n_heads = len(MU)
    omega = torch.rand(n_heads, D // 2, generator=g, dtype=F64) * 2
    omega_full = torch.stack([omega, -omega], -1).flatten(-2)             # conjugate pairs, as in the layer
    t = torch.cumsum(0.3 + torch.rand(B, N, generator=g, dtype=F64), 1)   # irregular times
    Q = torch.randn(B, N, n_heads, D, 2, generator=g, dtype=F64)
    K = torch.randn(B, N, n_heads, D, 2, generator=g, dtype=F64)
    cos, sin = rotation_angles(omega_full, t)
    Q_rot, K_rot = rotate(Q, cos, -sin), rotate(K, cos, -sin)
    lags = (t[:, :, None] - t[:, None, :]).abs()[..., None]
    E = torch.exp(-MU * lags)
    for theta in (None, THETA):
        log_h, c = inflation_terms(theta, MU, lags)
        d2 = dka_residual_norm_sq(Q_rot, K_rot, E, torch.exp(log_h), c) / dka_covariance(MU, SIGMA_SQ, ETA_SQ, lags, c)
        for b in range(B):
            for h in range(n_heads):
                A = torch.diag(torch.complex(-MU[h].expand(D), omega_full[h]))
                I = torch.eye(D, dtype=C128)
                v_inf = (SIGMA_SQ[h] / (2 * MU[h])).item()
                m = 0.0 if theta is None else v_inf * math.exp(theta[h].item())
                q = torch.complex(Q[b, :, h, :, 0], Q[b, :, h, :, 1])
                k = torch.complex(K[b, :, h, :, 0], K[b, :, h, :, 1])
                for i in range(N):
                    for j in range(i + 1):
                        Phi, H, Sigma = measurement_model(A, SIGMA_SQ[h].item() * I, ETA_SQ[h].item() * I,
                                                          (t[b, i] - t[b, j]).item(), inflation=m)
                        r = Phi @ k[j] - H @ q[i]
                        close(d2[b, i, j, h], (r.conj() @ torch.linalg.solve(Sigma, r)).real, 1e-10)


def test_calibrated_evidence():
    """S_bar is the sum of the reference's per-token information |H|^2 / Sigma over lags 0..L-1; for mu L >> 1 it
    approaches Appendix G's integral formula; and it is finite and continuous at mu = 0."""
    L = 60
    for theta in (None, THETA):
        S_bar = calibrated_evidence(MU, SIGMA_SQ, ETA_SQ, L, theta)
        for h in range(len(MU)):
            A, Q, R, v_inf = reference_head(h)
            m = 0.0 if theta is None else v_inf * math.exp(theta[h].item())
            total = sum((H[0, 0].abs() ** 2 / Sigma[0, 0].real).item()
                        for _, H, Sigma in (measurement_model(A, Q, R, float(dt), inflation=m) for dt in range(L)))
            close(S_bar[h], total, 1e-10)
    # Appendix G's integral approximation holds for slow heads; it underestimates fast ones (mu = 0.5: by 28%),
    # whose per-token information falls off within a token, which is why the exact sum is used.
    v_inf, eta_sq = 3.0, torch.tensor([0.4], dtype=F64)
    for mu, tol in ((0.01, 0.002), (0.5, None)):
        approx = math.log(1 + v_inf / eta_sq.item()) / (2 * mu * v_inf) + 1 / (2 * eta_sq.item())
        exact = calibrated_evidence(torch.tensor([mu], dtype=F64), torch.tensor([2 * mu * v_inf], dtype=F64),
                                    eta_sq, int(40 / mu)).item()
        assert (abs(approx / exact - 1) < tol) if tol else (approx / exact < 0.75), (mu, exact, approx)
    close(calibrated_evidence(torch.zeros(1, dtype=F64), torch.ones(1, dtype=F64), eta_sq, 512),
          calibrated_evidence(torch.full((1,), 1e-9, dtype=F64), torch.ones(1, dtype=F64), eta_sq, 512), 1e-5)


def test_float32_extremes():
    """No inf or NaN in values or gradients at extreme decay, noise, inflation and lag, in float32."""
    mu = torch.tensor([0.0, 1e-6, 1e-2, 5.0, 0.3, 0.3], requires_grad=True)
    sigma_sq = torch.tensor([1e-6, 10.0, 0.5, 2.0, 1e-8, 50.0], requires_grad=True)
    eta_sq = torch.tensor([1e-5, 1.0, 1e-5, 0.3, 1e-5, 1e-5], requires_grad=True)
    theta = torch.tensor([-40.0, 40.0, 0.0, 15.0, -15.0, 3.0], requires_grad=True)
    lags = torch.tensor([0.0, 1.0, 37.0, 4096.0, 1e5])[:, None]
    for th in (None, theta):
        log_h, c = inflation_terms(th, mu, lags)
        Sigma_n = dka_covariance(mu, sigma_sq, eta_sq, lags, c)
        S_bar = calibrated_evidence(mu, sigma_sq, eta_sq, 512, th)
        h = torch.exp(log_h)
        for x in (log_h, h, Sigma_n, S_bar):
            assert torch.isfinite(x).all() or (x is log_h and torch.isfinite(h).all())
        assert (h >= 0).all() and (h <= 1 + 1e-6).all() and (Sigma_n > 0).all()
        (h.sum() + Sigma_n.log().sum() + S_bar.log().sum()).backward()
        for p in (mu, sigma_sq, eta_sq, theta):
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), p.grad
                p.grad = None


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
