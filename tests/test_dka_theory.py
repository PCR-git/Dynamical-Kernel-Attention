"""
Checks of the DKA model itself (Dynamical_Kernel_Attention.tex, Sections 3.1-3.5 and Appendices B, C, F),
using the literal reference implementation in dka/reference.py.

Two kinds of check:
  * the reference against independent computations (numerical ODE integration, Monte Carlo simulation
    of the SDE, autograd), so that it can be trusted;
  * the paper's closed forms and propositions against the reference.

    python tests/test_dka_theory.py
"""

import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dka.reference import (transition, dle_covariance, stationary_covariance, measurement_model,
                           dka_update, simulate_sde)

C128 = torch.complex128


def close(a, b, tol=1e-9):
    a, b = (x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float64) for x in (a, b))
    err = (a - b).abs().max().item()
    assert err <= tol, f"max abs error {err:.3e} > {tol:.1e}"


def random_system(d=3, seed=0):
    """A general (non-diagonal, non-normal) stable complex system: A = S - (B B^dagger + 0.3 I), S skew-Hermitian."""
    g = torch.Generator().manual_seed(seed)
    def cplx(*shape):
        return torch.complex(torch.randn(*shape, generator=g, dtype=torch.float64),
                             torch.randn(*shape, generator=g, dtype=torch.float64))
    M, B, Gq, Gr = cplx(d, d), cplx(d, d) * 0.5, cplx(d, d) * 0.7, cplx(d, d) * 0.5
    A = (M - M.conj().T) / 2 - (B @ B.conj().T + 0.3 * torch.eye(d, dtype=C128))
    return A, Gq @ Gq.conj().T, Gr @ Gr.conj().T


def isotropic_system(mu, omega, sigma_sq, eta_sq):
    """Paper Section 3.4: A = -mu I + i Omega, Q = sigma^2 I, R = eta^2 I."""
    d = len(omega)
    A = torch.diag(torch.complex(torch.full((d,), -mu, dtype=torch.float64), torch.as_tensor(omega, dtype=torch.float64)))
    I = torch.eye(d, dtype=C128)
    return A, sigma_sq * I, eta_sq * I


# ---------------------------------------------------------------------------------------------------
# The dynamical model (Section 3.1)
# ---------------------------------------------------------------------------------------------------

def test_dle_matches_numerical_integration():
    """Van Loan's V(dt) equals RK4 integration of dV/ds = A V + V A^dagger + Q (Eq. 3)."""
    A, Q, _ = random_system()
    f = lambda V: A @ V + V @ A.conj().T + Q
    V, h, s_end = torch.zeros_like(A), 1e-3, 1.7
    for _ in range(round(s_end / h)):
        k1 = f(V); k2 = f(V + h / 2 * k1); k3 = f(V + h / 2 * k2); k4 = f(V + h * k3)
        V = V + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    close(dle_covariance(A, Q, s_end), V, 1e-10)


def test_lyapunov_and_stationarity():
    """V_inf solves Eq. 5, is the long-lag limit of V, and Phi V_inf Phi^dagger = V_inf - V(dt) (Eq. 6)."""
    A, Q, _ = random_system()
    V_inf = stationary_covariance(A, Q)
    close(A @ V_inf + V_inf @ A.conj().T + Q, 0, 1e-12)
    close(dle_covariance(A, Q, 60.0), V_inf, 1e-10)
    for dt in (0.0, 0.3, 1.0, 4.0):
        Phi = transition(A, dt)
        close(Phi @ V_inf @ Phi.conj().T, V_inf - dle_covariance(A, Q, dt), 1e-12)


# ---------------------------------------------------------------------------------------------------
# The measurement model (Section 3.2, Appendix B.1)
# ---------------------------------------------------------------------------------------------------

def test_measurement_model_closed_form():
    """Conditioning gives H = I - V V_inf^-1 and Sigma = H V + Phi R Phi^dagger (Eq. 8), for a general system."""
    A, Q, R = random_system()
    V_inf = stationary_covariance(A, Q)
    I = torch.eye(3, dtype=C128)
    for dt in (0.0, 0.2, 1.0, 3.0):
        Phi, H, Sigma = measurement_model(A, Q, R, dt)
        V = dle_covariance(A, Q, dt)
        close(H, I - V @ torch.linalg.inv(V_inf), 1e-10)
        close(Sigma, H @ V + Phi @ R @ Phi.conj().T, 1e-10)
        close(Sigma, Sigma.conj().T, 1e-10)                       # Hermitian
        assert torch.linalg.eigvalsh(Sigma).min() > 0             # positive definite
    _, H0, S0 = measurement_model(A, Q, R, 0.0)
    close(H0, I, 1e-12)                                           # H_ii = I
    close(S0, R, 1e-12)                                           # Sigma_ii = R
    _, H_far, _ = measurement_model(A, Q, R, 80.0)
    close(H_far, 0, 1e-10)                                        # distant tokens carry no information


def test_measurement_model_monte_carlo():
    """Simulate the SDE: regressing x_hat_ij on x_i recovers H, and the residual covariance recovers Sigma."""
    g = torch.Generator().manual_seed(1)
    A = torch.diag(torch.tensor([-0.5 + 2.0j, -0.2 + 0.7j], dtype=C128))
    G = torch.diag(torch.tensor([1.0, 0.6], dtype=C128))
    R = torch.diag(torch.tensor([0.3, 0.5], dtype=C128))
    Q = G @ G.conj().T
    lags = [0.5, 2.0, 5.0]
    n = 40000
    xs = simulate_sde(A, G, torch.tensor([0.0] + lags, dtype=torch.float64), n, step=2e-3, burn_in=30.0, generator=g)
    x_j = xs[0]
    eps = torch.complex(torch.randn(n, 2, generator=g, dtype=torch.float64),
                        torch.randn(n, 2, generator=g, dtype=torch.float64)) * (R.diagonal().real / 2).sqrt()
    for k, dt in enumerate(lags):
        x_i = xs[k + 1]
        y = (x_j + eps) @ transition(A, dt).T                     # x_hat_ij = Phi (x_j + eps_j)
        C_yx = y.T @ x_i.conj() / n
        C_xx = x_i.T @ x_i.conj() / n
        H_mc = C_yx @ torch.linalg.inv(C_xx)
        res = y - x_i @ H_mc.T
        Sigma_mc = res.T @ res.conj() / n
        _, H, Sigma = measurement_model(A, Q, R, dt)
        close(H_mc, H, 0.02)
        rel = ((Sigma_mc - Sigma).abs() / Sigma.diagonal().real.max()).max().item()
        assert rel < 0.04, f"lag {dt}: Sigma relative error {rel:.3f}"


def test_isotropic_closed_form():
    """Eq. 14: h = E^2, Sigma = E^2 Sigma_n with Sigma_n = v_inf (1 - E^2) + eta^2, v_inf = sigma^2 / (2 mu)."""
    mu, sigma_sq, eta_sq = 0.3, 0.8, 0.4
    A, Q, R = isotropic_system(mu, [0.0, 1.3, -1.3, 0.2], sigma_sq, eta_sq)
    v_inf = sigma_sq / (2 * mu)
    I = torch.eye(4, dtype=C128)
    close(stationary_covariance(A, Q), v_inf * I, 1e-12)
    for dt in (0.0, 0.7, 2.5, 9.0):
        E = math.exp(-mu * dt)
        _, H, Sigma = measurement_model(A, Q, R, dt)
        close(H, E**2 * I, 1e-12)
        close(Sigma, E**2 * (v_inf * (1 - E**2) + eta_sq) * I, 1e-12)


def test_variance_inflation_closed_form():
    """
    Appendix F: with past-state prior N(0, v_inf + m), h = E^2 / c and Var = h Sigma_zeta, where
    zeta = m / (v_inf + m), c = (1 - zeta) + zeta E^2, Sigma_zeta = v_inf (1 - E^2) + c eta^2;
    the stable form log h = log sigmoid(theta - 2 mu dt) - log sigmoid(theta) with zeta = sigmoid(theta);
    and m -> infinity gives RFA's measurement, H = I and Sigma = V + Phi R Phi^dagger.
    """
    mu, sigma_sq, eta_sq = 0.25, 0.6, 0.5
    A, Q, R = isotropic_system(mu, [0.4, -0.4], sigma_sq, eta_sq)
    v_inf = sigma_sq / (2 * mu)
    I = torch.eye(2, dtype=C128)
    for m in (0.0, 0.3, 5.0):
        zeta = m / (v_inf + m)
        theta = math.log(zeta / (1 - zeta)) if 0 < zeta < 1 else None
        for dt in (0.0, 1.0, 4.0):
            E2 = math.exp(-2 * mu * dt)
            c = (1 - zeta) + zeta * E2
            h = E2 / c
            _, H, Sigma = measurement_model(A, Q, R, dt, inflation=m)
            close(H, h * I, 1e-12)
            close(Sigma, h * (v_inf * (1 - E2) + c * eta_sq) * I, 1e-12)
            if theta is not None:
                sig = lambda x: 1 / (1 + math.exp(-x))
                close(math.log(sig(theta - 2 * mu * dt)) - math.log(sig(theta)), math.log(h), 1e-12)
    dt = 1.5
    Phi, H, Sigma = measurement_model(A, Q, R, dt, inflation=1e9)
    close(H, I, 1e-8)
    close(Sigma, dle_covariance(A, Q, dt) + Phi @ R @ Phi.conj().T, 1e-6)


# ---------------------------------------------------------------------------------------------------
# The update (Section 3.3, Appendix B.3) and its isotropic form (Section 3.4, Appendix B.4)
# ---------------------------------------------------------------------------------------------------

def _random_tokens(N, d, seed):
    g = torch.Generator().manual_seed(seed)
    def cplx():
        return torch.complex(torch.randn(N, d, generator=g, dtype=torch.float64),
                             torch.randn(N, d, generator=g, dtype=torch.float64))
    t = torch.cumsum(0.3 + torch.rand(N, generator=g, dtype=torch.float64), 0)   # irregular times
    return cplx(), cplx(), cplx(), t


def test_update_minimizes_objective():
    """With the weights held fixed, V_i + dV_i minimizes the objective of Eq. 10 (autograd gradient is zero)."""
    A, Q, R = random_system(seed=2)
    U = torch.diag(torch.tensor([0.5, 1.0, 2.0], dtype=C128))
    Xq, Xk, Xv, t = _random_tokens(6, 3, seed=3)
    out = dka_update(Xq, Xk, Xv, t, A, Q, R, U, nu=5.0)
    U_inv = torch.linalg.inv(U)
    for i in (0, 3, 5):
        terms = []
        for j in range(i + 1):
            Phi, H, Sigma = measurement_model(A, Q, R, (t[i] - t[j]).item())
            terms.append((out["w"][i, j], Phi @ Xv[j], H, torch.linalg.inv(Sigma)))

        def objective(x):
            e = x - Xv[i]
            L = 0.5 * (e.conj() @ U_inv @ e).real
            for w, y, H, P in terms:
                r = y - H @ x
                L = L + 0.5 * w * (r.conj() @ P @ r).real
            return L

        x_bar = (Xv[i] + out["dV"][i]).detach()
        re, im = x_bar.real.clone().requires_grad_(), x_bar.imag.clone().requires_grad_()
        objective(torch.complex(re, im)).backward()
        close(re.grad, 0, 1e-10)
        close(im.grad, 0, 1e-10)


def test_isotropic_update_matches_paper():
    """
    Eqs. 15-18 (rotated residual, logits, innovation), written out with complex tensors, equal the
    literal update of Eq. 10 for an isotropic head; also the expanded residual norm (Eq. 16) and the
    stabilized forms (Appendix A): row-max subtraction, and the null slot included in the row max.
    """
    N, d = 7, 4
    mu, sigma_sq, eta_sq, gamma_sq, nu_s = 0.2, 0.9, 0.35, 1.7, 1.5
    omega = torch.tensor([0.9, -0.9, 0.25, -0.25], dtype=torch.float64)
    A, Q, R = isotropic_system(mu, omega, sigma_sq, eta_sq)
    Xq, Xk, Xv, t = _random_tokens(N, d, seed=4)
    ref = dka_update(Xq, Xk, Xv, t, A, Q, R, gamma_sq * torch.eye(d, dtype=C128), nu=nu_s * d)

    v_inf = sigma_sq / (2 * mu)
    dt = t[:, None] - t[None, :]
    causal = dt >= 0
    E = torch.exp(-mu * dt.clamp_min(0))
    Sigma_n = v_inf * (1 - E**2) + eta_sq
    Phi_minus = torch.exp(-1j * omega[None, :] * t[:, None])        # [N, d]
    Qr, Kr, Vr = Phi_minus * Xq, Phi_minus * Xk, Phi_minus * Xv
    R_qk = Kr[None, :, :] - E[:, :, None] * Qr[:, None, :]          # [i, j, d]
    R_sq = R_qk.abs().pow(2).sum(-1)
    expanded = (Xk.abs().pow(2).sum(-1)[None, :] + E**2 * Xq.abs().pow(2).sum(-1)[:, None]
                - 2 * E * (Qr.conj() @ Kr.T).real)
    close(R_sq, expanded, 1e-12)

    L = -torch.log(Sigma_n) - (nu_s + 1) * torch.log1p(R_sq / Sigma_n / (nu_s * d))
    L = L.masked_fill(~causal, float("-inf"))
    a = torch.exp(L)
    close(a * Sigma_n, ref["w"], 1e-12)                             # a_ij = w_ij / Sigma_n
    S = (a * E**2).sum(1)
    M = Phi_minus.conj() * ((a * E).to(C128) @ Vr)                  # counter-rotate the aggregate
    dV = (M - S[:, None] * Xv) / (1 / gamma_sq + S[:, None])
    close(dV, ref["dV"], 1e-12)
    close(S, torch.stack([ref["J"][i].diagonal().real.mean() for i in range(N)]), 1e-12)  # S_i is the information

    # Stabilized: subtract the row max m_i (Appendix A) ...
    m = L.max(1).values
    a_t = torch.exp(L - m[:, None])
    S_t = (a_t * E**2).sum(1)
    M_t = Phi_minus.conj() * ((a_t * E).to(C128) @ Vr)
    close((M_t - S_t[:, None] * Xv) / (torch.exp(-m)[:, None] / gamma_sq + S_t[:, None]), ref["dV"], 1e-12)
    # ... or the max over the keys and the null slot, whose logit is -log gamma^2
    m0 = torch.maximum(m, torch.full_like(m, -math.log(gamma_sq)))
    a_0 = torch.exp(L - m0[:, None])
    S_0 = (a_0 * E**2).sum(1)
    M_0 = Phi_minus.conj() * ((a_0 * E).to(C128) @ Vr)
    null = torch.exp(-math.log(gamma_sq) - m0)
    close((M_0 - S_0[:, None] * Xv) / (null[:, None] + S_0[:, None]), ref["dV"], 1e-12)


def test_inflated_update_matches_appendix_f():
    """Appendix F's algorithm (Sigma_zeta, residual c|K|^2 - 2E Re<Q,K> + h|Q|^2, S = sum a h) equals Eq. 10 with an inflated prior."""
    N, d = 6, 2
    mu, sigma_sq, eta_sq, gamma_sq, nu_s, m_infl = 0.3, 0.7, 0.45, 0.8, 2.0, 0.6
    omega = torch.tensor([0.6, -0.6], dtype=torch.float64)
    A, Q, R = isotropic_system(mu, omega, sigma_sq, eta_sq)
    Xq, Xk, Xv, t = _random_tokens(N, d, seed=5)
    ref = dka_update(Xq, Xk, Xv, t, A, Q, R, gamma_sq * torch.eye(d, dtype=C128), nu=nu_s * d, inflation=m_infl)

    v_inf = sigma_sq / (2 * mu)
    zeta = m_infl / (v_inf + m_infl)
    dt = t[:, None] - t[None, :]
    causal = dt >= 0
    E = torch.exp(-mu * dt.clamp_min(0))
    c = (1 - zeta) + zeta * E**2
    h = E**2 / c
    Sigma_z = v_inf * (1 - E**2) + c * eta_sq
    Phi_minus = torch.exp(-1j * omega[None, :] * t[:, None])
    Qr, Kr, Vr = Phi_minus * Xq, Phi_minus * Xk, Phi_minus * Xv
    R_sq = (c * Xk.abs().pow(2).sum(-1)[None, :] - 2 * E * (Qr.conj() @ Kr.T).real
            + h * Xq.abs().pow(2).sum(-1)[:, None])
    L = -torch.log(Sigma_z) - (nu_s + 1) * torch.log1p(R_sq / Sigma_z / (nu_s * d))
    a = torch.exp(L.masked_fill(~causal, float("-inf")))
    S = (a * h).sum(1)
    M = Phi_minus.conj() * ((a * E).to(C128) @ Vr)
    close((M - S[:, None] * Xv) / (1 / gamma_sq + S[:, None]), ref["dV"], 1e-12)


# ---------------------------------------------------------------------------------------------------
# Counting evidence (Propositions 1 and 2, Appendix C)
# ---------------------------------------------------------------------------------------------------

def _exact_information(N, mu, v_inf, eta_sq):
    """J_N: posterior precision of x_i from the N most recent tokens minus the prior precision, by joint conditioning."""
    k = torch.arange(N, dtype=torch.float64)                       # lags of the tokens
    states = v_inf * torch.exp(-mu * (k[:, None] - k[None, :]).abs())  # Cov(x_{i-a}, x_{i-b})
    decay = torch.exp(-mu * k)
    C_yy = decay[:, None] * decay[None, :] * (states + eta_sq * torch.eye(N, dtype=torch.float64))
    C_yx = decay * states[:, 0]                                    # Cov(x_hat_ij, x_i)
    post_var = v_inf - C_yx @ torch.linalg.solve(C_yy, C_yx)
    return (1 / post_var - 1 / v_inf).item()


def test_proposition_1_evidence():
    mu, v_inf, eta_sq = 0.05, 2.0, 0.5
    mu_bar = math.exp(-2 * mu)
    s = v_inf * (1 - mu_bar)
    L = 1 / (1 - mu_bar)
    info = lambda dt: mu_bar**dt / (v_inf * (1 - mu_bar**dt) + eta_sq)
    # DKA's per-token information is h^2 / Sigma of the reference measurement model
    A, Q, R = isotropic_system(mu, [0.0], 2 * mu * v_inf, eta_sq)
    for dt in (0, 1, 7):
        _, H, Sigma = measurement_model(A, Q, R, float(dt))
        close((H[0, 0].abs()**2 / Sigma[0, 0].real).item(), info(dt), 1e-12)
    D = 0.0
    for N in range(1, 301):
        D += info(N - 1)
        J = _exact_information(N, mu, v_inf, eta_sq)
        assert 1 / eta_sq - 1e-9 <= J <= 1 / eta_sq + mu_bar / s + 1e-9, (N, J)
        assert J - 1e-9 <= D <= L / eta_sq + 1e-9, (N, J, D)
        if N <= 2:
            close(D, J, 1e-9)
    rfa_total = sum(1 / (mu_bar**k * eta_sq + (1 - mu_bar**k) * v_inf) for k in range(300))
    assert rfa_total > 300 / v_inf                                  # RFA's total grows linearly


def test_proposition_2_pull():
    """RFA's estimate retains a fraction f_N of x_i with N f_N -> K in [L, (v_inf / eta^2) L]."""
    mu, v_inf, eta_sq = 0.05, 2.0, 0.5
    mu_bar = math.exp(-2 * mu)
    L = 1 / (1 - mu_bar)
    k = torch.arange(200000, dtype=torch.float64)
    Sigma_rfa = mu_bar**k * eta_sq + (1 - mu_bar**k) * v_inf
    K = v_inf * (mu_bar**k / Sigma_rfa).sum().item()
    assert L <= K <= v_inf / eta_sq * L, K
    N = 200000
    f_N = ((mu_bar**k / Sigma_rfa).sum() / (1 / Sigma_rfa).sum()).item()
    assert abs(N * f_N / K - 1) < 1e-3, N * f_N / K


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
