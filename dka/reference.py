"""
Slow, literal reference implementation of the DKA model (Dynamical_Kernel_Attention.tex, Section 3).

Everything here works with explicit complex matrices and derives each quantity from first principles
(matrix exponentials, Gaussian conditioning, per-pair loops) rather than from the closed forms of the
paper, so that the closed forms and the fast attention layer can be checked against it. Nothing here is
fast; use complex128 tensors and small sizes.

Notation follows the paper:
    dx = A x dt + G dW,   Q = G G^dagger,   z_j = C (x_j + eps_j),  eps_j ~ CN(0, R)
    Phi(dt) = e^{A dt},   V(dt) solves the DLE with V(0) = 0,   V_inf solves the Lyapunov equation.
"""

import torch


def _H(M):
    """Conjugate transpose of the last two dims."""
    return M.transpose(-1, -2).conj()


def transition(A, dt):
    """Phi(dt) = e^{A dt}."""
    return torch.linalg.matrix_exp(A * dt)


def dle_covariance(A, Q, dt, max_step=0.25):
    """
    V(dt) = int_0^dt e^{A s} Q e^{A^dagger s} ds, the solution of dV/ds = A V + V A^dagger + Q, V(0) = 0,
    by Van Loan's method: exp([[-A, Q], [0, A^dagger]] h) = [[*, F12], [0, F22]], V(h) = F22^dagger F12.
    Van Loan's block contains e^{-A h}, which grows with h, so it is applied over steps h <= max_step and
    the steps are composed exactly: V(s + h) = V(h) + Phi(h) V(s) Phi(h)^dagger.
    """
    d = A.shape[-1]
    n = max(1, int(-(-abs(dt) // max_step)))
    h = dt / n
    M = torch.zeros(2 * d, 2 * d, dtype=A.dtype)
    M[:d, :d] = -A
    M[:d, d:] = Q
    M[d:, d:] = _H(A)
    F = torch.linalg.matrix_exp(M * h)
    V_h = _H(F[d:, d:]) @ F[:d, d:]
    Phi_h = transition(A, h)
    V = V_h
    for _ in range(n - 1):
        V = V_h + Phi_h @ V @ _H(Phi_h)
    return V


def stationary_covariance(A, Q):
    """V_inf solving A V + V A^dagger + Q = 0 (Kronecker form, column-major vec)."""
    d = A.shape[-1]
    I = torch.eye(d, dtype=A.dtype)
    K = torch.kron(I, A) + torch.kron(A.conj(), I)
    vecV = torch.linalg.solve(K, -Q.T.reshape(-1))
    return vecV.reshape(d, d).T


def measurement_model(A, Q, R, dt, inflation=0.0):
    """
    The transported estimate x_hat_ij = Phi (x_j + eps_j) as a linear measurement of the current state,
    x_hat_ij | x_i ~ N(H x_i, Sigma), by Gaussian conditioning on the joint distribution of (x_hat_ij, x_i).

    The past state has prior N(0, V_inf + inflation * I): inflation = 0 is DKA's stationary prior (Eq. 8),
    inflation -> infinity the flat prior of RFA (Appendix F).

    Returns (Phi, H, Sigma).
    """
    Phi = transition(A, dt)
    V = dle_covariance(A, Q, dt)
    P0 = stationary_covariance(A, Q) + inflation * torch.eye(A.shape[-1], dtype=A.dtype)
    shared = Phi @ P0 @ _H(Phi)                  # Cov(x_hat_ij, x_i): the transported past state
    var_x = shared + V                           # Var(x_i)
    var_y = shared + Phi @ R @ _H(Phi)           # Var(x_hat_ij)
    H = shared @ torch.linalg.inv(var_x)
    Sigma = var_y - H @ _H(shared)
    return Phi, H, Sigma


def student_t_weight(d2, nu, d):
    """Tempered IRLS weight of Eq. 9: (1 + d^2 / nu)^(-kappa), kappa = (nu + d) / d."""
    return (1 + d2 / nu) ** (-(nu + d) / d)


def dka_update(Xq, Xk, Xv, t, A, Q, R, U, nu, inflation=0.0):
    """
    One DKA layer for a single head, literally per Eqs. 9-10 with p = 1 (absorbed into U).

    Xq, Xk, Xv: complex [N, d], the current estimates read through the query, key and value projections.
    t: [N] increasing times. U: proximal prior covariance [d, d]. nu: Student-t degrees of freedom.

    For each query i and key j <= i, with (Phi, H, Sigma) the measurement model at lag t_i - t_j:
        weights:  r_ij = Phi Xk_j - H Xq_i,   d_ij^2 = r^dagger Sigma^{-1} r,   w_ij = (1 + d^2/nu)^-kappa
        update:   dV_i = (U^{-1} + J_i)^{-1} sum_j w_ij H^dagger P (Phi Xv_j - H Xv_i),
                  J_i  = sum_j w_ij H^dagger P H.

    Returns dict with dV [N, d], weights w [N, N] (0 above the diagonal), information J [N, d, d].
    """
    N, d = Xq.shape
    U_inv = torch.linalg.inv(U)
    dV = torch.zeros_like(Xv)
    w = torch.zeros(N, N, dtype=torch.float64)
    J = torch.zeros(N, d, d, dtype=Xq.dtype)
    for i in range(N):
        g = torch.zeros(d, dtype=Xq.dtype)
        for j in range(i + 1):
            Phi, H, Sigma = measurement_model(A, Q, R, (t[i] - t[j]).item(), inflation)
            P = torch.linalg.inv(Sigma)
            r = Phi @ Xk[j] - H @ Xq[i]
            d2 = (r.conj() @ P @ r).real
            w[i, j] = student_t_weight(d2, nu, d)
            HP = _H(H) @ P
            g = g + w[i, j] * HP @ (Phi @ Xv[j] - H @ Xv[i])
            J[i] = J[i] + w[i, j] * HP @ H
        dV[i] = torch.linalg.solve(U_inv + J[i], g)
    return dict(dV=dV, w=w, J=J)


def simulate_sde(A, G, times, n_paths, step=1e-3, burn_in=None, generator=None):
    """
    Euler-Maruyama simulation of dx = A x dt + G dW with circular complex noise (E|dW|^2 = dt per
    component), started at x = 0 a time `burn_in` before times[0] so that the paths are (nearly)
    stationary. `times` must lie on the grid of `step` relative to times[0].

    Returns x at the requested times: complex [len(times), n_paths, d].
    """
    d = A.shape[-1]
    if burn_in is None:
        burn_in = 10.0 / (-torch.linalg.eigvals(A).real).min().item()
    t0 = times[0].item()
    idx = torch.round((times - t0) / step).long()
    n_steps = int(round(burn_in / step)) + idx[-1].item()
    record = {int(k) + int(round(burn_in / step)): n for n, k in enumerate(idx.tolist())}
    x = torch.zeros(n_paths, d, dtype=A.dtype)
    out = torch.zeros(len(times), n_paths, d, dtype=A.dtype)
    step_A = (A * step).T
    G_T = G.T
    scale = (step / 2) ** 0.5
    for k in range(n_steps + 1):
        if k in record:
            out[record[k]] = x
        noise = torch.complex(torch.randn(n_paths, d, generator=generator, dtype=torch.float64),
                              torch.randn(n_paths, d, generator=generator, dtype=torch.float64)) * scale
        x = x + x @ step_A + noise @ G_T
    return out
