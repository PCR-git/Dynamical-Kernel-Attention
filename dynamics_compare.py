"""
Stage 5 check of DKA on the dynamics simulation of the RFA notebook (Main - Dynamics Training.ipynb): a noisy
2D linear spiral (eigenvalues -0.1 +- i, Euler steps of 0.01, measured every 0.1 time units with unit
measurement noise, x0 at radius 20 +- 3), embedded in d_e = 256 dimensions, and one attention layer (one head
of width 256, RoPE frequencies, learned rotations, fixed decay mu) trained to predict the next measurement.

Every training step draws fresh trajectories (a pool of steps x batch distinct trajectories per seed), so the
training loss measures generalization. Models (same data, seed, optimizer and schedule):
    rfa       RFA layer, output = W_o V_bar (the estimate; as in the notebook and the published runs)
    rfa_inn   x + RFA layer with the innovation output W_o (V_bar - V)
    dka       x + DKA layer (output W_o dV)
    dka_z     x + DKA layer with learned variance inflation (zeta = 0.5 at init)

References, computed on the test trajectories without training:
    kalman    the optimal one-step predictor of the true system
    oracle    the mechanisms with the true dynamics and noise, values = measurements, no learning: one DKA step
              from the current measurement, every past measurement transported to the current time and treated
              as an independent measurement of the current state (DKA's conditioned model, Eq. 8, with
              past-state prior N(0, V_inf + m I)). m = 0 is DKA, m -> infinity RFA's flat prior, m in between
              DKA with inflation zeta = m / (v_inf + m). Weights either Gaussian (the layer's rule as
              nu -> infinity) or 1 (pure precision weighting: the composite-likelihood estimate).

    python dynamics_compare.py                                   # writes runs/dynamics_compare/<date>_v2/
    python dynamics_compare.py --data stationary --test-length 400 --out-dir runs/dynamics_compare/stationary_extrap
                                    # stationary data (x0 from the stationary law), trained at length 101, tested
                                    # at 400: positions beyond 101 measure length extrapolation
    python dynamics_compare.py --steps 300 --seeds 0 --decays 0.01 --models dka   # quick test
    python dynamics_compare.py --check                           # oracle vs dka.reference
"""

import argparse
import datetime
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from dka.model import build_rfa_layer
from dka.training.presets import make_args
from dka.utils import complex_matmul, seed_everything

MODELS = ("rfa", "rfa_inn", "dka", "dka_z")
LABELS = {"rfa": "RFA (estimate)", "rfa_inn": "RFA (innovation)", "dka": "DKA", "dka_z": "DKA + inflation"}
COLORS = {"rfa": "#1f77b4", "rfa_inn": "#17becf", "dka": "#d62728", "dka_z": "#ff7f0e"}
SDE_KEYS = ("sigma_", "eta_", "gamma_")   # noise parameters: the notebook's slower optimizer group
F64 = torch.float64

# Simulation settings of the notebook
DT, N_SUB, N_MEAS = 0.01, 10, 102          # Euler step, Euler steps per measurement, measurements per trajectory
SIGMA_PROCESS, SIGMA_MEASURE = 0.5, 1.0
D_E = 256


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------

def spiral_matrix():
    """A = S D S^-1 with eigenvalues -0.1 +- i (notebook cell 12), real [2, 2] (float64, CPU)."""
    D1, S1, Si1 = (torch.zeros(2, 2, 2, dtype=F64) for _ in range(3))
    D1[0] = torch.tensor([[-0.1, 0.0], [0.0, -0.1]])
    D1[1] = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
    S1[0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    S1[1] = torch.tensor([[-1.0, 1.0], [0.0, 0.0]])
    Si1[0] = 0.5 * torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    Si1[1] = 0.5 * torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    return complex_matmul(S1, complex_matmul(D1, Si1))[0]


def discrete_model():
    """The simulated system between measurements: x_{k+1} = Phi x_k + w, w ~ N(0, Q); z = x + v, v ~ N(0, R)."""
    M = torch.eye(2, dtype=F64) + DT * spiral_matrix()
    Phi, Q = torch.eye(2, dtype=F64), torch.zeros(2, 2, dtype=F64)
    for _ in range(N_SUB):
        Phi, Q = M @ Phi, M @ Q @ M.T + SIGMA_PROCESS ** 2 * DT * torch.eye(2, dtype=F64)
    return Phi, Q, SIGMA_MEASURE ** 2 * torch.eye(2, dtype=F64)


def simulate(n, seed, device, stationary=False, keep_fine=0, n_meas=N_MEAS):
    """
    n trajectories: true states X and measurements Z at n_meas measurement times, [n, n_meas, 2]
    (float32 on device). x0 at radius 20 +- 3 (notebook), or from the stationary distribution.
    keep_fine > 0 also returns the fine Euler path of the first keep_fine trajectories.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = spiral_matrix().float().to(device)
    if stationary:
        Phi, Q, _ = discrete_model()
        L = torch.linalg.cholesky(stationary_cov(Phi, Q)).float()
        x = (torch.randn(n, 2, generator=g) @ L.T).to(device)
    else:
        theta = torch.rand(n, generator=g) * 2 * np.pi
        r = 20.0 + (torch.rand(n, generator=g) * 2 - 1) * 3.0
        x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], -1).to(device)
    steps = (n_meas - 1) * N_SUB
    noise = (torch.randn(steps, n, 2, generator=g) * SIGMA_PROCESS * DT ** 0.5).to(device)
    X, fine = [x], [x[:keep_fine]]
    for k in range(steps):
        x = x + DT * x @ A.T + noise[k]
        if keep_fine:
            fine.append(x[:keep_fine])
        if (k + 1) % N_SUB == 0:
            X.append(x)
    X = torch.stack(X, 1)
    Z = X + (torch.randn(X.shape, generator=g) * SIGMA_MEASURE).to(device)
    out = dict(X=X, Z=Z)
    if keep_fine:
        out["fine"] = torch.stack(fine, 1)
    return out


def embed(Z):
    """[..., 2] -> [..., d_e]: the first two coordinates (the notebook's rand_embed = 0)."""
    return torch.cat([Z, Z.new_zeros(*Z.shape[:-1], D_E - 2)], -1)


# ---------------------------------------------------------------------------------------------------
# References: Kalman predictor and composite-likelihood (DKA / RFA) oracles
# ---------------------------------------------------------------------------------------------------

def stationary_cov(Phi, Q):
    """V_inf = Phi V_inf Phi^T + Q."""
    d = Phi.shape[0]
    vec = torch.linalg.solve(torch.eye(d * d, dtype=F64) - torch.kron(Phi, Phi), Q.reshape(-1))
    return vec.reshape(d, d)


def kalman_predictions(Z, P0):
    """x_hat(t_{i+1} | z_0..z_i) for measurements Z [B, L, 2], x0 ~ N(0, P0)."""
    Phi, Q, R = discrete_model()
    Z = Z.double().cpu()
    x, P, preds = torch.zeros(Z.shape[0], 2, dtype=F64), P0.clone(), []
    for i in range(Z.shape[1]):
        K = P @ torch.linalg.inv(P + R)
        x = x + (Z[:, i] - x) @ K.T
        P = (torch.eye(2, dtype=F64) - K) @ P
        x, P = x @ Phi.T, Phi @ P @ Phi.T + Q
        preds.append(x)
    return torch.stack(preds, 1).float()


def measurement_models(Phis, Vs, V_inf, R, inflation):
    """
    Per lag k: (Phi_k, H_k, Sigma_k) of the transported measurement x_hat = Phi_k z_{i-k} as a measurement of
    x_i, by Gaussian conditioning with past-state prior N(0, V_inf + inflation I), as dka.reference.
    inflation = inf gives RFA's flat prior (H = I, Sigma = V_k + Phi_k R Phi_k^T).
    """
    out = []
    for Phi, V in zip(Phis, Vs):
        if inflation == float("inf"):
            out.append((Phi, torch.eye(2, dtype=F64), V + Phi @ R @ Phi.T))
            continue
        shared = Phi @ (V_inf + inflation * torch.eye(2, dtype=F64)) @ Phi.T
        H = shared @ torch.linalg.inv(shared + V)
        out.append((Phi, H, shared + Phi @ R @ Phi.T - H @ shared.T))
    return out


def composite_estimates(Z, models, gated):
    """
    One DKA step from the current measurement, with values = queries = keys = the measurements (C = I):

        x_i = z_i + J_i^-1 sum_{k <= i} w_ik H_k^T P_k (Phi_k z_{i-k} - H_k z_i),   J_i = sum_k w_ik H_k^T P_k H_k,

    with a flat proximal prior. gated=False: w = 1, the composite-likelihood estimate (exact in one step,
    since the objective is quadratic). gated=True: Gaussian weights w = exp(-d^2 / d), d^2 = r^T P r,
    r = Phi_k z_{i-k} - H_k z_i (the nu -> infinity limit of the Student-t weight; d = 2).
    Z: [B, L, 2]. Returns the filtered estimates [B, L, 2] (float64, CPU).
    """
    Z = Z.double().cpu()
    B, L, _ = Z.shape
    g = torch.zeros(B, L, 2, dtype=F64)
    J = torch.zeros(B, L, 2, 2, dtype=F64)
    for k, (Phi, H, Sigma) in enumerate(models[:L]):
        P = torch.linalg.inv(Sigma)
        r = Z[:, : L - k] @ Phi.T - Z[:, k:] @ H.T                       # [B, L-k, 2]
        w = torch.exp(-(r @ P * r).sum(-1) / 2) if gated else torch.ones(B, L - k, dtype=F64)
        g[:, k:] += w[..., None] * (r @ (H.T @ P).T)
        J[:, k:] += w[..., None, None] * (H.T @ P @ H)
    return Z + torch.linalg.solve(J, g.unsqueeze(-1)).squeeze(-1)


def oracle_predictions(Z, inflation, gated):
    """One-step predictions Phi x_i of the composite estimator of the simulated (discrete) system."""
    Phi, Q, R = discrete_model()
    Phis, Vs = [torch.eye(2, dtype=F64)], [torch.zeros(2, 2, dtype=F64)]
    for _ in range(Z.shape[1] - 1):
        Phis.append(Phi @ Phis[-1])
        Vs.append(Phi @ Vs[-1] @ Phi.T + Q)
    est = composite_estimates(Z, measurement_models(Phis, Vs, stationary_cov(Phi, Q), R, inflation), gated)
    return (est @ Phi.T).float()


def check_oracle():
    """
    composite_estimates(gated=True) equals z_i + dV_i of dka.reference.dka_update (nu -> infinity, flat
    proximal prior); gated=False equals the weighted least-squares solution over all transported measurements.
    """
    from dka.reference import dka_update, transition, dle_covariance, stationary_covariance
    torch.manual_seed(0)
    A = spiral_matrix().to(torch.complex128)
    Q, R = 0.3 * torch.eye(2, dtype=torch.complex128), torch.eye(2, dtype=torch.complex128)
    t = torch.arange(8, dtype=F64) * 0.1
    Z = torch.randn(3, 8, 2, dtype=F64)
    for m in (0.0, 2.0):
        Phis = [transition(A, k * 0.1).real for k in range(8)]
        Vs = [dle_covariance(A, Q, k * 0.1).real for k in range(8)]
        models = measurement_models(Phis, Vs, stationary_covariance(A, Q).real, R.real, m)
        mine = composite_estimates(Z, models, gated=True)
        plain = composite_estimates(Z, models, gated=False)
        for b in range(3):
            z = Z[b].to(torch.complex128)
            ref = dka_update(z, z, z, t, A, Q, R, 1e12 * torch.eye(2, dtype=torch.complex128), nu=1e8, inflation=m)
            err = (mine[b] - (Z[b] + ref["dV"].real)).abs().max().item()
            assert err < 1e-6, (m, b, err)
            for i in range(8):     # stacked least squares: rows P_k^(1/2) H_k x = P_k^(1/2) Phi_k z_{i-k}
                rows, rhs = [], []
                for k in range(i + 1):
                    Phi, H, Sigma = models[k]
                    Ls = torch.linalg.cholesky(torch.linalg.inv(Sigma)).T
                    rows.append(Ls @ H)
                    rhs.append(Ls @ Phi @ Z[b, i - k])
                x = torch.linalg.lstsq(torch.cat(rows), torch.cat(rhs)).solution
                assert (x - plain[b, i]).abs().max() < 1e-8, (m, b, i)
    print("oracle matches dka.reference.dka_update (Gaussian weights) and least squares (unit weights)")


# ---------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------

def model_args(name, decay):
    """One head of width 256 with learned rotations, RoPE frequencies and fixed mu = decay per step."""
    common = dict(d_e=D_E, n_heads=1, d_k_total=D_E, d_v_total=D_E, seq_len=N_MEAS - 1, use_SC_RoPE=False,
                  use_log_linear_decay=False, scale_decay_by_time_interval=False, damping=decay, zero_frac=0.0,
                  learn_rotations=True, use_ss_process_noise=True, target_ss_process_var=0.1,
                  target_measurement_var=1.0, min_exponent=-30)
    extra = {"rfa": dict(rfa_output="estimate"), "rfa_inn": dict(rfa_output="innovation"),
             "dka": dict(attention_type="dka"), "dka_z": dict(attention_type="dka", dka_inflation=0.5)}[name]
    return make_args("M1", **common, **extra)


class DynamicsModel(nn.Module):
    """One attention layer; with residual=True the prediction is x + layer(x) (the layer outputs a correction)."""

    def __init__(self, name, decay):
        super().__init__()
        self.layer = build_rfa_layer(model_args(name, decay), n_heads=1, input_dim=D_E, query_key_dim_total=D_E,
                                     value_dim_total=D_E)
        assert torch.allclose(self.layer.mu, torch.tensor([decay])), self.layer.mu   # mu = damping * omega_max, omega_max = 1
        self.residual = name != "rfa"

    def forward(self, x, return_metadata=False):
        out, info = self.layer(x, causal=True, return_metadata=return_metadata)
        return (x + out if self.residual else out), info


LOG_KEYS = ("sigma_tilde_sq_v", "eta_sq_v", "gamma_sq_v", "nu_over_d", "tau", "prior_weight", "zeta", "step_mean")


def train(name, decay, seed, steps, batch, lr, device, log_every, stationary=False):
    seed_everything(seed)
    pool = simulate(steps * batch, seed=10_000 + seed, device=device, stationary=stationary)["Z"]   # fresh at every step
    model = DynamicsModel(name, decay).to(device)
    sde = [p for n, p in model.named_parameters() if any(k in n for k in SDE_KEYS)]
    feat = [p for n, p in model.named_parameters() if not any(k in n for k in SDE_KEYS)]
    opt = torch.optim.Adam([{"params": feat, "lr": lr, "betas": (0.9, 0.999)},
                            {"params": sde, "lr": lr / 2, "betas": (0.0, 0.999), "eps": 1e-7}])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr / 2], total_steps=steps, pct_start=0.05,
                                                anneal_strategy="cos", div_factor=25.0, final_div_factor=1000.0)
    hist = {"loss": [], "it": [], "omega": []}
    t0 = time.time()
    for step in range(steps):
        Z = embed(pool[step * batch:(step + 1) * batch])
        pred, info = model(Z[:, :-1])
        loss = nn.functional.mse_loss(pred, Z[:, 1:])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if sde:
            nn.utils.clip_grad_norm_(sde, max_norm=1e-4)   # as single_epoch_rfa_dynamics
        opt.step()
        sched.step()
        hist["loss"].append(loss.item())
        if step % log_every == 0 or step == steps - 1:
            hist["it"].append(step)
            hist["omega"].append(model.layer.sde_params()["omega"][0, ::2].detach().cpu().numpy().copy())
            for key in LOG_KEYS:
                if key in info:
                    hist.setdefault(key, []).append(info[key].detach().float().cpu().numpy().copy())
    print(f"  {name:7s} mu={decay:<6} seed={seed}: {time.time() - t0:.0f}s, "
          f"final loss {np.mean(hist['loss'][-100:]):.5f}", flush=True)
    return model, hist


@torch.no_grad()
def evaluate(model, test):
    model.eval()
    pred, meta = model(embed(test["Z"][:, :-1]), return_metadata=True)
    pred2 = pred[..., :2]
    mse = ((pred2 - test["X"][:, 1:]) ** 2).sum(-1).mean(0)
    return pred2.cpu(), mse.cpu().numpy(), {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in meta.items()}


def plane_eigs(pred2, Z):
    """Eigenvalues of the least-squares linear map z_i -> prediction over positions i >= 50."""
    Zi = Z[:, 50:-1].reshape(-1, 2).cpu().double()
    Y = pred2[:, 50:].reshape(-1, 2).double()
    coef = torch.linalg.lstsq(torch.cat([Zi, torch.ones(len(Zi), 1, dtype=F64)], 1), Y).solution
    return np.linalg.eigvals(coef[:2].T.numpy())


# ---------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------

def mse_vs(pred, test):
    return ((pred - test["X"][:, 1:].cpu()) ** 2).sum(-1).mean(0).numpy()


TRAIN_LEN = N_MEAS - 1     # input positions seen in training (0..100)


def in_window(mse):
    """Mean error over positions 10..100 (inside the training length, after the start-up)."""
    return float(mse[10:TRAIN_LEN].mean())


def beyond(mse):
    """Mean error over positions beyond the training length (None if the test is not longer)."""
    return float(mse[TRAIN_LEN:].mean()) if len(mse) > TRAIN_LEN else None


def mark_train_length(ax, length):
    if length > TRAIN_LEN:
        ax.axvline(TRAIN_LEN - 0.5, color="gray", lw=1, ls="--")
        ax.text(TRAIN_LEN + 2, ax.get_ylim()[1], "beyond training length", va="top", fontsize=9, color="gray")


def plot_references(refs, path, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (mse, style) in refs.items():
        extra = f", beyond {beyond(mse):.3f}" if beyond(mse) is not None else ""
        ax.plot(mse, style, label=f"{label} (mean {in_window(mse):.3f}{extra})", lw=1.4)
    ax.set_yscale("log"); ax.set_xlabel("position i"); ax.set_ylabel("squared error vs true x(t_{i+1})")
    ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.3); ax.legend(frameon=False, fontsize=9)
    mark_train_length(ax, len(next(iter(refs.values()))[0]))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_decay(results, refs, test, out_dir, show_idx=1):
    """Figures for one decay value: errors (mean over seeds), and seed-0 diagnostics."""
    os.makedirs(out_dir, exist_ok=True)
    names = list(results)
    first = {n: results[n][0] for n in names}

    # 1. Training loss (fresh data: this is also the generalization loss), all seeds
    fig, ax = plt.subplots(figsize=(9, 5))
    for n in names:
        for s, r in enumerate(results[n]):
            loss = np.convolve(np.array(r["hist"]["loss"]), np.ones(50) / 50, mode="valid")
            ax.plot(loss, color=COLORS[n], alpha=0.8, lw=1, label=LABELS[n] if s == 0 else None)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE, next measurement (256 dims; fresh data)")
    ax.grid(True, alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "1_loss.png"), dpi=110); plt.close(fig)

    # 2. Prediction error by position, mean over seeds, with the references
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for n in names:
        m = np.mean([r["mse"] for r in results[n]], 0)
        spread = np.std([in_window(r["mse"]) for r in results[n]])
        extra = (f", beyond {beyond(m):.3f} +- {np.std([beyond(r['mse']) for r in results[n]]):.3f}"
                 if beyond(m) is not None else "")
        ax.plot(m, color=COLORS[n], label=f"{LABELS[n]} ({in_window(m):.3f} +- {spread:.3f}{extra})")
    for label, (mse, style) in refs.items():
        extra = f", beyond {beyond(mse):.3f}" if beyond(mse) is not None else ""
        ax.plot(mse, style, lw=1.2, label=f"{label} ({in_window(mse):.3f}{extra})")
    ax.set_yscale("log"); ax.set_xlabel("position i"); ax.set_ylabel("squared error vs true x(t_{i+1})")
    ax.set_title("test trajectories, mean over seeds (mean over positions 10-100 +- sd over seeds"
                 + (", and beyond the training length)" if beyond(m) is not None else ")"), fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(frameon=False, fontsize=8, ncol=2)
    mark_train_length(ax, len(m))
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "2_prediction_error.png"), dpi=110); plt.close(fig)

    # 3. Trajectories (seed 0)
    fine = test["fine"][show_idx].cpu().numpy()
    Zs = test["Z"][show_idx].cpu().numpy()
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), sharex=True, sharey=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        P = first[n]["pred"][show_idx].numpy()
        ax.plot(fine[:, 0], fine[:, 1], "k", lw=1, label="true")
        ax.plot(Zs[1:, 0], Zs[1:, 1], "b--", lw=0.6, alpha=0.6, label="measured")
        ax.plot(P[:, 0], P[:, 1], color=COLORS[n], lw=1.3, label="predicted")
        ax.set_title(LABELS[n], fontsize=11); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "3_trajectories.png"), dpi=110); plt.close(fig)

    # 4. Attention matrices (weights applied to the values), seed 0, power 0.25 for contrast
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.5))
    for ax, n in zip(np.atleast_1d(axes), names):
        meta = first[n]["meta"]
        W = meta["decayed_attn_mat"] if "decayed_attn_mat" in meta else meta["attn_mat"]
        ax.imshow(W[show_idx, :, :, 0].clamp_min(0).numpy() ** 0.25, cmap="magma")
        ax.set_title(LABELS[n] + ("\n(A E)" if "decayed_attn_mat" in meta else "\n(A_hat / (gamma^-2 + S))"),
                     fontsize=10)
        ax.set_xlabel("key j"); ax.set_ylabel("query i")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "4_attention.png"), dpi=110); plt.close(fig)

    # 5. Lag profile (mean over queries i >= 50 and test trajectories) and DKA step / null share by position
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for n in names:
        meta = first[n]["meta"]
        W = (meta["decayed_attn_mat"] if "decayed_attn_mat" in meta else meta["attn_mat"])[:, :, :, 0].mean(0)
        prof = [torch.diagonal(W, offset=-l)[50 - l:].mean().item() for l in range(50)]
        axes[0].plot(prof, color=COLORS[n], label=LABELS[n])
        if "step" in meta:
            axes[1].plot(meta["step"][:, :, 0].mean(0).numpy(), color=COLORS[n], label=LABELS[n] + ": step")
            axes[1].plot(meta["null_share"][:, :, 0].mean(0).numpy(), color=COLORS[n], ls=":",
                         label=LABELS[n] + ": null share")
    axes[0].set_yscale("log"); axes[0].set_xlabel("lag i - j"); axes[0].set_ylabel("mean weight (queries i >= 50)")
    axes[0].grid(True, alpha=0.3); axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("position i"); axes[1].set_ylabel("DKA step S/(gamma^-2+S), null share")
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3); axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "5_lag_profile_and_step.png"), dpi=110); plt.close(fig)

    # 6. Parameters over training, all seeds
    keys = [("sigma_tilde_sq_v", "v_inf (stationary var.)"), ("eta_sq_v", "eta^2"), ("nu_over_d", "nu / d"),
            ("gamma_sq_v", "gamma^2 (RFA floor; DKA 1/prior prec.)"), ("tau", "tau (RFA)"),
            ("prior_weight", "c_h (DKA prior, tokens)"), ("zeta", "zeta (inflation)"), ("step_mean", "mean step (DKA)")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for ax, (k, title) in zip(axes.flat, keys):
        for n in names:
            for s, r in enumerate(results[n]):
                h = r["hist"]
                if k in h:
                    ax.plot(h["it"], np.array(h[k])[:, 0], color=COLORS[n], lw=1, label=LABELS[n] if s == 0 else None)
        ax.set_title(title, fontsize=11); ax.set_yscale("log" if k not in ("zeta", "step_mean") else "linear")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "6_parameters.png"), dpi=110); plt.close(fig)

    # 7. Eigenvalues: learned frequencies (seed 0) and the fitted plane map of every model and reference
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for n in names:
        axes[0].plot(np.sort(np.abs(first[n]["hist"]["omega"][-1]))[::-1], color=COLORS[n], label=LABELS[n])
    axes[0].plot(np.sort(np.abs(first[names[0]]["hist"]["omega"][0]))[::-1], color="gray", ls=":", label="init (RoPE)")
    axes[0].axhline(0.1, color="k", ls="--", lw=1, label="true omega dt = 0.1")
    axes[0].set_yscale("log"); axes[0].set_xlabel("frequency index (sorted)"); axes[0].set_ylabel("|omega| per step")
    axes[0].grid(True, alpha=0.3); axes[0].legend(frameon=False, fontsize=9)
    for n in names:
        ev = np.concatenate([r["plane_eigs"] for r in results[n]])
        axes[1].scatter(ev.real, ev.imag, color=COLORS[n], s=40, alpha=0.8, label=LABELS[n])
    for label, (ev, marker) in test["ref_eigs"].items():
        axes[1].scatter(ev.real, ev.imag, marker=marker, color="k", s=80, label=label)
    axes[1].set_xlabel("Re"); axes[1].set_ylabel("Im"); axes[1].grid(True, alpha=0.3)
    axes[1].set_title("eigenvalues of the least-squares map z_i -> prediction (i >= 50)", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "7_eigenvalues.png"), dpi=110); plt.close(fig)


def plot_summary(table, refs, path, ylabel="squared error vs true next state (mean over positions 10-100)"):
    """Mean error of every model and decay, +- sd over seeds, with the references."""
    decays = sorted({d for (_, d) in table})
    names = [n for n in MODELS if any((n, d) in table for d in decays)]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(decays))
    for k, d in enumerate(decays):
        xs = [i + (k - (len(decays) - 1) / 2) * width for i in range(len(names))]
        means = [np.mean(table[(n, d)]) if (n, d) in table else np.nan for n in names]
        sds = [np.std(table[(n, d)]) if (n, d) in table else 0 for n in names]
        ax.bar(xs, means, width * 0.9, yerr=sds, capsize=3, label=f"mu = {d} per step", alpha=0.85)
    for label, (value, style) in refs.items():
        ax.axhline(value, ls=style, color="k", lw=1, label=f"{label}: {value:.3f}")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([LABELS[n] for n in names])
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    p.add_argument("--decays", nargs="*", type=float, default=[0.005, 0.01],
                   help="fixed decay mu per step (the notebook used 0.005; the true system decays at ~0.01)")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--test-samples", type=int, default=1024)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--check", action="store_true", help="check the oracle against dka.reference and exit")
    p.add_argument("--references-only", action="store_true", help="compute the untrained references and exit")
    p.add_argument("--data", choices=("transient", "stationary"), default="transient",
                   help="training and test data: the notebook's spirals from radius 20, or x0 from the stationary law")
    p.add_argument("--test-length", type=int, default=TRAIN_LEN,
                   help="input positions at test time (training uses 101); longer tests length extrapolation")
    p.add_argument("--out-dir", default=os.path.join("runs", "dynamics_compare", f"{datetime.date.today()}_v2"))
    cli = p.parse_args()
    if cli.check:
        check_oracle()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cli.out_dir, exist_ok=True)
    Phi, Q, R = discrete_model()
    V_inf = stationary_cov(Phi, Q)
    summary = {"stationary_std_per_coordinate": V_inf.diagonal().sqrt().tolist()}

    # --- References on the notebook's (transient) data and on stationary data ---
    zetas = [0.0, 0.5, 0.9, 0.99, 0.999, 1.0]
    v_bar = V_inf.diagonal().mean().item()
    infl = lambda z: float("inf") if z == 1.0 else v_bar * z / (1 - z)
    ref_tables = {}
    for kind in ("transient", "stationary"):
        test = simulate(cli.test_samples, seed=999, device=device, stationary=(kind == "stationary"), keep_fine=4,
                        n_meas=cli.test_length + 1)
        P0 = V_inf if kind == "stationary" else (20.0 ** 2 + 3.0 ** 2 / 3) / 2 * torch.eye(2, dtype=F64)
        Zin = test["Z"][:, :-1]
        mses = {"Kalman (optimal)": mse_vs(kalman_predictions(Zin, P0), test),
                "last measurement": mse_vs(Zin.cpu().float(), test)}
        sweep = {}
        for gated in (False, True):
            for z in zetas:
                mse = mse_vs(oracle_predictions(Zin, infl(z), gated), test)
                sweep[(gated, z)] = in_window(mse)
                if z in (0.0, 1.0):
                    mses[f"{'DKA' if z == 0 else 'RFA'} oracle, {'Gaussian weights' if gated else 'w = 1'}"] = mse
        ref_tables[kind] = {k: in_window(v) for k, v in mses.items()}
        if cli.test_length > TRAIN_LEN:
            ref_tables[kind]["beyond_training_length"] = {k: beyond(v) for k, v in mses.items()}
        ref_tables[kind]["zeta_sweep"] = {f"{'gauss' if g else 'w1'}_zeta={z:g}": v for (g, z), v in sweep.items()}
        styles = {"Kalman (optimal)": "k--", "last measurement": "k:", "DKA oracle, w = 1": "r-",
                  "DKA oracle, Gaussian weights": "r--", "RFA oracle, w = 1": "b-", "RFA oracle, Gaussian weights": "b--"}
        plot_references({k: (v, styles[k]) for k, v in mses.items()}, os.path.join(cli.out_dir, f"0_references_{kind}.png"),
                        f"untrained references, {kind} data (means over positions 10-100)")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for gated, style in ((False, "o-"), (True, "s--")):
            ax.plot(range(len(zetas)), [sweep[(gated, z)] for z in zetas], style,
                    label="Gaussian weights" if gated else "w = 1")
        ax.axhline(ref_tables[kind]["Kalman (optimal)"], color="k", ls="--", lw=1, label="Kalman")
        ax.set_xticks(range(len(zetas)))
        ax.set_xticklabels([f"{z:g}" for z in zetas])
        ax.set_xlabel("inflation zeta (0 = DKA, 1 = RFA)")
        ax.set_ylabel("mean squared error, positions 10-100")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend(frameon=False)
        ax.set_title(f"oracle estimators vs inflation, {kind} data", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(cli.out_dir, f"0_oracle_vs_zeta_{kind}.png"), dpi=110); plt.close(fig)
        if kind == cli.data:
            refs = {"Kalman (optimal)": (mses["Kalman (optimal)"], "k--"),
                    "DKA oracle": (mses["DKA oracle, Gaussian weights"], "k-."),
                    "RFA oracle": (mses["RFA oracle, Gaussian weights"], "k:")}
            test["ref_eigs"] = {"true one-step map": (np.linalg.eigvals(Phi.numpy()), "x"),
                                "Kalman (same fit)": (plane_eigs(kalman_predictions(Zin, P0), test["Z"]), "+")}
            test_main = test
    summary["references"] = ref_tables
    print(json.dumps(ref_tables, indent=1), flush=True)
    if cli.references_only:
        with open(os.path.join(cli.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        return

    # --- Trained models ---
    table, table_beyond = {}, {}
    summary["data"], summary["test_length"] = cli.data, cli.test_length
    for decay in cli.decays:
        results = {}
        for name in cli.models:
            results[name] = []
            for seed in cli.seeds:
                model, hist = train(name, decay, seed, cli.steps, cli.batch, cli.lr, device, cli.log_every,
                                    stationary=(cli.data == "stationary"))
                pred2, mse, meta = evaluate(model, test_main)
                if seed != cli.seeds[0]:
                    meta = None        # keep diagnostics of the first seed only
                results[name].append(dict(hist=hist, pred=pred2, mse=mse, meta=meta,
                                          plane_eigs=plane_eigs(pred2, test_main["Z"])))
                table[(name, decay)] = table.get((name, decay), []) + [in_window(mse)]
                if beyond(mse) is not None:
                    table_beyond[(name, decay)] = table_beyond.get((name, decay), []) + [beyond(mse)]
                del model
        plot_decay(results, refs, test_main, os.path.join(cli.out_dir, f"mu_{decay}"))
        summary[f"mu_{decay}"] = {
            n: dict(mse_mean=float(np.mean(table[(n, decay)])), mse_sd=float(np.std(table[(n, decay)])),
                    mse_per_seed=table[(n, decay)],
                    **({"beyond_mean": float(np.mean(table_beyond[(n, decay)])),
                        "beyond_sd": float(np.std(table_beyond[(n, decay)])),
                        "beyond_per_seed": table_beyond[(n, decay)]} if (n, decay) in table_beyond else {}),
                    final_params_seed0={k: np.array(results[n][0]["hist"][k][-1]).round(4).tolist()
                                        for k in LOG_KEYS if k in results[n][0]["hist"]})
            for n in results}
        with open(os.path.join(cli.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)

    plot_summary(table, {k: (in_window(v[0]), st) for k, (v, st) in
                         zip(refs, zip(refs.values(), ("--", "-.", ":")))},
                 os.path.join(cli.out_dir, "summary.png"))
    if table_beyond:
        plot_summary(table_beyond, {k: (beyond(v[0]), st) for k, (v, st) in
                                    zip(refs, zip(refs.values(), ("--", "-.", ":")))},
                     os.path.join(cli.out_dir, "summary_beyond_training_length.png"),
                     ylabel=f"squared error vs true next state (mean over positions {TRAIN_LEN}-{cli.test_length - 1})")
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("mu_")}, indent=1))
    print(f"Figures in {cli.out_dir}")


if __name__ == "__main__":
    main()
