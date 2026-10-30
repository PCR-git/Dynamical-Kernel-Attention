"""
Stage 5 check of DKA on the dynamics simulation of the RFA notebook (Main - Dynamics Training.ipynb): a noisy
2D linear spiral (eigenvalues -0.1 +- i, measured every 0.1 time units with unit measurement noise), embedded
in d_e = 256 dimensions, and one attention layer trained to predict the next measurement.

Models (same data, seed, optimizer and schedule):
    rfa       RFA layer, output = W_o V_bar (the estimate; as in the notebook and the published runs)
    rfa_inn   x + RFA layer with the innovation output W_o (V_bar - V)
    dka       x + DKA layer (output W_o dV)
    dka_z     x + DKA layer with learned variance inflation (zeta = 0.5 at init)
The optimal (Kalman) one-step predictor of the true system is the reference for the prediction error.

    python dynamics_compare.py                       # writes runs/dynamics_compare/<date>/
    python dynamics_compare.py --steps 500 --models dka
"""

import argparse
import datetime
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from dka import MultiheadIsotropicDKA
from dka.model import build_rfa_layer
from dka.training.presets import make_args
from dka.utils import complex_matmul, seed_everything
from dynamics import construct_random_mapping, create_train_loader_dynamics

MODELS = ("rfa", "rfa_inn", "dka", "dka_z")
LABELS = {"rfa": "RFA (estimate)", "rfa_inn": "RFA (innovation)", "dka": "DKA", "dka_z": "DKA + inflation",
          "kalman": "Kalman (optimal)", "last": "last measurement"}
COLORS = {"rfa": "#1f77b4", "rfa_inn": "#17becf", "dka": "#d62728", "dka_z": "#ff7f0e", "kalman": "black",
          "last": "gray"}
SDE_KEYS = ("sigma_", "eta_", "gamma_")   # names of the noise parameters (slower group, as in the notebook)


# ---------------------------------------------------------------------------------------------------
# Data: the notebook's system and simulation settings
# ---------------------------------------------------------------------------------------------------

def spiral_system(device):
    """A = S D S^-1 with eigenvalues -0.1 +- i (notebook cell 12), as a real [1, 2, 2] matrix."""
    D1, S1, Si1 = (torch.zeros(2, 2, 2, device=device) for _ in range(3))
    D1[0] = torch.tensor([[-0.1, 0.0], [0.0, -0.1]])
    D1[1] = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
    S1[0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    S1[1] = torch.tensor([[-1.0, 1.0], [0.0, 0.0]])
    Si1[0] = 0.5 * torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    Si1[1] = 0.5 * torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    A = complex_matmul(S1, complex_matmul(D1, Si1))[0].unsqueeze(0)
    return A, S1, Si1


def sim_args(device, num_samp):
    a = argparse.Namespace(device=device, rand_embed=0, m=2, d_e=256, tf=10.0, dt=0.01, n=10, k_step=1,
                           t_equal=True, batch_size=32, num_samp=num_samp)
    a.N_t = int(a.tf / a.dt)
    a.N_t_tot = a.N_t + a.n * a.k_step
    a.seq_len = int(a.N_t / a.n) + 1
    a.total_L = a.seq_len + a.k_step
    return a


SIGMA_PROCESS, SIGMA_MEASURE = 0.5, 1.0


def make_data(device, num_samp, seed):
    seed_everything(seed)
    A, S1, Si1 = spiral_system(device)
    a = sim_args(device, num_samp)
    t_vector = torch.arange(a.N_t_tot + 1, device=device) * a.dt
    Pu, Pd, R1, R1i = construct_random_mapping(S1, Si1, a)
    loader, dataset, X_true, X_meas, _ = create_train_loader_dynamics(A, S1, Si1, Pu, Pd, R1, R1i, t_vector,
                                                                     SIGMA_PROCESS, SIGMA_MEASURE, a, a.k_step)
    return dict(A=A, args=a, loader=loader, dataset=dataset, X_true=X_true, X_meas=X_meas, Pd=Pd, R1i=R1i)


def to_2d(Y, data):
    """[B, L, d_e] -> [B, L, 2] (inverse of the embedding)."""
    return torch.matmul(data["Pd"] @ data["R1i"], Y.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------------------------------
# Kalman one-step predictor of the true (Euler-discretized) system
# ---------------------------------------------------------------------------------------------------

def kalman_predictions(A, a, Z):
    """
    x_hat(t_{i+1} | z_0..z_i) for measurements Z [B, L, 2] of the simulated system: n Euler steps of
    x <- (I + A dt) x + sigma_p sqrt(dt) w between measurements, z = x + sigma_m v. x0 has mean 0 and
    covariance r^2/2 I with E[r^2] = 20^2 + 3^2/3.
    """
    M = torch.eye(2, dtype=torch.float64) + a.dt * A[0].double().cpu()
    Phi, Q = torch.eye(2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64)
    for _ in range(a.n):
        Phi, Q = M @ Phi, M @ Q @ M.T + SIGMA_PROCESS ** 2 * a.dt * torch.eye(2, dtype=torch.float64)
    R = SIGMA_MEASURE ** 2 * torch.eye(2, dtype=torch.float64)
    Z = Z.double().cpu()
    B, L, _ = Z.shape
    x = torch.zeros(B, 2, dtype=torch.float64)
    P = (20.0 ** 2 + 3.0 ** 2 / 3) / 2 * torch.eye(2, dtype=torch.float64)
    preds = []
    for i in range(L):
        K = P @ torch.linalg.inv(P + R)
        x = x + (Z[:, i] - x) @ K.T
        P = (torch.eye(2, dtype=torch.float64) - K) @ P
        x, P = x @ Phi.T, Phi @ P @ Phi.T + Q
        preds.append(x)
    return torch.stack(preds, 1).float()


# ---------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------

def model_args(name):
    """One head of width 256 with learned rotations, RoPE frequencies and mu = 0.005 (notebook settings)."""
    common = dict(d_e=256, n_heads=1, d_k_total=256, d_v_total=256, seq_len=101, use_SC_RoPE=False,
                  use_log_linear_decay=False, scale_decay_by_time_interval=False, damping=0.005, zero_frac=0.0,
                  learn_rotations=True, use_ss_process_noise=True, target_ss_process_var=0.1,
                  target_measurement_var=1.0, min_exponent=-30)
    extra = {"rfa": dict(rfa_output="estimate"), "rfa_inn": dict(rfa_output="innovation"),
             "dka": dict(attention_type="dka"), "dka_z": dict(attention_type="dka", dka_inflation=0.5)}[name]
    return make_args("M1", **common, **extra)


class DynamicsModel(nn.Module):
    """One attention layer; with residual=True the prediction is x + layer(x) (the layer outputs a correction)."""

    def __init__(self, name):
        super().__init__()
        self.layer = build_rfa_layer(model_args(name), n_heads=1, input_dim=256, query_key_dim_total=256,
                                     value_dim_total=256)
        self.residual = name != "rfa"

    def forward(self, x, return_metadata=False):
        out, info = self.layer(x, causal=True, return_metadata=return_metadata)
        return (x + out if self.residual else out), info


def train(name, data, steps, lr, seed, log_every):
    seed_everything(seed)
    model = DynamicsModel(name).to(data["args"].device)
    sde = [p for n, p in model.named_parameters() if any(k in n for k in SDE_KEYS)]
    feat = [p for n, p in model.named_parameters() if not any(k in n for k in SDE_KEYS)]
    opt = torch.optim.Adam([{"params": feat, "lr": lr, "betas": (0.9, 0.999)},
                            {"params": sde, "lr": lr / 2, "betas": (0.0, 0.999), "eps": 1e-7}])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr / 2], total_steps=steps, pct_start=0.05,
                                                anneal_strategy="cos", div_factor=25.0, final_div_factor=1000.0)
    hist = {"loss": [], "it": []}
    step = 0
    t0 = time.time()
    while step < steps:
        for x, y, *_ in data["loader"]:
            pred, info = model(x)
            loss = nn.functional.mse_loss(pred, y)
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
                for key in ("mu_v", "omega", "sigma_tilde_sq_v", "eta_sq_v", "gamma_sq_v", "nu_over_d", "tau",
                            "prior_weight", "zeta", "step_mean"):
                    if key == "omega":
                        value = model.layer.sde_params()["omega"][0, ::2]
                    elif key in info:
                        value = info[key]
                    else:
                        continue
                    hist.setdefault(key, []).append(value.detach().float().cpu().numpy().copy())
            step += 1
            if step >= steps:
                break
    print(f"  {name}: {steps} steps in {time.time() - t0:.0f}s, final loss {np.mean(hist['loss'][-50:]):.4g}")
    return model, hist


# ---------------------------------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, test):
    """Prediction (2D) for the test trajectories, per-position MSE against the true next state, metadata."""
    model.eval()
    x, _, X_true, _, _ = test["dataset"][:]
    pred, meta = model(x, return_metadata=True)
    a = test["args"]
    target_true = X_true[:, a.n::a.n][:, : x.shape[1]]          # true state at t_{i+1}
    pred2 = to_2d(pred, test)
    mse = ((pred2 - target_true) ** 2).sum(-1).mean(0)            # [L], squared error in the plane
    return pred2.cpu(), mse.cpu().numpy(), {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in meta.items()}


@torch.no_grad()
def effective_eigenvalues(model):
    """Eigenvalues of the learned per-step linear map in the data plane, log(eig)/1 step (cf. -0.01 +- 0.1i)."""
    layer = model.layer
    d = 256
    W_v = torch.stack((layer.W_v.weight[:d], layer.W_v.weight[d:]))      # [2, d, d_e]
    W_o = torch.stack((layer.W_o.weight[:, :d], -layer.W_o.weight[:, d:]))
    p = layer.sde_params()
    lam = torch.stack((-p["mu"][:, None].expand_as(p["omega"]).flatten(), p["omega"].flatten()))
    return lam.cpu().numpy(), W_o, W_v


def plot_all(results, kal_mse, last_mse, test, out_dir, show_idx=1):
    os.makedirs(out_dir, exist_ok=True)
    names = list(results)
    a = test["args"]

    # 1. Training loss
    fig, ax = plt.subplots(figsize=(9, 5))
    for n in names:
        loss = np.array(results[n]["hist"]["loss"])
        k = 20
        ax.plot(np.convolve(loss, np.ones(k) / k, mode="valid"), color=COLORS[n], label=LABELS[n])
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("training MSE (next measurement, 256 dims)")
    ax.grid(True, alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "1_loss.png"), dpi=110); plt.close(fig)

    # 2. Prediction error against the true next state, by position, on held-out trajectories
    fig, ax = plt.subplots(figsize=(9, 5))
    for n in names:
        ax.plot(results[n]["mse"], color=COLORS[n], label=f"{LABELS[n]} (mean {results[n]['mse'][10:].mean():.2f})")
    ax.plot(kal_mse, color="black", ls="--", label=f"Kalman, optimal (mean {kal_mse[10:].mean():.2f})")
    ax.plot(last_mse, color="gray", ls=":", label=f"last measurement (mean {last_mse[10:].mean():.2f})")
    ax.set_yscale("log"); ax.set_xlabel("position i"); ax.set_ylabel("squared error vs true x(t_{i+1})")
    ax.set_title("held-out trajectories (means over positions 10-100)", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "2_prediction_error.png"), dpi=110); plt.close(fig)

    # 3. Trajectories
    x, _, X_true, X_meas, _ = test["dataset"][show_idx]
    Xt, Xm = X_true.cpu().numpy(), X_meas.cpu().numpy()
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), sharex=True, sharey=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        P = results[n]["pred"][show_idx].numpy()
        ax.plot(Xt[:, 0], Xt[:, 1], "k", lw=1, label="true")
        ax.plot(Xm[1:, 0], Xm[1:, 1], "b--", lw=0.6, alpha=0.6, label="measured")
        ax.plot(P[:, 0], P[:, 1], color=COLORS[n], lw=1.3, label="predicted")
        ax.set_title(LABELS[n], fontsize=11); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "3_trajectories.png"), dpi=110); plt.close(fig)

    # 4. Attention matrices (weights applied to the values), power 0.25 for contrast
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.5))
    for ax, n in zip(np.atleast_1d(axes), names):
        meta = results[n]["meta"]
        W = meta["decayed_attn_mat"] if "decayed_attn_mat" in meta else meta["attn_mat"]
        ax.imshow(W[show_idx, :, :, 0].clamp_min(0).numpy() ** 0.25, cmap="magma")
        ax.set_title(LABELS[n] + ("\n(A E)" if "decayed_attn_mat" in meta else "\n(A_hat / (gamma^-2 + S))"),
                     fontsize=10)
        ax.set_xlabel("key j"); ax.set_ylabel("query i")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "4_attention.png"), dpi=110); plt.close(fig)

    # 5. Attention profile by lag (mean over queries i >= 50) and DKA step / evidence by position
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for n in names:
        meta = results[n]["meta"]
        W = (meta["decayed_attn_mat"] if "decayed_attn_mat" in meta else meta["attn_mat"])[:, :, :, 0]
        prof = np.array([np.mean([W[:, i, i - l].mean().item() for i in range(50, W.shape[1])]) for l in range(50)])
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

    # 6. Parameters over training
    keys = [("sigma_tilde_sq_v", "v_inf (stationary var.)"), ("eta_sq_v", "eta^2"), ("nu_over_d", "nu / d"),
            ("gamma_sq_v", "gamma^2 (RFA floor; DKA 1/prior prec.)"), ("tau", "tau (RFA)"),
            ("prior_weight", "c_h (DKA prior, tokens)"), ("zeta", "zeta (inflation)"), ("step_mean", "mean step (DKA)")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for ax, (k, title) in zip(axes.flat, keys):
        for n in names:
            h = results[n]["hist"]
            if k in h:
                ax.plot(h["it"], np.array(h[k])[:, 0], color=COLORS[n], label=LABELS[n])
        ax.set_title(title, fontsize=11); ax.set_yscale("log" if k not in ("zeta", "step_mean") else "linear")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "6_parameters.png"), dpi=110); plt.close(fig)

    # 7. Eigenvalues: learned rotation frequencies (init vs final), the true system's 0.1 rad/step,
    #    and the eigenvalues of the learned per-step map in the data plane
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for n in names:
        om = np.array(results[n]["hist"]["omega"])
        axes[0].plot(np.sort(np.abs(om[-1]))[::-1], color=COLORS[n], label=LABELS[n] + " (final)")
    axes[0].plot(np.sort(np.abs(np.array(results[names[0]]["hist"]["omega"])[0]))[::-1], color="gray", ls=":",
                 label="init (RoPE)")
    axes[0].axhline(0.1, color="k", ls="--", lw=1, label="true omega dt = 0.1")
    axes[0].set_yscale("log"); axes[0].set_xlabel("frequency index (sorted)"); axes[0].set_ylabel("|omega| per step")
    axes[0].grid(True, alpha=0.3); axes[0].legend(frameon=False, fontsize=9)
    for n in names:
        ev = results[n]["plane_eigs"]
        axes[1].scatter(ev.real, ev.imag, color=COLORS[n], s=60, label=LABELS[n])
    true = np.linalg.eigvals(results[names[0]]["true_map"])
    axes[1].scatter(true.real, true.imag, marker="x", color="k", s=90, label="true one-step map")
    axes[1].set_xlabel("Re"); axes[1].set_ylabel("Im"); axes[1].set_title("eigenvalues of the learned map z_i -> prediction\n"
                                                                          "(least squares on test data)", fontsize=10)
    axes[1].grid(True, alpha=0.3); axes[1].legend(frameon=False, fontsize=9); axes[1].set_aspect("equal", "datalim")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "7_eigenvalues.png"), dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--train-samples", type=int, default=32)
    p.add_argument("--test-samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out-dir", default=os.path.join("runs", "dynamics_compare", str(datetime.date.today())))
    cli = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = make_data(device, cli.train_samples, cli.seed)
    test = make_data(device, cli.test_samples, cli.seed + 1)
    a = test["args"]
    x, _, X_true, X_meas, _ = test["dataset"][:]
    target_true = X_true[:, a.n::a.n][:, : x.shape[1]].cpu()
    kal = kalman_predictions(test["A"], a, X_meas[:, : x.shape[1]])
    kal_mse = ((kal - target_true) ** 2).sum(-1).mean(0).numpy()
    last_mse = ((X_meas[:, : x.shape[1]].cpu() - target_true) ** 2).sum(-1).mean(0).numpy()

    # True one-step map of the measurement plane
    M = torch.eye(2, dtype=torch.float64) + a.dt * test["A"][0].double().cpu()
    true_map = torch.linalg.matrix_power(M, a.n).numpy()

    results = {}
    for name in cli.models:
        model, hist = train(name, train_data, cli.steps, cli.lr, cli.seed, cli.log_every)
        pred2, mse, meta = evaluate(model, test)
        # Effective linear map from the current measurement to the prediction, by least squares over positions
        # i >= 50 (the attention also averages the past, so this is a summary, not the layer's exact form)
        Z = X_meas[:, 50: x.shape[1]].reshape(-1, 2).cpu().double()
        Y = pred2[:, 50:].reshape(-1, 2).double()
        coef = torch.linalg.lstsq(torch.cat([Z, torch.ones(len(Z), 1, dtype=torch.float64)], 1), Y).solution
        results[name] = dict(hist=hist, pred=pred2, mse=mse, meta=meta,
                             plane_eigs=np.linalg.eigvals(coef[:2].T.numpy()), true_map=true_map)

    plot_all(results, kal_mse, last_mse, test, cli.out_dir)
    summary = {n: dict(mse_mean_10_100=float(results[n]["mse"][10:].mean()),
                       final_train_loss=float(np.mean(results[n]["hist"]["loss"][-50:])),
                       **{k: np.array(results[n]["hist"][k][-1]).round(4).tolist()
                          for k in ("sigma_tilde_sq_v", "eta_sq_v", "nu_over_d", "gamma_sq_v", "tau", "prior_weight",
                                    "zeta", "step_mean") if k in results[n]["hist"]})
               for n in results}
    summary["kalman_mse_mean_10_100"] = float(kal_mse[10:].mean())
    summary["last_measurement_mse_mean_10_100"] = float(last_mse[10:].mean())
    with open(os.path.join(cli.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"Figures in {cli.out_dir}")


if __name__ == "__main__":
    main()
