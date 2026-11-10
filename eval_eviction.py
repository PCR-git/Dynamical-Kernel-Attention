"""KV-cache eviction sweep: perplexity against cache size at length L (default 4096) for trained checkpoints.

Policies (every head, every layer; lag 0 is always kept):
  uniform W      keep lags < W                                     (sliding window)
  stream W       keep the first 4 tokens + lags < W - 4            (StreamingLLM)
  h2o W          keep lags < W/2 + the W/2 earlier keys with the largest attention accumulated over the queries so
                 far (H2O; scores from the unevicted attention, so this is an optimistic version of H2O)
  thresh eps     head h keeps lags < T_h = ln(1/eps) / s_h, where s_h is the decay rate of a key's weight in the
                 output: mu_h for DKA (the numerator weights a E), 2 mu_h for B5 (bias slope), the ALiBi slope for B2.
                 It ignores how the content logits vary with lag (e.g. DKA's logit floor, Sigma growing from eta^2).
  cal eps        head h keeps lags < T_h, the smallest lag at which the head's measured weight beyond T_h, averaged
                 over queries, is below eps (calibrated on 8 validation blocks). Same rule for every model; for
                 DKA the weights are the numerator shares softmax(L - mu dt), for RFA softmax(tau L - mu dt) (E
                 multiplies the values after the softmax), for the baselines the softmax.
  bound eps      DKA only, content-independent: the robust weight is at most its value in agreement, a <= 1/Sigma,
                 and the denominator is at least the null slot gamma^-2 = c / eta^2, so dropping every key at lag
                 >= T changes the head's update by at most (memory-window proposition)
                     B(T) = (eta^2 / c) sum_{lag >= T} (E + (2 + K) E^2) / Sigma     (in units of the value norm),
                     K    = (eta^2 / c) sum_{lag >= 0} E / Sigma
                 (E / Sigma from the numerator, E^2 / Sigma from S V, and K E^2 / Sigma from the denominator change
                 times the update, whose norm is at most 1 + K since sum a E can exceed gamma^-2 + S).
                 T_h = min T with B(T) <= eps. Uses only the learned mu, sigma^2, eta^2, c of each head.
  boundcap / calcap eps   the same windows capped at the training length (512), so no head keeps more than a
                 sliding window of the training length.

The cache size reported is measured: kept keys per head, averaged over heads, layers and queries in 2048-4096.
dNLL p99 / p99.9: percentiles of the per-token change in NLL from full context, over tokens at positions >= 512.

usage: python eval_eviction.py <out.txt> <name=run_dir> [...] [--blocks N] [--length L] [--cpu]
       e.g. python eval_eviction.py eviction.txt D3=runs/wikitext103/D3__<date> B2=runs/wikitext103/B2__<date>
Window sizes for the uniform, stream and h2o policies: SWEEP_W="32,64,128,256,512" (environment variable).
SWEEP_BOUND_EPS / SWEEP_BOUNDCAP_EPS override the bound / boundcap eps lists; SWEEP_POLICIES="bound,cal" runs only
those policies (and full context).
Eviction is applied by masking the logits of the attention modules (DKA, RFA and baselines) in place, so the
models run unchanged; this measures quality, not memory or speed.
"""
import glob
import math
import os
import re
import sys
import torch
import torch.nn.functional as F
import dka.attention as ra
import dka.dka_attention as da
from dka.kernels import temper_content
from dka.attention import MultiheadIsotropicRFA
import dka.model.baselines.attention as bl
from dka.dka_attention import MultiheadIsotropicDKA
from dka.model.baselines import MultiHeadAttentionLayer
from dka.training.lm_data import load_tokenizer
from dka.training.presets import build_language_model
from eval_extrapolation import test_blocks

WINDOWS = tuple(int(x) for x in os.environ.get("SWEEP_W", "32,64,128,256,512").split(","))
EPS = (0.5, 0.2, 0.05, 1e-2, 1e-3)
N_SINKS = 4
CUT = {"policy": None, "T": {}, "W": None, "layer": None, "kept": [], "prof": {}}
CAL_EPS = (0.1, 0.03, 0.01, 3e-3, 1e-3)
BOUND_EPS = tuple(float(x) for x in os.environ.get("SWEEP_BOUND_EPS", "0.3,0.1,0.03,0.01,3e-3").split(","))
BOUNDCAP_EPS = tuple(float(x) for x in os.environ.get("SWEEP_BOUNDCAP_EPS", "0.1,0.03,0.01").split(","))
POLICIES = os.environ.get("SWEEP_POLICIES")    # e.g. "bound,boundcap,cal,calcap"; None = all (full context always)
CAP = 512
GRID = sorted({int(round(1.25 ** k)) for k in range(60)})


def record_profile(log_w, head_dim):
    """Accumulate, per head, the mean weight on lags >= T for T in GRID, over queries in the second half."""
    w = log_w.exp()
    w = w if head_dim == 1 else w.permute(0, 3, 1, 2)                # [B, H, Nq, Nk]
    N = w.shape[-1]
    cw = w.float().cumsum(dim=-1)
    i = torch.arange(N // 2, N, device=w.device)
    grid = [T for T in GRID if T < N]
    out = []
    cw = cw[:, :, i, :]
    for T in grid:
        idx = (i - T).clamp(min=0)
        tail = cw.gather(-1, idx[None, None, :, None].expand(w.shape[0], w.shape[1], -1, 1))[..., 0]
        out.append(tail.mean(dim=(0, 2)))                            # [H]
    prof = torch.stack(out).cpu()                                    # [len(grid), H]
    old = CUT["prof"].get(CUT["layer"])
    CUT["prof"][CUT["layer"]] = prof if old is None else old + prof


def calibrated_T(eps, N, n_cal):
    grid = torch.tensor([T for T in GRID if T < N])
    T = {}
    for m, prof in CUT["prof"].items():
        ok = (prof / n_cal) <= eps                                   # [len(grid), H]
        first = torch.where(ok.any(0), ok.float().argmax(0), torch.full((ok.shape[1],), len(grid)))
        T[m] = torch.cat([grid, torch.tensor([N])])[first]
    return T


def slopes(module):
    """[H] log-weight slope per head, or None (RoPE)."""
    if isinstance(module, (MultiheadIsotropicDKA, MultiheadIsotropicRFA)):
        with torch.no_grad():
            return module.sde_params()["mu"].flatten().float()
    if getattr(module, "bias_slopes", None) is not None:
        return module.bias_slopes.flatten().float()
    if getattr(module, "alibi_slopes", None) is not None:
        return module.alibi_slopes.flatten().float()
    return None


def bound_T(module, eps, N):
    """Content-independent window of a DKA head (see 'bound eps' above). [H] long."""
    with torch.no_grad():
        p = module.sde_params()
        p = {k: v.detach().cpu() for k, v in p.items() if torch.is_tensor(v)}
        mu, eta_sq = p["mu"].double(), p["eta_sq"].double()
        v_inf = p["sigma_sq"].double() / (2 * mu)
        c_over = torch.exp(p["log_prior_precision"].double()) * eta_sq        # c_h (gamma^-2 = c / eta^2)
        lag = torch.arange(N, dtype=torch.float64)[:, None]
        E = torch.exp(-mu * lag)
        Sigma = v_inf * (1 - E * E) + eta_sq
        K = (E / Sigma).sum(0) * eta_sq / c_over                               # [H], sum over the context
        term = (E + (2 + K) * E * E) / Sigma                                   # [N, H]
        tail = term.flip(0).cumsum(0).flip(0) * eta_sq / c_over                # B(T) for T = 0..N-1
        ok = tail <= eps
        return torch.where(ok.any(0), ok.double().argmax(0), torch.full_like(mu, N)).clamp(min=1).long()


def keep_mask(X, key_dim, head_dim, log_w=None):
    """Boolean keep mask broadcastable to X, for the current policy and layer. X: logits with queries on dim
    key_dim - 1. log_w: log attention weights (same shape as X), needed for h2o."""
    N = X.shape[key_dim]
    i = torch.arange(N, device=X.device)
    lag = i[:, None] - i[None, :]                                    # [Nq, Nk]
    H = X.shape[head_dim]
    pol, W = CUT["policy"], CUT["W"]

    def per_head(m):                                                 # m [H, Nq, Nk] -> layout of X
        return m.permute(1, 2, 0)[None] if head_dim == 3 else m[None]

    if pol == "uniform":
        keep = per_head((lag < W).expand(H, N, N))
    elif pol == "stream":
        keep = per_head(((lag < W - N_SINKS) | (i[None, :] < N_SINKS)).expand(H, N, N))
    elif pol in ("thresh", "cal", "bound", "calcap", "boundcap"):
        T = CUT["T"][CUT["layer"]].to(X.device)                       # [H]
        keep = per_head(lag[None] < T[:, None, None])
    elif pol == "h2o":
        r = W // 2
        w = log_w.exp()                                              # [B, ...], zero above the diagonal
        w = w if head_dim == 1 else w.permute(0, 3, 1, 2)            # [B, H, Nq, Nk]
        acc = w.cumsum(dim=2)                                        # attention received over queries <= i
        cand = (lag >= r) & (lag >= 0)
        acc = acc.masked_fill(~cand, -1.0)
        idx = acc.topk(min(W - r, N), dim=-1).indices
        heavy = torch.zeros_like(acc, dtype=torch.bool).scatter_(-1, idx, True) & cand
        keep = heavy | ((lag < r) & (lag >= 0))
        keep = keep if head_dim == 1 else keep.permute(0, 2, 3, 1)
    causal = (lag >= 0)[None, None] if head_dim == 1 else (lag >= 0)[None, :, :, None]
    q0 = N // 2
    k = (keep & causal).sum(dim=key_dim).float()                     # kept keys per query
    k = k[:, :, q0:] if head_dim == 1 else k[:, q0:, :]
    CUT["kept"].append(float(k.mean()))
    return keep


def wrap_logits(fn):                     # DKA logits [B, Nq, Nk, H]
    def g(*a, **kw):
        L = fn(*a, **kw)
        if CUT["policy"] is None or L.dim() < 4:     # RFA's null logit (prior="tokens") is not a key matrix
            return L
        log_w = None
        if CUT["policy"] in ("h2o", "profile"):   # numerator shares: softmax(L - mu dt) over the keys
            N = L.shape[1]
            i = torch.arange(N, device=L.device)
            lag = (i[:, None] - i[None, :]).clamp(min=0).float()
            s = slopes(CUT["layer"]).to(L.device)
            if isinstance(CUT["layer"], MultiheadIsotropicRFA):
                with torch.no_grad():
                    L_eff = CUT["layer"].sde_params()["tau"].to(L.device) * L
            elif getattr(CUT["layer"], "log_tau", None) is not None:           # DKA, tempered content term
                L_eff = temper_content(L, a[1], torch.exp(CUT["layer"].log_tau.detach()).to(L.device))
            else:
                L_eff = L
            z = (L_eff - lag[None, :, :, None] * s[None, None, None, :]).masked_fill(
                (i[:, None] < i[None, :])[None, :, :, None], float("-inf"))
            log_w = z.log_softmax(dim=2)
        if CUT["policy"] == "profile":
            record_profile(log_w, 3)
            return L
        return L.masked_fill(~keep_mask(L, 2, 3, log_w), float("-inf"))
    return g


da.robust_logits = wrap_logits(da.robust_logits)
ra.robust_logits = wrap_logits(ra.robust_logits)


class _F:                                # baseline scores [B, H, Nq, Nk] go through F.softmax
    def __getattr__(self, name):
        return getattr(F, name)

    @staticmethod
    def softmax(x, dim=-1, **kw):
        if CUT["policy"] == "profile":
            record_profile(x.log_softmax(dim=-1), 1)
        elif CUT["policy"] is not None:
            log_w = x.log_softmax(dim=-1) if CUT["policy"] == "h2o" else None
            x = x.masked_fill(~keep_mask(x, 3, 1, log_w), float("-inf"))
        return F.softmax(x, dim=dim, **kw)


bl.F = _F()
_softmax_plus_one = bl.softmax_plus_one


def softmax_plus_one(x, null_logit=0.0):  # baselines with a null logit (B9, B10; learned in B1.I.null) bypass F.softmax
    if CUT["policy"] == "profile":
        record_profile(x.log_softmax(dim=-1), 1)
    elif CUT["policy"] is not None:
        log_w = x.log_softmax(dim=-1) if CUT["policy"] == "h2o" else None
        x = x.masked_fill(~keep_mask(x, 3, 1, log_w), float("-inf"))
    return _softmax_plus_one(x, null_logit)


bl.softmax_plus_one = softmax_plus_one


def latest(run_dir):
    files = glob.glob(os.path.join(run_dir, "lm_epoch_*.pt"))
    return max(files, key=lambda f: int(re.search(r"lm_epoch_(\d+)\.pt$", f).group(1)))


@torch.no_grad()
def run(model, blocks, dev):
    CUT["kept"] = []
    out = []
    for row in blocks:
        ids = row[None].to(dev)
        logits, _ = model(ids)
        out.append(F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none").cpu())
    kept = sum(CUT["kept"]) / len(CUT["kept"]) if CUT["kept"] else float("nan")
    return torch.stack(out), kept                                   # [n_blocks, N - 1]


def main():
    argv = sys.argv[1:]
    opt = {"--blocks": 32, "--length": 4096}
    for k in list(opt):
        if k in argv:
            j = argv.index(k)
            opt[k] = int(argv[j + 1])
            del argv[j:j + 2]
    cpu = "--cpu" in argv
    argv = [a for a in argv if a != "--cpu"]
    out_path, runs = argv[0], [a.split("=", 1) for a in argv[1:]]
    dev = torch.device("cpu" if cpu else "cuda")
    N = opt["--length"]
    all_blocks = test_blocks("wikitext103", load_tokenizer(), N, "datasets", 4)
    blocks = all_blocks[:opt["--blocks"]]
    cal_blocks = test_blocks("wikitext103", load_tokenizer(), N, "datasets", 4, split="validation")[:8]
    hdr = f"{'model':<6} {'policy':<14}{'cache':>8}{f'ppl {N // 2}-{N}':>14}{'ppl all':>9}{'dNLL p99':>10}{'p99.9':>8}"
    lines = [f"Eviction sweep at L = {N}, {len(blocks)} WikiText-103 test blocks", hdr]
    print("\n".join(lines), flush=True)
    for name, run_dir in runs:
        ckpt = torch.load(latest(run_dir), map_location="cpu", weights_only=False)
        args = ckpt["args"]
        args.device = dev
        model = build_language_model(args).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        attn = [m for m in model.modules()
                if isinstance(m, (MultiheadIsotropicDKA, MultiheadIsotropicRFA, MultiHeadAttentionLayer))]
        for m in attn:
            m.register_forward_pre_hook(lambda mod, inp: CUT.update(layer=mod))
        has_slopes = slopes(attn[0]) is not None
        settings = [(None, None)] + [("uniform", w) for w in WINDOWS] + [("h2o", w) for w in WINDOWS]
        if not has_slopes or name.startswith("B2"):
            settings += [("stream", w) for w in WINDOWS]
        if has_slopes:
            settings += [("thresh", e) for e in EPS]
        settings += [("cal", e) for e in CAL_EPS] + [("calcap", e) for e in (0.03, 0.01)]
        if isinstance(attn[0], MultiheadIsotropicDKA):
            settings += [("bound", e) for e in BOUND_EPS] + [("boundcap", e) for e in BOUNDCAP_EPS]
            for m in attn:
                print(f"   bound T at eps=0.01: {bound_T(m, 0.01, N).tolist()}", flush=True)
        if POLICIES:
            settings = [x for x in settings if x[0] is None or x[0] in POLICIES.split(",")]
        CUT.update(policy="profile", prof={})
        run(model, cal_blocks, dev)
        for m in attn:
            print(f"   calibrated T at eps=0.01: {calibrated_T(0.01, N, len(cal_blocks))[m].tolist()}", flush=True)
        base = None
        for pol, val in settings:
            CUT.update(policy=pol, W=val if pol != "thresh" else None)
            if pol == "thresh":
                CUT["T"] = {m: (math.log(1 / val) / slopes(m)).ceil().clamp(min=1, max=N).long() for m in attn}
            if pol in ("bound", "boundcap"):
                CUT["T"] = {m: bound_T(m, val, N) for m in attn}
            if pol in ("cal", "calcap"):
                CUT["T"] = calibrated_T(val, N, len(cal_blocks))
            if pol.endswith("cap") if pol else False:
                CUT["T"] = {m: T.clamp(max=CAP) for m, T in CUT["T"].items()}
            nll, kept = run(model, blocks, dev)
            if base is None:
                base, kept = nll, float(N + 1) / 2 * 1.5   # full context: mean of i + 1 over queries N/2..N
            d = (nll - base)[:, min(511, N // 4):].flatten()
            label = "full" if pol is None else f"{pol} {val:g}"
            r = (f"{name:<6} {label:<14}{kept:>8.1f}{math.exp(nll[:, N // 2 - 1:].mean()):>14.2f}"
                 f"{math.exp(nll.mean()):>9.2f}{d.quantile(0.99):>10.3f}{d.quantile(0.999):>8.3f}")
            lines.append(r)
            print(r, flush=True)
        CUT.update(policy=None, W=None)
        del model
        torch.cuda.empty_cache()
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
