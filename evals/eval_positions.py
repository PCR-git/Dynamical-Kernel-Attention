"""In-window diagnostics for trained checkpoints at the training length (default L = 512).

For each model:
  - test perplexity by target position (bins 0-16, 16-64, 64-128, then doubling up to L), which separates how
    well a model does with little context from how well it uses a full window (the average over positions hides
    this);
  - per head (layer 0): the decay rate of a key's weight with lag (mu_h for DKA and RFA, the bias slope for
    B5 / B10 / ALiBi) and its horizon 1 / rate;
  - per head (mean over layers): the share of the update not taken by the null slot, S / (gamma^-2 + S) for DKA,
    the sum of the softmax-plus-one weights for B9 / B10 (1 without a null slot);
  - per head (mean over layers): the learned inverse temperature, if the model has one (RFA, DKA with
    dka_temperature).

usage: python evals/eval_positions.py <out.txt> <name=run_dir> [...] [--length L] [--blocks N] [--cpu]
       e.g. python evals/eval_positions.py positions.txt D3=runs/wikitext103/D3__<date> M3=runs/wikitext103/M3__<date>
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "evals")]   # run from the repo root: python evals/eval_positions.py
import math
import sys

import torch

from eval_eviction import (MultiHeadAttentionLayer, MultiheadIsotropicDKA, MultiheadIsotropicRFA, build_language_model,
                           latest, load_tokenizer, run, slopes, test_blocks)


def load(run_dir, dev):
    ckpt = torch.load(latest(run_dir), map_location="cpu", weights_only=False)
    args = ckpt["args"]
    args.device = dev
    model = build_language_model(args).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def bins(N):
    edges = [0, 16, 64]
    while edges[-1] * 2 < N:
        edges.append(edges[-1] * 2)
    return list(zip(edges, edges[1:] + [N - 1]))


def temperature(m):
    with torch.no_grad():
        if isinstance(m, MultiheadIsotropicRFA):
            return m.sde_params()["tau"].flatten().float().cpu()
        if getattr(m, "log_tau", None) is not None:
            return torch.exp(m.log_tau).float().cpu()
    return None


def main():
    argv = sys.argv[1:]
    opt = {"--blocks": 300, "--length": 512}
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
    blocks = test_blocks("wikitext103", load_tokenizer(), N, "datasets", 4)[:opt["--blocks"]]
    B = bins(N)
    lines = [f"Perplexity by target position at L = {N}, {len(blocks)} WikiText-103 test blocks",
             f"{'model':<8}" + "".join(f"{f'{a}-{b}':>10}" for a, b in B) + f"{'all':>9}"]
    print("\n".join(lines), flush=True)
    heads = []
    for name, run_dir in runs:
        model = load(run_dir, dev)
        attn = [m for m in model.modules()
                if isinstance(m, (MultiheadIsotropicDKA, MultiheadIsotropicRFA, MultiHeadAttentionLayer))]
        share = {}

        def hook(mod, inp, out):
            info = out[1]
            if "step_mean" in info:                                   # DKA: S / (gamma^-2 + S) per head
                s = info["step_mean"].float().cpu()
            elif "attn_mat" in info and getattr(mod.config, "softmax_plus_one", False):
                s = info["attn_mat"].float().sum(-1).mean(dim=(0, 2)).cpu()
            else:
                return
            share[mod] = share.get(mod, 0) + s
        handles = [m.register_forward_hook(hook) for m in attn]
        with torch.no_grad():
            nll, _ = run(model, blocks, dev)
        for h in handles:
            h.remove()
        r = f"{name:<8}" + "".join(f"{math.exp(nll[:, a:b].mean()):>10.2f}" for a, b in B) + f"{math.exp(nll.mean()):>9.2f}"
        lines.append(r)
        print(r, flush=True)
        s = slopes(attn[0])
        if s is not None:
            heads.append(f"{name:<8} decay rate per head (layer 0): " + " ".join(f"{x:.4g}" for x in s.tolist()))
            heads.append(f"{name:<8} horizon 1/rate (tokens):        "
                         + " ".join("inf" if x == 0 else f"{1 / x:.4g}" for x in s.tolist()))
        if share:
            avg = sum(share.values()) / len(share) / len(blocks)
            heads.append(f"{name:<8} update share not taken by the null slot (mean over layers): "
                         + " ".join(f"{x:.2f}" for x in avg.tolist()))
        taus = [t for t in (temperature(m) for m in attn) if t is not None]
        if taus:
            heads.append(f"{name:<8} inverse temperature per head (mean over layers): "
                         + " ".join(f"{x:.3g}" for x in (sum(taus) / len(taus)).tolist()))
        del model
        if not cpu:
            torch.cuda.empty_cache()
    lines += [""] + heads
    print("\n" + "\n".join(heads), flush=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
