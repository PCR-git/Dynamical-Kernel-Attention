"""Long-range retrieval by repetition: how much does a model's loss on a passage drop when the same passage
appeared D tokens earlier?

Each trial builds a sequence of fixed length L (default 4096) ending in a P-token passage (default 64):
    repeat:   [filler] [passage] [gap filler] [passage]
    control:  [filler] [other  ] [gap filler] [passage]
with the second passage at the same final positions and the copies D tokens apart (start to start). Filler,
passages and gaps are drawn from the WikiText-103 test split. The retrieval gain at distance D is the control
NLL minus the repeat NLL on the second passage (tokens 2..P, nats per token), paired over trials; position and
context length are the same in both, so only the presence of the earlier copy differs.

A model with a 512-token sliding window (":swa" after the run directory) cannot retrieve beyond D ~ 512 by
construction; it is the reference for what a model gets without long-range retrieval.

usage: python evals/eval_retrieval.py <out.txt> <name=run_dir[:swa]> [...] [--trials N] [--length L] [--passage P]
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "evals")]   # run from the repo root: python evals/eval_retrieval.py
import math
import sys
import torch
import torch.nn.functional as F
import eval_eviction as ev
from dka.attention import MultiheadIsotropicRFA
from dka.dka_attention import MultiheadIsotropicDKA
from dka.model.baselines import MultiHeadAttentionLayer
from dka.training.lm_data import load_tokenizer
from dka.training.presets import build_language_model
from eval_extrapolation import test_blocks

DISTANCES = (64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 3968)


def build_trials(stream, n_trials, L, P, seed=0):
    """[n_distances, n_trials, 2, L] token ids (repeat, control) and the distances used."""
    g = torch.Generator().manual_seed(seed)
    dists = [d for d in DISTANCES if d + P <= L]
    out = torch.empty(len(dists), n_trials, 2, L, dtype=torch.long)
    for t in range(n_trials):
        o = torch.randint(0, len(stream) - L - 2 * P, (3,), generator=g)
        passage, other = stream[o[0]:o[0] + P], stream[o[1]:o[1] + P]
        filler = stream[o[2]:o[2] + L]                                # the rest of the sequence, contiguous text
        for k, d in enumerate(dists):
            for c, first in enumerate((passage, other)):
                x = filler.clone()
                x[L - P:] = passage                                    # second copy at the end
                x[L - P - d:L - d] = first                             # first copy (or the control passage)
                out[k, t, c] = x
    return out, dists


@torch.no_grad()
def passage_nll(model, x, P, dev):
    logits, _ = model(x[None].to(dev))
    nll = F.cross_entropy(logits[0, -P:-1].float(), x[None, -P + 1:].to(dev)[0], reduction="none")
    return float(nll.mean())                                           # tokens 2..P of the second copy


def main():
    argv = sys.argv[1:]
    opt = {"--trials": 32, "--length": 4096, "--passage": 64}
    for k in list(opt):
        if k in argv:
            j = argv.index(k)
            opt[k] = int(argv[j + 1])
            del argv[j:j + 2]
    out_path, runs = argv[0], [a.split("=", 1) for a in argv[1:]]
    L, P = opt["--length"], opt["--passage"]
    dev = torch.device("cuda")
    stream = test_blocks("wikitext103", load_tokenizer(), L, "datasets", 4).flatten()
    trials, dists = build_trials(stream, opt["--trials"], L, P)
    hdr = f"{'model':<12}" + "".join(f"{d:>8}" for d in dists)
    lines = [f"Retrieval gain (control NLL - repeat NLL, nats/token on a {P}-token passage) vs distance D, "
             f"L = {L}, {opt['--trials']} trials, WikiText-103 test", hdr]
    print("\n".join(lines), flush=True)
    se_lines, ctl_lines = [f"{'s.e.':<12}"], []
    for name, path in runs:
        swa = path.endswith(":swa")
        path = path[:-4] if swa else path
        ckpt = torch.load(ev.latest(path), map_location="cpu", weights_only=False)
        args = ckpt["args"]
        args.device = dev
        model = build_language_model(args).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        for m in model.modules():
            if isinstance(m, (MultiheadIsotropicDKA, MultiheadIsotropicRFA, MultiHeadAttentionLayer)):
                m.register_forward_pre_hook(lambda mod, inp: ev.CUT.update(layer=mod))
        ev.CUT.update(policy="uniform" if swa else None, W=512, kept=[])
        gain = torch.zeros(len(dists), trials.shape[1], dtype=torch.float64)
        ctl = torch.zeros(len(dists), dtype=torch.float64)
        for k in range(len(dists)):
            for t in range(trials.shape[1]):
                rep, con = (passage_nll(model, trials[k, t, c], P, dev) for c in (0, 1))
                gain[k, t] = con - rep
                ctl[k] += con / trials.shape[1]
        ev.CUT.update(policy=None)
        m, s = gain.mean(1), gain.std(1) / math.sqrt(gain.shape[1])
        row = f"{name:<12}" + "".join(f"{x:>8.3f}" for x in m.tolist())
        lines.append(row)
        se_lines.append(f"  {name:<10}" + "".join(f"{x:>8.3f}" for x in s.tolist()))
        ctl_lines.append(f"  {name:<10}" + "".join(f"{math.exp(x):>8.1f}" for x in ctl.tolist()))
        print(row, flush=True)
        del model
        torch.cuda.empty_cache()
    lines += ["", "standard errors over trials"] + se_lines[1:]
    lines += ["", "control perplexity on the passage (no earlier copy)"] + ctl_lines
    print("\n".join(lines[-(len(se_lines) + len(ctl_lines) + 3):]), flush=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
