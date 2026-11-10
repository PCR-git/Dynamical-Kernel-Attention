"""
Two tests of what DKA's evidence S_i means (CPU; prints tables, caches C:/dka_runs/evidence_checks_<name>.npz):

1. Induction: on the repeated-text trials of eval_retrieval at D = 128 (64 trials, 512 tokens), each head's weight
   from a second-copy token at t onto key t - D + 1 (the token after the earlier occurrence of the current token,
   the classic induction target), repeat minus control, against the head's change in log S_i. Does the evidence
   pick out the induction heads?
2. Relevant vs irrelevant context: a 64-token passage after its own preceding 448 tokens, against the same passage
   after 448 tokens from a distant part of the test split (64 trials). The change in each head's log S_i and null
   share gamma^-2 / (gamma^-2 + S_i), and in the loss, on passage tokens 17-64 (so that the fast heads' local
   context lies inside the passage in both conditions).

    python figures/dka_evidence_checks.py [run_name]      # default D3.b1.r1e4
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

N, P, D, N_TRIALS, SKIP = 512, 64, 128, 64, 16
CTX = N - P


@torch.no_grad()
def forward(model, x):
    """NLL of the next token [N-1]; per layer: weights [N, N, H], log S_i [N, H], null share [N, H]."""
    logits, infos = model(x[None], causal=True, return_metadata=True)
    nll = F.cross_entropy(logits[0, :-1].float(), x[1:], reduction="none").numpy()
    return nll, [(i["attn_mat"][0].numpy(), i["log_evidence"][0].numpy(), i["null_share"][0].numpy()) for i in infos]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import REPO, latest_run, load_model
    from eval_extrapolation import test_blocks
    from eval_retrieval import build_trials
    from dka.training.lm_data import load_tokenizer

    name = sys.argv[1] if len(sys.argv) > 1 else "D3.b1.r1e4"
    model, _ = load_model(latest_run(name), torch.device("cpu"))
    horizon = 1 / model.backbone.blocks[0].attn.sde_params()["mu"].numpy()
    tokenizer = load_tokenizer(os.path.join(REPO, "gpt2_tokenizer"))
    stream = test_blocks("wikitext103", tokenizer, N, os.path.join(REPO, "datasets"), 4).reshape(-1)

    # --- 1. Induction scores at D = 128 ---
    trials, dists = build_trials(stream, N_TRIALS, N, P)
    k = dists.index(D)
    t = np.arange(N - P, N)                                     # second-copy positions
    ind, dlogS = [], []                                         # [trial, layers, H]
    for tr in range(N_TRIALS):
        (_, rep), (_, ctl) = forward(model, trials[k, tr, 0]), forward(model, trials[k, tr, 1])
        ind.append([(A_r[t, t - D + 1] - A_c[t, t - D + 1]).mean(0) for (A_r, _, _), (A_c, _, _) in zip(rep, ctl)])
        dlogS.append([(s_r[t[1:]] - s_c[t[1:]]).mean(0) for (_, s_r, _), (_, s_c, _) in zip(rep, ctl)])
    ind, dlogS = np.array(ind), np.array(dlogS)

    # --- 2. Relevant vs irrelevant context ---
    g = torch.Generator().manual_seed(1)
    q = np.arange(CTX + SKIP, N)                                # passage tokens 17..64
    rel = {"dlogS": [], "dnull": [], "dnll": []}
    for tr in range(N_TRIALS):
        o = int(torch.randint(CTX, len(stream) - P, (1,), generator=g))
        while True:                                             # an unrelated context, far from the passage
            r = int(torch.randint(0, len(stream) - CTX, (1,), generator=g))
            if abs(r - o) > 20000:
                break
        passage = stream[o:o + P]
        same, other = torch.cat([stream[o - CTX:o], passage]), torch.cat([stream[r:r + CTX], passage])
        (n_s, l_s), (n_o, l_o) = forward(model, same), forward(model, other)
        rel["dlogS"].append([(s1[q] - s2[q]).mean(0) for (_, s1, _), (_, s2, _) in zip(l_s, l_o)])
        rel["dnull"].append([(z1[q] - z2[q]).mean(0) for (_, _, z1), (_, _, z2) in zip(l_s, l_o)])
        rel["dnll"].append(n_s[q[:-1]].mean() - n_o[q[:-1]].mean())
    rel = {key: np.array(v) for key, v in rel.items()}
    np.savez(f"C:/dka_runs/evidence_checks_{name}.npz", ind=ind, dlogS=dlogS, horizon=horizon,
             **{"rel_" + key: v for key, v in rel.items()})

    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    mse = lambda x: (x.mean(0), x.std(0) / np.sqrt(len(x)))
    print("horizons:", np.round(horizon, 3))
    m, se = mse(ind)
    print(f"\n=== 1. Induction score at D = {D} (weight on key t-D+1, repeat - control) ===\n", m, "\n s.e.\n", se)
    print("Spearman over the 48 heads, induction score vs d log S:", round(spearman(m.ravel(), dlogS.mean(0).ravel()), 3))
    order = np.argsort(-m.ravel())[:6]
    print("top heads by induction score: " + ", ".join(
        f"L{i // 8 + 1} h{horizon[i % 8]:.0f} ind {m.ravel()[i]:.3f} dlogS {dlogS.mean(0).ravel()[i]:+.2f}" for i in order))
    for key, label in (("dlogS", "d log S_i"), ("dnull", "d null share")):
        m, se = mse(rel[key])
        print(f"\n=== 2. Relevant - irrelevant context: {label} on passage tokens {SKIP + 1}-{P} ===\n", m,
              "\n s.e.\n", se)
    m, se = mse(rel["dnll"])
    print(f"\nloss, relevant - irrelevant: {m:+.3f} +- {se:.3f} nats/token")


if __name__ == "__main__":
    main()
