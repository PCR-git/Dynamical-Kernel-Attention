"""Per-head far-context probe: does each DKA head's weight on keys beyond the training length help or hurt?

At test length N (default 4096), one head at a time (or one head index in every layer, a "band") drops every key at
lag >= T (default 512, the training length): its logits are set to -inf, so the keys leave both the numerator and
the information S, as in eviction. The change in NLL against full context, over tokens at positions >= T, is
reported in milli-nats per token (mean over test blocks, with its standard error over blocks):
    dNLL > 0   the head's far weight was helping;   dNLL < 0   it was hurting (the far gains came in spite of it).

usage: python probe_heads.py <out.txt> <name=run_dir> [...] [--blocks 32] [--length 4096] [--cut 512] [--bands-only]
"""
import math
import sys
import torch
import eval_eviction as ev
from dka.dka_attention import MultiheadIsotropicDKA
from dka.training.lm_data import load_tokenizer
from dka.training.presets import build_language_model
from eval_extrapolation import test_blocks


def main():
    argv = sys.argv[1:]
    opt = {"--blocks": 32, "--length": 4096, "--cut": 512}
    bands_only = "--bands-only" in argv
    argv = [a for a in argv if a != "--bands-only"]
    for k in list(opt):
        if k in argv:
            j = argv.index(k)
            opt[k] = int(argv[j + 1])
            del argv[j:j + 2]
    out_path, runs = argv[0], [a.split("=", 1) for a in argv[1:]]
    N, T, dev = opt["--length"], opt["--cut"], torch.device("cuda")
    blocks = test_blocks("wikitext103", load_tokenizer(), N, "datasets", 4)[:opt["--blocks"]]
    lines = [f"Per-head far-context probe: drop keys at lag >= {T} in one head (or one head index in every layer) "
             f"at L = {N}, {len(blocks)} WikiText-103 test blocks; dNLL in milli-nats/token over positions >= {T} "
             f"(+ = the far keys helped), +- standard error over blocks"]
    print(lines[0], flush=True)
    for name, run_dir in runs:
        ckpt = torch.load(ev.latest(run_dir), map_location="cpu", weights_only=False)
        args = ckpt["args"]
        args.device = dev
        model = build_language_model(args).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        attn = [m for m in model.modules() if isinstance(m, MultiheadIsotropicDKA)]
        for m in attn:
            m.register_forward_pre_hook(lambda mod, inp: ev.CUT.update(layer=mod))
        H = attn[0].config.n_heads
        full = {m: torch.full((H,), N, dtype=torch.long) for m in attn}

        def probe(cut):                                        # cut: set of (layer, head) -> dNLL stats
            ev.CUT.update(policy="bound", T={m: torch.where(torch.tensor([(li, h) in cut for h in range(H)]),
                                                            T, full[m]) for li, m in enumerate(attn)})
            nll, _ = ev.run(model, blocks, dev)
            d = (nll - base)[:, T - 1:].mean(1) * 1000         # per block
            return float(d.mean()), float(d.std() / math.sqrt(len(d)))

        ev.CUT.update(policy=None)
        base, _ = ev.run(model, blocks, dev)
        with torch.no_grad():
            horizons = [(1 / m.sde_params()["mu"]).flatten().tolist() for m in attn]
        lines += ["", f"{name}: {ev.latest(run_dir)}", f"full context: ppl over positions >= {T} = "
                  f"{math.exp(base[:, T - 1:].mean()):.2f}",
                  "horizon 1/mu per head (layer 0): " + " ".join(f"{x:.0f}" for x in horizons[0])]
        print("\n".join(lines[-3:]), flush=True)

        s, e = probe({(li, h) for li in range(len(attn)) for h in range(H)})
        lines.append(f"every head: {s:+.1f} +- {e:.1f}")
        print(lines[-1], flush=True)
        lines.append(f"{'band (head index, every layer)':<34}" + "".join(f"{h:>14}" for h in range(H)))
        row = []
        for h in range(H):
            s, e = probe({(li, h) for li in range(len(attn))})
            row.append(f"{s:+8.1f} +-{e:4.1f}")
        lines.append(f"{'':<34}" + "".join(f"{x:>14}" for x in row))
        print("\n".join(lines[-2:]), flush=True)
        if not bands_only:
            lines.append(f"{'single head (layer: heads 0..)':<34}" + "".join(f"{h:>14}" for h in range(H)))
            for li in range(len(attn)):
                row = []
                for h in range(H):
                    s, e = probe({(li, h)})
                    row.append(f"{s:+8.1f} +-{e:4.1f}")
                lines.append(f"{f'layer {li}':<34}" + "".join(f"{x:>14}" for x in row))
                print(lines[-1], flush=True)
        ev.CUT.update(policy=None)
        del model
        torch.cuda.empty_cache()
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
