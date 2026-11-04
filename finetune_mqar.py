"""Multi-query associative recall (MQAR) after fine-tuning: can a pretrained LM, taught the task at the training
length, recall key-value bindings across longer gaps?

Each sequence holds N key-value pairs, a gap of filler text, then the N keys again in a random order, each followed
by its value (teacher forcing, as in Zoology / Arora et al. 2023):
    [prefix filler] k1 v1 k2 v2 ... kN vN [gap filler] k_p1 v_p1 k_p2 v_p2 ... k_pN v_pN
Keys and values are drawn per sequence from two disjoint fixed sets of GPT-2 tokens, so a binding exists only in
the context. The loss and accuracy are on the values of the query block only.

Fine-tuning uses sequences of the training length (512) with N drawn from --train-pairs and the gap uniform over
what is left; filler comes from WikiText-103 validation. Evaluation uses test-split filler, a 16-token prefix and
gaps from GAPS (beyond 512 is extrapolation), at each N of --eval-pairs; it runs before fine-tuning (zero-shot) and
after. ":swa" after the run directory evaluates with a 512-token sliding window (as eval_retrieval.py); fine-tuning
is unaffected, since its sequences fit in the window.

usage: python finetune_mqar.py <out.txt> <name=run_dir[:swa]> [...] [--steps 2000] [--lr 1e-4] [--batch-size 16]
           [--train-pairs 4,8,16,32] [--eval-pairs 8,32] [--trials 128] [--scratch] [--save]
--scratch trains the same architecture from random initialization instead of the checkpoint; --save writes the
fine-tuned weights to <run_dir>/mqar_<N>steps.pt.
"""
import math
import os
import sys
import time
import torch
import eval_eviction as ev
from dka.attention import MultiheadIsotropicRFA
from dka.dka_attention import MultiheadIsotropicDKA
from dka.model.baselines import MultiHeadAttentionLayer
from dka.training.lm_data import load_tokenizer
from dka.training.presets import build_language_model
from dka.utils import seed_everything
from eval_extrapolation import test_blocks

GAPS = (0, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072)
SEQ_LEN, PREFIX, N_SYMBOLS = 512, 16, 512


def symbol_sets(seed=0):
    """Two disjoint sets of N_SYMBOLS token ids (keys, values) from the common part of the GPT-2 vocabulary."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(20000 - 256, generator=g)[:2 * N_SYMBOLS] + 256
    return ids[:N_SYMBOLS], ids[N_SYMBOLS:]


def make_sequence(stream, keys, vals, n_pairs, gap, prefix, g):
    """(input_ids [L], labels [L]) with labels = -100 except at the values of the query block."""
    kv = torch.randperm(N_SYMBOLS, generator=g)[:n_pairs], torch.randperm(N_SYMBOLS, generator=g)[:n_pairs]
    k, v = keys[kv[0]], vals[kv[1]]
    pairs = torch.stack([k, v], 1).flatten()
    perm = torch.randperm(n_pairs, generator=g)
    query = torch.stack([k[perm], v[perm]], 1).flatten()
    o = int(torch.randint(0, len(stream) - prefix - gap - 1, (1,), generator=g))
    filler = stream[o:o + prefix + gap]
    x = torch.cat([filler[:prefix], pairs, filler[prefix:], query])
    labels = torch.full_like(x, -100)
    labels[-2 * n_pairs + 1::2] = x[-2 * n_pairs + 1::2]              # the values of the query block
    return x, labels


def train_batch(stream, keys, vals, train_pairs, batch_size, g):
    xs, ys = [], []
    for _ in range(batch_size):
        n = train_pairs[int(torch.randint(len(train_pairs), (1,), generator=g))]
        free = SEQ_LEN - 4 * n
        prefix = int(torch.randint(0, free + 1, (1,), generator=g))  # gap uniform over 0..free
        x, y = make_sequence(stream, keys, vals, n, free - prefix, prefix, g)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def evaluate(model, stream, keys, vals, eval_pairs, trials, swa, dev, max_tokens=4096):
    """{(n_pairs, gap): (accuracy, mean probability of the value)} on the query values."""
    model.eval()
    ev.CUT.update(policy="uniform" if swa else None, W=512, kept=[])
    out = {}
    for n in eval_pairs:
        for gap in GAPS:
            g = torch.Generator().manual_seed(1000 + 7 * n + gap)    # the same sequences for every model
            correct = prob = count = 0
            batch = max(1, max_tokens // (PREFIX + gap + 4 * n))      # full-vocabulary logits dominate memory
            for b0 in range(0, trials, batch):
                xs, ys = zip(*(make_sequence(stream, keys, vals, n, gap, PREFIX, g)
                               for _ in range(min(batch, trials - b0))))
                x, y = torch.stack(xs).to(dev), torch.stack(ys).to(dev)
                logits, _ = model(x)
                pred_pos = (y[:, 1:] != -100)                        # logits at t predict token t + 1
                lg, tgt = logits[:, :-1][pred_pos].float(), y[:, 1:][pred_pos]
                correct += int((lg.argmax(-1) == tgt).sum())
                prob += float(lg.softmax(-1).gather(1, tgt[:, None]).sum())
                count += tgt.numel()
            out[n, gap] = (correct / count, prob / count)
    ev.CUT.update(policy=None)
    model.train()
    return out


def format_table(name, stage, res, eval_pairs):
    lines = []
    for n in eval_pairs:
        lines.append(f"{name:<12}{stage:<8}N={n:<4}" + "".join(f"{100 * res[n, gap]:>7.1f}" for gap in GAPS))
    return lines


def main():
    argv = sys.argv[1:]
    opt = {"--steps": "2000", "--lr": "1e-4", "--batch-size": "16", "--train-pairs": "4,8,16,32",
           "--eval-pairs": "8,32", "--trials": "128"}
    flags = {f: f in argv for f in ("--scratch", "--save")}
    argv = [a for a in argv if a not in flags]
    for k in list(opt):
        if k in argv:
            j = argv.index(k)
            opt[k] = argv[j + 1]
            del argv[j:j + 2]
    out_path, runs = argv[0], [a.split("=", 1) for a in argv[1:]]
    steps, lr, bs = int(opt["--steps"]), float(opt["--lr"]), int(opt["--batch-size"])
    train_pairs = [int(x) for x in opt["--train-pairs"].split(",")]
    eval_pairs = [int(x) for x in opt["--eval-pairs"].split(",")]
    trials = int(opt["--trials"])
    dev = torch.device("cuda")

    tok = load_tokenizer()
    train_stream = test_blocks("wikitext103", tok, SEQ_LEN, "datasets", 4, split="validation").flatten()
    test_stream = test_blocks("wikitext103", tok, SEQ_LEN, "datasets", 4).flatten()
    keys, vals = symbol_sets()

    hdr = f"{'model':<12}{'stage':<8}{'pairs':<6}" + "".join(f"{g:>7}" for g in GAPS)
    lines = [f"MQAR accuracy (%) on query values vs gap; fine-tuned {steps} steps x {bs} sequences of {SEQ_LEN} "
             f"tokens, N in {train_pairs}, lr {lr}{', from scratch' if flags['--scratch'] else ''}; "
             f"{trials} test sequences per cell", hdr]
    print("\n".join(lines), flush=True)
    prob_lines = ["", "mean probability of the correct value (final)", hdr]
    for name, path in runs:
        swa = path.endswith(":swa")
        path = path[:-4] if swa else path
        ckpt = torch.load(ev.latest(path), map_location="cpu", weights_only=False)
        args = ckpt["args"]
        args.device = dev
        seed_everything(2025)
        model = build_language_model(args).to(dev)
        if not flags["--scratch"]:
            model.load_state_dict(ckpt["model_state_dict"])
        for m in model.modules():
            if isinstance(m, (MultiheadIsotropicDKA, MultiheadIsotropicRFA, MultiHeadAttentionLayer)):
                m.register_forward_pre_hook(lambda mod, inp: ev.CUT.update(layer=mod))

        res0 = evaluate(model, test_stream, keys, vals, eval_pairs, trials, swa, dev)
        rows = format_table(name, "before", {k: v[0] for k, v in res0.items()}, eval_pairs)
        lines += rows
        print("\n".join(rows), flush=True)

        opt_ = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        warm = max(1, steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt_, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
        g = torch.Generator().manual_seed(2025)
        model.backbone.compile()                                      # fixed training shapes (~20x faster DKA)
        model.train()
        t0, run_loss = time.time(), 0.0
        for step in range(steps):
            x, y = train_batch(train_stream, keys, vals, train_pairs, bs, g)
            loss, _ = model.loss(x.to(dev), y.to(dev), causal=True)
            opt_.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_.step()
            sched.step()
            run_loss += loss.item()
            if (step + 1) % 100 == 0:
                print(f"  {name} step {step + 1}/{steps}  loss {run_loss / 100:.3f}  {time.time() - t0:.0f}s",
                      flush=True)
                run_loss = 0.0

        model.backbone._compiled_call_impl = None                     # eager again for the varying eval lengths
        res = evaluate(model, test_stream, keys, vals, eval_pairs, trials, swa, dev)
        rows = format_table(name, "after", {k: v[0] for k, v in res.items()}, eval_pairs)
        lines += rows
        prob_lines += [f"{name:<12}{'after':<8}N={n:<4}" + "".join(f"{res[n, gap][1]:>7.3f}" for gap in GAPS)
                       for n in eval_pairs]
        print("\n".join(rows), flush=True)
        if flags["--save"]:
            torch.save({"model_state_dict": model.state_dict(), "args": args},
                       os.path.join(path, f"mqar_{steps}steps{'_scratch' if flags['--scratch'] else ''}.pt"))
        del model, opt_
        torch.cuda.empty_cache()
        with open(out_path, "w") as f:                                # rewritten after every model
            f.write("\n".join(lines + prob_lines) + "\n")


if __name__ == "__main__":
    main()
