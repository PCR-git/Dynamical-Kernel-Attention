# Dynamical Kernel Attention (DKA)

Code for [*Dynamical Kernel Attention: Estimating the Present from a Record of the Past*](https://arxiv.org/abs/2610.24026)
(Peter Racioppo, 2026).

Standard self-attention compares tokens by similarity but does not account for how much of a past state survives
to the present. DKA derives attention from robust estimation of a latent state evolving under stochastic dynamics.
As a past state decays toward equilibrium, the information it carries about the present vanishes, which bounds the
total evidence a head can draw. A null slot in the normalization lets a head abstain when the evidence is weak, and
discarding old tokens has a bounded effect on a head's output. DKA has the parallel structure and the computational
cost of standard attention.

DKA builds on [Robust Filter Attention (RFA)](https://github.com/PCR-git/Robust-Filter-Attention) (ICML 2026), and
this repository contains both layers. The implementation follows the preprint and is tested against a literal
implementation of its equations. Language-model experiments are in progress.

---

## The model

Each head treats the token sequence as noisy observations of a latent state $x(t)$ that evolves under a linear
stochastic differential equation, $dx = Ax\,dt + G\,dW$, with per-head eigenvalues $\lambda_k = -\mu + i\omega_k$.
The rotations $e^{i\omega_k \Delta t}$ are RoPE's; the decay rate $\mu$ sets how fast a past state is forgotten.

**Past tokens as measurements of the present.** A token at lag $\Delta t$, carried forward by the dynamics,
predicts the current state. Conditioning on the current state under the stationary process gives a scalar
measurement model per head:

$$
h_{ij} = E_{ij}^2, \qquad E_{ij} = e^{-\mu \Delta t_{ij}}, \qquad
\Sigma_n[i,j] = v_\infty\,(1 - E_{ij}^2) + \eta^2 ,
$$

where $h$ is the observation coefficient, $v_\infty = \sigma^2 / 2\mu$ the stationary variance and $\eta^2$ the
measurement noise. As the lag grows, $h \to 0$: a distant token's prediction has decayed to the equilibrium, which
the prior already knows, so it carries no information about the present.

**Robust weights.** Each transported key is compared with the query's backward prediction in the rotated frame,
$R_{ij} = \tilde K_j - E_{ij}\tilde Q_i$, and weighted by a Student-t likelihood:

$$
L_{ij} = -\log \Sigma_n[i,j] - (\nu_s + 1)\log\!\Big(1 + \frac{\|R_{ij}\|^2}{\nu_s\, d\, \Sigma_n[i,j]}\Big),
\qquad a_{ij} = e^{L_{ij}} .
$$

**Update.** Each layer takes one proximal (Kalman information-form) step from the current estimate $V_i$:

$$
\Delta V_i = \frac{\tilde\Phi^+_i \odot \sum_{j \le i} a_{ij} E_{ij} \tilde V_j - S_i V_i}{\gamma^{-2} + S_i},
\qquad S_i = \sum_{j \le i} a_{ij} h_{ij},
$$

that is, a step of size $S_i / (\gamma^{-2} + S_i)$ toward the estimate the context supports. The output is added
to the residual stream, $Z^+ = Z + W_o \Delta V$.

Compared with softmax attention:

- **Information, not total weight.** The normalizer weights each token by $h_{ij}$, so it counts information about
  the present. Distant tokens add nothing to it.
- **A null slot.** The prior contributes $\gamma^{-2}$ to the denominator, equivalent to an extra slot with logit
  $-\log\gamma^2$ and zero innovation. With little evidence the step is small and the head abstains. Zero-decay
  heads with $\gamma^2 = 1$ recover "softmax plus one".
- **A memory window.** For $\mu > 0$, dropping all tokens older than $T$ changes a head's output by at most
  $C e^{-\mu T}$, for any data and any context length (Proposition 3), so its cache can be sized from its decay
  rate.

DKA recovers RFA in the flat-prior limit. Under isotropic decay and noise, the only $O(N^2 d)$ operation is the
query-key product, as in standard attention.

---

## Installation

Python 3.11 and PyTorch 2.11 (CUDA 12.8); `requirements.txt` pins the exact versions.

```
git clone https://github.com/PCR-git/Dynamical-Kernel-Attention.git
cd Dynamical-Kernel-Attention
pip install -r requirements.txt
pip install -e .                  # makes `dka` importable from other projects
```

`pip install -e ".[compile]"` adds Triton for `--compile` on Windows (Linux PyTorch wheels include it);
`".[notebooks]"` adds Jupyter and SciPy.

## Data

```
python download_data.py           # WikiText-103 -> datasets/wikitext-103/
```

This fetches the WikiText-103 parquet files from Hugging Face (`Salesforce/wikitext`, `wikitext-103-v1`, about
300 MB). `python download_data.py --pg19` also fetches PG-19 (the `emozilla/pg19` parquet mirror, about 11 GB) for the
scaled runs; it is tokenized on first use into `datasets/pg19/tokens/` (3.07B training tokens, about 6 GB). The GPT-2 tokenizer is included in `gpt2_tokenizer/`. If Hugging Face is not reachable, download the four
files of `wikitext-103-v1` from the dataset page by other means and put them in `datasets/wikitext-103/`. BabyLM 2025 (used for some RFA results) is not downloaded automatically; place
its 100M-word corpus at `datasets/BabyLM_2025/{train_100M,dev,test}/`.

---

## Using the DKA layer

```python
import torch
from dka import DKAConfig, MultiheadIsotropicDKA

config = DKAConfig(d_model=256, n_heads=8, head_dim=32,      # head_dim is the complex dimension per head
                   frequency_init="spectral", damping=5.0)    # decay coupled to frequency: mu_h = 5 * omega_h,max
layer = MultiheadIsotropicDKA(config)
x = torch.randn(4, 512, 256)
out, info = layer(x)                                          # add `out` to the residual stream
out, info = layer(x, return_metadata=True)                    # + per-token step, evidence, posterior variance
```

Main options of `DKAConfig` (see `dka/config.py`):

| Option | Meaning |
| --- | --- |
| `frequency_init`, `decay_init`, `damping` | RoPE or spectrally coupled frequencies; decay rates $\mu_h = b\,\omega_{h,\max}$ |
| `prior` | the null slot: `"tokens"` (default, $\gamma^{-2} = c_h/\eta^2$, worth $c_h$ agreeing tokens), `"window"`, `"direct"` |
| `process_noise` | learn the forcing $\sigma^2$ (`"variance"`) or the stationary variance $v_\infty$ (`"steady_state"`) |
| `robust_weight` | Student-t weights (default) or the exponential kernel |
| `flat_precision` | $\Sigma_n = \eta^2$ at every lag; with `robust_weight=False` this is the lean variant used at scale |
| `kernel` | how the update is computed: `"explicit"` (default; materializes the weights, all diagnostics), `"flex"` (FlexAttention), `"triton"` (the fused kernel; lean variant), `"two_pass"` / `"blocked"` (references) |
| `inflation` | variance inflation $\zeta$ of the stationary prior (Appendix F of the preprint); `None` = stationary prior |
| `saturation` | experimental: a measurement error shared across tokens, which caps the evidence $S_i$ |

`info` holds the per-head parameters ($\mu$, $\sigma^2$, $\eta^2$, $\gamma^2$, $\nu$, ...) and the mean step; with
`return_metadata=True` also the per-token step $S_i/(\gamma^{-2}+S_i)$, the evidence $\log S_i$, the posterior
variance and the attention weights.

## Fused kernels

The explicit layer materializes every pair's weight, so its memory grows with the square of the length. Two fused
implementations compute the same update without doing so (memory linear in the length), and both are tested against
the explicit layer:

- `dka/flex.py`: DKA and RFA through PyTorch's FlexAttention, as two softmax passes that share the logits
  (`kernel="flex"`; Student-t or exponential weight, lag-dependent precision).
- `dka/fused.py`: Triton kernels for the lean variants (flat precision, exponential weight) of DKA and RFA
  (`kernel="triton"`): one pass over the keys with a running maximum and the sums $\sum p\tilde V$ and $\sum pE$, and a
  backward pass that reduces every gradient over rows, columns or blocks, without atomic additions. At width 128 the
  lean DKA costs about 1.1 times softmax attention in the same kernels. Needs a CUDA GPU and `2 * head_dim`
  a power of two, at least 16.
- `dka/fused_reference.py`: the fused kernels' specification, and a blocked PyTorch reference of them, tile by tile,
  with the hand-written backward (`kernel="blocked"`, CPU), checked against the explicit layer and by `gradcheck`.

The baselines' fused path (`AttentionConfig.kernel="fused"`) uses `scaled_dot_product_attention` (flash or cuDNN
attention) for RoPE and FlexAttention for ALiBi and the null slot. `train_lm.py --kernel flex|triton` selects these
for every model at once.

## Building models

A `Backbone` is a stack of pre-norm residual blocks; the attention is DKA, RFA or a baseline depending on the type
of `config.attention`:

```python
from dka import DKAConfig
from dka.model import BackboneConfig, Backbone, LanguageModel

attn = DKAConfig(d_model=256, n_heads=8, head_dim=32, frequency_init="spectral", damping=5.0)
backbone = Backbone(BackboneConfig(attention=attn, n_blocks=6, ffn_dim=1024, io_projections=False))
lm = LanguageModel(backbone, vocab_size=50257, embed_dim=256)
```

`dka/training/presets.py` defines the named models below (`build_language_model(make_args("D3"))`).

## Training language models

```
python train_lm.py --dataset wikitext103 --preset D3 --compile
python train_lm.py --dataset wikitext103 --preset D2 --set damping=0.5     # override any args field
python train_lm.py --dataset wikitext103 --preset D3.b1.r1e4 --compile --kernel flex --bf16   # fused kernels, bf16
python train_lm.py --dataset pg19 --preset D3.b1.r1e4.pg4096 --compile       # scaled: lean DKA through Triton
python eval_extrapolation.py --dataset wikitext103 runs/wikitext103/D3__<date>   # test perplexity at L = 512 ... 4096
python evals/eval_eviction.py eviction.txt D3=runs/wikitext103/D3__<date>             # perplexity vs KV-cache size at 4096
python evals/eval_positions.py positions.txt D3=runs/wikitext103/D3__<date>          # perplexity by position at 512, per-head diagnostics
python evals/eval_window.py window.txt B1=runs/wikitext103/B1__<date> D3=runs/wikitext103/D3__<date>   # sliding window vs full context, paired
python evals/eval_long.py runs/pg19/D3.b1.r1e4.pg4096__<date> ...   # PG-19 test perplexity within books, 4k-32k, same tokens at every length
```

Each run writes to `<out-dir>/<dataset>/<name>__<date>/` (`--out-dir` defaults to `$DKA_RUNS_DIR`, else `./runs`):
the weights of the last two epochs, `resume.pt` while unfinished (`--resume`), progress plots and `metrics.txt`.
The setup is a 6-layer, 8-head model with $d = 256$ (19.4M parameters), trained at length 512 for 15 epochs;
one run takes 6-8 hours on one GPU. `run_queue.ps1` runs several trainings in sequence.

| Presets | Models |
| --- | --- |
| D3.b1.r1e4 | the preprint's DKA: spectrally coupled, $b = 1$, RoPE base $10^4$, learned temperature; its ablations are suffixes (`.est`, `.noVR`, `.exp`, `.dot`, `.flat`, `.lean`, `.N`, `.nonSC`, `.noR`, `.mu0`, `.plain`, ...; see `presets.py`) |
| M3.b1.r1e4 | the preprint's RFA baseline (B3) on the same spectrum; `.lean` its lean form |
| D3.b1.r1500, D3.b1.r5e4, D3.b1.r\<base\>.L1024, B1.L1024 | DKA at RoPE bases 1500 / $10^4$ / $5\cdot10^4$, trained at 512 and 1024 tokens; RoPE at 1024 |
| \<B1, B2, M3.b1.r1e4, D3.b1.r1e4\>.s2 / .s3 | the headline models with two more seeds |
| \<B1, B2, M3.b1.r1e4, D3.b1.r1e4\>.pg1024 / .pg2048 / .pg4096 | the scaled models on PG-19: 12 layers of width 768, 12 heads (153M), bf16, 262k tokens per step by gradient accumulation; DKA (and RFA) in the lean form through the Triton kernel; `.lr<lr>` the 300M-token learning-rate check |
| D1, D2, D3 | DKA: isotropic; spectrally coupled with $b = 0.05$ (D2) and $b = 5$ (D3) |
| D2.Z, D3.Z | DKA with learned variance inflation |
| D2.S, D3.S | DKA with evidence saturation (experimental) |
| D2.N, D3.N | DKA without the null slot |
| D3.T | D3 with a learned inverse temperature per head on the content term (frees the exponent from $\nu$) |
| D3.b02, D3.b05, D3.b1, D3.b2 | DKA with $b = 0.2$ / $0.5$ / $1$ / $2$ and a smaller RoPE base (249 / 710 / 1568 / 3462), keeping D3's slowest head (horizon 625 tokens), with the learned temperature (.noT: without; .h312 / .h156: slowest horizon 312 or 156 tokens); M3.b02 / M3.b05 and B1.b02 / B1.b05 are the RFA and RoPE controls at the same spectrum; B5.b02 / B5.b05 and B10.b02 / B10.b05 the ablations there, with .x05 / .x2 scaling B5's slopes $2\mu_h$ by 0.5 / 2 |
| M1, M2, M3 | RFA (as in the RFA paper); M1.I-M3.I with the innovation output, as DKA |
| M2.1-M2.8 | RFA ablations from the RFA paper |
| B1-B10 | baselines: RoPE, ALiBi, RoPE + decay, SC-RoPE + decay, SC-RoPE + linear bias with slopes $2\mu_h$ (DKA without the noise model and null slot), KERPLE-log, xPos, NoPE, RoPE + softmax plus one, B5 + softmax plus one (DKA without the noise model) |

Speed and memory: `--compile` applies `torch.compile` to the backbone. `--kernel flex|triton` uses the fused kernels
(above), `--bf16` bf16 autocast and `--tf32` TF32 matrix products. `--grad-accum N` accumulates N micro-batches of
`--batch-size` sequences per optimizer step. `LanguageModel.loss()` computes the loss in chunks, so the full logits are
never stored, and `--grad-checkpointing` recomputes each block in the backward pass. Presets can set these defaults
(`train_bf16`, `train_grad_accum`, `train_batch_size`, `train_lr`, `train_epochs`, `train_max_tokens`); on PG-19,
`--max-tokens` trains on a fixed subset of the single pass, and `--epochs` splits the pass into parts.
`benchmarks/` estimates training time and memory on a given GPU (see `benchmarks/README.md`).

## Dynamics experiment

`dynamics/dynamics_compare.py` trains single DKA and RFA layers on a noisy two-dimensional linear system (embedded in 256
dimensions) with fresh simulated data at every step, and compares them with the optimal Kalman filter and with
untrained estimators that use the true dynamics. `--data stationary --test-length 400` tests length extrapolation.

## Tests

```
python tests/test_dka_theory.py       # the preprint's derivations, checked numerically
python tests/test_dka_kernels.py      # DKA kernels against dka/reference.py
python tests/test_dka_layer.py        # the layer against the literal reference, its limits (RFA, RoPE, softmax + 1), Prop. 3,
                                      # and the fused kernels (FlexAttention, Triton: GPU tests skip without a GPU)
python tests/test_dka_integration.py  # DKA inside the models, presets, training logs, torch.compile, bf16
python tests/check_regression.py      # RFA and baseline models against stored reference outputs
python tests/test_smoke.py            # RFA and baseline behaviors
```

`dka/reference.py` implements the model literally (matrix covariances, per-pair loops), as the ground truth for the
layer.

## Repository structure

```
dka/
├── dka_attention.py      # MultiheadIsotropicDKA
├── attention.py          # MultiheadIsotropicRFA
├── config.py             # DKAConfig, RFAConfig
├── kernels.py            # rotations, decay, covariances, residual norms, logits
├── flex.py               # DKA and RFA through FlexAttention (two passes)
├── fused.py              # Triton kernels of the lean DKA and RFA (forward and backward)
├── fused_reference.py    # the fused kernels' specification and blocked reference
├── reference.py          # literal implementation of the DKA model (ground truth for tests)
├── init.py               # frequency, decay and complex-projection initialization
├── legacy.py             # loading checkpoints from earlier versions of the RFA code
├── model/                # Backbone, LanguageModel; baselines/ holds the baseline attention (presets B1-B10)
├── training/             # training loops, datasets, presets
└── utils/
train_lm.py, eval_extrapolation.py, download_data.py   # training, perplexity against length, data download
evals/                    # evaluations of trained models: eviction, positions, sliding window, paired comparisons,
                          # retrieval, per-head probes, and the MQAR / offset fine-tunes
figures/                  # the paper figures
benchmarks/               # GPU benchmark (bench_gpu.py) and training-time / memory calculator (train_time.py)
dynamics/, visualization/, *.ipynb   # simulators (and dynamics_compare.py), plots and notebooks from the RFA paper
tests/
```

The notebooks (`Main - *.ipynb`, `Test LMs.ipynb`) and `figures/` reproduce the RFA paper; see the
[RFA repository](https://github.com/PCR-git/Robust-Filter-Attention) for details.

## License

MIT (see `LICENSE`).

## Citation

```bibtex
@misc{racioppo2026dka,
  title         = {Dynamical Kernel Attention: Estimating the Present from a Record of the Past},
  author        = {Racioppo, Peter},
  year          = {2026},
  eprint        = {2610.24026},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.24026}
}

@inproceedings{racioppo2026rfa,
  title     = {Robust Filter Attention: Self-Attention as Precision-Weighted State Estimation},
  author    = {Racioppo, Peter},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
