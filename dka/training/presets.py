"""
Named model configurations from the paper, as overrides of the notebook-style `args`.

    args = make_args("M3", device=device)     # defaults + preset + any overrides
    model = build_language_model(args)

Presets starting with "B" are dot-product baselines (TransformerNetwork), those starting with "D" are DKA,
the others are RFA. RFA presets ending in ".I" output the innovation W_o (V_bar - V) instead of the
estimate W_o V_bar (the published runs).
"""

from argparse import Namespace

import torch.nn as nn

# Defaults shared by every model (paper setup: 6 blocks, d=256, 8 heads, trained at L=512)
BASE_ARGS = dict(
    # Architecture
    vocab_size=50257, d_e=256, d_k_total=256, d_v_total=256, n_heads=8, num_blocks=6,
    seq_len=512,                      # training context; also sets the decay scale when scale_decay_by_time_interval
    use_real_input_norm=True, use_real_output_norm=True, use_outer_residual=True,
    grad_checkpointing=False,         # recompute blocks in backward (much less memory, ~30% slower)
    io_projections=True,              # extra d->d maps around the blocks (as in the published runs)
    compute_metadata=False,           # return attention matrices etc. in eval mode (for plots)

    # RFA dynamics
    use_SC_RoPE=False, damping=0.05, zero_frac=0.25,
    use_log_linear_decay=True, scale_decay_by_time_interval=True, max_fixed_decay=5.0,
    omega_rescale=1.0, learn_rotations=False,
    rope_base=10000.0,                # base of the RoPE spectrum (RoPE, SC-RoPE, SC-RFA and SC-DKA frequencies)
    bias_slope_scale=1.0,             # B5 / B10: multiplies the coupled slopes 2 mu_h (1 = DKA's prediction)

    # RFA noise model
    use_ss_process_noise=False, target_ss_process_var=0.1, target_measurement_var=1.0,

    # RFA attention weights and ablation switches
    use_robust_weight=True, additive_bias_type=1, multiplicative_bias_type=1,
    use_full_residual_norm=1, rotate_values=True, zero_rotations=False, lambda_real_zero=False,
    zero_process_noise=False, zero_key_measurement_noise=False,

    # Numerics
    epsilon=1e-5, min_exponent=-10, max_exponent=0,

    # Attention family of the non-baseline presets: "rfa" or "dka"
    attention_type="rfa",
    rfa_output="estimate",            # RFA: "estimate" (published runs) or "innovation" (paper Alg. 1, as DKA)
    rfa_prior="flat",                 # RFA: "flat" (the paper) or "tokens" (DKA's null slot, gamma^-2 worth c_h tokens)
    rfa_init_prior_weight=1.0,        # initial c_h for rfa_prior="tokens"

    # DKA (only used when attention_type="dka"; dynamics and noise use the RFA flags above)
    dka_prior="tokens",               # null slot gamma^-2 = c_h / eta^2 ("tokens"), c_h S_bar ("window"), "direct", or "none"
    dka_init_prior_weight=1.0,        # initial c_h: the prior is worth c_h agreeing tokens
    dka_init_prior_var=1.0,           # initial gamma^2 for dka_prior="direct"
    dka_inflation=None,               # None = stationary prior; zeta in [0, 1) = initial variance inflation (App. F)
    dka_learn_inflation=True,
    dka_saturation=None,              # None = independent tokens; n_sat > 0 = evidence saturates at n_sat / eta^2 (R12)
    dka_learn_saturation=True,
    dka_temperature=False,            # True = learned inverse temperature per head on DKA's content term
    dka_output="innovation",          # "innovation" = W_o dV (the paper) or "estimate" = W_o (V + dV), as RFA's runs

    # Baseline attention (only used by "B" presets)
    use_rope=True, use_alibi=False, use_relative_decay_vanilla=False, use_sc_linear_bias=False,
    use_kerple_log=False, use_xpos=False, use_softmax_one=False,
    yarn_original_length=None,        # inference: e.g. --set yarn_original_length=512 applies YaRN to RoPE models
)

_SC = dict(use_SC_RoPE=True, scale_decay_by_time_interval=False, use_log_linear_decay=False)
def sc_base(b, horizon, n_heads=8, n_freqs=16):
    """RoPE base for which the slowest SC head has decay horizon 1/mu = `horizon` tokens at damping b.

    The SC spectrum splits base^(-k / (H F)), k = 0..H F - 1, across H heads of F frequencies, highest first, and
    mu_h = b omega_h,max. The fastest head has omega_max = 1, so its horizon is 1/b whatever the base; the slowest
    head's largest frequency is base^(-(H - 1) F / (H F)), so its horizon is base^((H - 1) / H) / b.
    """
    return (horizon * b) ** (n_heads / (n_heads - 1))


# Spectra for the paper-size model (8 heads of 16 frequencies), parametrized by b and the slowest horizon.
# 625 tokens is D3's slowest horizon (b = 5, base 10000), about 1.2 L.
_BASE_B02 = sc_base(0.2, 625)       # ~249
_BASE_B05 = sc_base(0.5, 625)       # ~710
_DKA = dict(attention_type="dka", zero_frac=0.0)
_NO_DLE = dict(zero_process_noise=True, zero_key_measurement_noise=True, additive_bias_type=0, multiplicative_bias_type=0)

PRESETS = {
    # --- Baselines ---
    "B1":   dict(use_rope=True, use_alibi=False),                                   # RoPE
    "B2":   dict(use_rope=False, use_alibi=True),                                   # ALiBi
    "B3":   dict(use_rope=True, use_alibi=False, use_relative_decay_vanilla=True,   # RoPE + decay
                 use_SC_RoPE=False, damping=0.05),
    "B4":   dict(use_rope=False, use_alibi=False, use_relative_decay_vanilla=True,  # SC-RoPE + decay
                 use_SC_RoPE=True, damping=0.05),
    "B5":   dict(use_rope=False, use_alibi=False, use_sc_linear_bias=True,          # SC-RoPE + coupled linear bias
                 use_SC_RoPE=True, damping=5.0),                                   # slopes 2 mu_h, as D3 / M3
    "B6":   dict(use_rope=False, use_alibi=False, use_kerple_log=True),             # KERPLE-log
    "B7":   dict(use_rope=False, use_alibi=False, use_xpos=True),                   # xPos
    "B8":   dict(use_rope=False, use_alibi=False),                                  # NoPE
    "B9":   dict(use_rope=True, use_alibi=False, use_softmax_one=True),             # RoPE + softmax plus one
    "B10":  dict(use_rope=False, use_alibi=False, use_sc_linear_bias=True,          # B5 + softmax plus one: DKA
                 use_SC_RoPE=True, damping=5.0, use_softmax_one=True),             # without the noise model
    # --- RFA models ---
    "M1":   dict(use_log_linear_decay=False, damping=0.05),                         # Isotropic RFA
    "M2":   dict(_SC, damping=0.05),                                                # SC-RFA (in-window)
    "M3":   dict(_SC, damping=5.0, zero_frac=0.0, target_ss_process_var=0.5),       # SC-RFA (extrapolation)
    # --- Structural ablations of M2 ---
    "M2.1": dict(_SC, damping=0.05, use_robust_weight=False),                       # exponential weight
    "M2.2": dict(_SC, damping=0.05, **_NO_DLE),                                     # flat prior
    "M2.3": dict(_SC, damping=0.05, multiplicative_bias_type=0),                    # no multiplicative gate
    "M2.4": dict(_SC, damping=0.05, rotate_values=False),                           # no value rotations
    "M2.5": dict(_SC, damping=0.05, zero_rotations=True),                           # no rotations
    "M2.6": dict(_SC, damping=0.0, use_robust_weight=False, use_full_residual_norm=0, # pure rotation
                 lambda_real_zero=True, max_fixed_decay=0.0, **_NO_DLE),
    # Minimal SC-RFA: rotary dot-product attention plus only the additive precision bias
    # -log(sigma^2 dt + eta^2 + gamma^2) (zero decay, exponential weight, no gate, no value rotation).
    # damping=0.05 is kept so sigma^2 is initialized coupled to the spectrum (2 * 0.05 * omega_max * 0.1).
    # (Trained 2026-10-23 as run "M2.8", before the presets were renumbered to match the paper.)
    "M2.7": dict(_SC, damping=0.05, lambda_real_zero=True, use_robust_weight=False, use_full_residual_norm=0,
                 multiplicative_bias_type=0, rotate_values=False),
    # Rotary limit: M2.6 without value rotation, i.e. exactly rotary dot-product attention (conjugate-pair
    # frequencies, learned temperature), RoPE implemented inside RFA. Add --set use_SC_RoPE=False for the
    # standard (unpartitioned) RoPE spectrum in every head.
    # (Trained 2026-10-18/19 as runs "M2.7" and "M2.7_rope_spec", before the renumbering.)
    "M2.8": dict(_SC, damping=0.0, use_robust_weight=False, use_full_residual_norm=0,
                 lambda_real_zero=True, max_fixed_decay=0.0, rotate_values=False, **_NO_DLE),
    # --- RFA with the innovation output (comparable with DKA) ---
    "M1.I": dict(use_log_linear_decay=False, damping=0.05, rfa_output="innovation"),
    "M2.I": dict(_SC, damping=0.05, rfa_output="innovation"),
    "M3.I": dict(_SC, damping=5.0, zero_frac=0.0, target_ss_process_var=0.5, rfa_output="innovation"),
    # --- DKA models (counterparts of M1-M3, with slow heads instead of zero-decay heads) ---
    "D1":   dict(_DKA, use_log_linear_decay=False, damping=0.05),                   # Isotropic DKA
    "D2":   dict(_DKA, **_SC, damping=0.05),                                        # SC-DKA (in-window)
    "D3":   dict(_DKA, **_SC, damping=5.0, target_ss_process_var=0.5),              # SC-DKA (extrapolation)
    # Variance inflation (App. F), learned from zeta = 0.5
    "D2.Z": dict(_DKA, **_SC, damping=0.05, dka_inflation=0.5),
    "D3.Z": dict(_DKA, **_SC, damping=5.0, target_ss_process_var=0.5, dka_inflation=0.5),
    # Evidence saturation (R12), learned from n_sat = 1 (D2's evidence within 512 tokens is ~0.1-1 agreeing tokens)
    "D2.S": dict(_DKA, **_SC, damping=0.05, dka_saturation=1.0),
    "D3.S": dict(_DKA, **_SC, damping=5.0, target_ss_process_var=0.5, dka_saturation=1.0),
    # No null slot (gamma^-2 = 0), stationary measurement model kept: isolates the null slot (vs D2 / D3)
    "D2.N": dict(_DKA, **_SC, damping=0.05, dka_prior="none"),
    "D3.N": dict(_DKA, **_SC, damping=5.0, target_ss_process_var=0.5, dka_prior="none"),
    # Learned inverse temperature on the content term (frees kappa from nu; -log Sigma and the null slot unchanged)
    "D3.T": dict(_DKA, **_SC, damping=5.0, target_ss_process_var=0.5, dka_temperature=True),
    # --- Spectrum: b sets each head's Q = 1 / (2b) (phase advance 1/b rad over its horizon 1/mu), the base sets
    # the range. Bases chosen so the slowest head keeps D3's horizon 1/mu = 625 tokens (see _BASE_B02 above).
    # These include the learned temperature (as RFA does); the .noT variants turn it off.
    "D3.b02": dict(_DKA, **_SC, damping=0.2, rope_base=_BASE_B02, target_ss_process_var=0.5, dka_temperature=True),
    "D3.b05": dict(_DKA, **_SC, damping=0.5, rope_base=_BASE_B05, target_ss_process_var=0.5, dka_temperature=True),
    "D3.b02.noT": dict(_DKA, **_SC, damping=0.2, rope_base=_BASE_B02, target_ss_process_var=0.5),
    "D3.b05.noT": dict(_DKA, **_SC, damping=0.5, rope_base=_BASE_B05, target_ss_process_var=0.5),
    # b = 1 and 2 at the same slowest horizon (625 tokens): locate the extrapolation boundary along b
    "D3.b1": dict(_DKA, **_SC, damping=1.0, rope_base=sc_base(1.0, 625), target_ss_process_var=0.5, dka_temperature=True),
    "D3.b2": dict(_DKA, **_SC, damping=2.0, rope_base=sc_base(2.0, 625), target_ss_process_var=0.5, dka_temperature=True),
    # b = 0.5 at RoPE's base 10^4 (the best SC-RFA damping in the RFA paper) and its RFA twin, no zero-decay heads
    "D3.b05.r1e4": dict(_DKA, **_SC, damping=0.5, target_ss_process_var=0.5, dka_temperature=True),
    "M3.b05.r1e4": dict(_SC, damping=0.5, zero_frac=0.0, target_ss_process_var=0.5),
    "D3.b1.r1e4": dict(_DKA, **_SC, damping=1.0, target_ss_process_var=0.5, dka_temperature=True),   # b = 1, base 10^4
    "M3.b1.r1e4": dict(_SC, damping=1.0, zero_frac=0.0, target_ss_process_var=0.5),
    # b = 2 at base 10^4 (slowest horizon ~1581): with b = 0.5, 1, 5 the spectrum sweep at RoPE's base
    "D3.b2.r1e4": dict(_DKA, **_SC, damping=2.0, target_ss_process_var=0.5, dka_temperature=True),
    # Output form and null slot at b = 1, base 10^4: DKA in RFA's estimate form, and RFA with DKA's null slot
    # (innovation form), which keeps the flat prior on past states (evidence sum a, not sum a E^2)
    "D3.b1.r1e4.est": dict(_DKA, **_SC, damping=1.0, target_ss_process_var=0.5, dka_temperature=True,
                           dka_output="estimate"),
    "M3.b1.r1e4.null": dict(_SC, damping=1.0, zero_frac=0.0, target_ss_process_var=0.5, rfa_output="innovation",
                            rfa_prior="tokens"),
    "D3.b1.r1e4.N": dict(_DKA, **_SC, damping=1.0, target_ss_process_var=0.5, dka_temperature=True,
                         dka_prior="none"),                                 # without the null slot
    "D3.b05.r1e4.N": dict(_DKA, **_SC, damping=0.5, target_ss_process_var=0.5, dka_temperature=True,
                          dka_prior="none"),                                # without the null slot
    # Fair controls at the same spectrum: SC-RFA (no zero-decay heads) and RoPE with the same base
    # Slowest-horizon sweep at b = 0.2 and 0.5, with the temperature: a half and a quarter of D3's 625 tokens
    # (D3.b05 at 625 does not extrapolate: its phases turn ~1/b rad per horizon over lags never seen in training)
    "D3.b02.h312": dict(_DKA, **_SC, damping=0.2, rope_base=sc_base(0.2, 312.5), target_ss_process_var=0.5,
                      dka_temperature=True),
    "D3.b02.h156": dict(_DKA, **_SC, damping=0.2, rope_base=sc_base(0.2, 156.25), target_ss_process_var=0.5,
                      dka_temperature=True),
    "D3.b05.h312": dict(_DKA, **_SC, damping=0.5, rope_base=sc_base(0.5, 312.5), target_ss_process_var=0.5,
                      dka_temperature=True),
    "D3.b05.h156": dict(_DKA, **_SC, damping=0.5, rope_base=sc_base(0.5, 156.25), target_ss_process_var=0.5,
                      dka_temperature=True),
    "M3.b02": dict(_SC, damping=0.2, rope_base=_BASE_B02, zero_frac=0.0, target_ss_process_var=0.5),
    "M3.b05": dict(_SC, damping=0.5, rope_base=_BASE_B05, zero_frac=0.0, target_ss_process_var=0.5),
    "B1.b02": dict(use_rope=True, use_alibi=False, rope_base=_BASE_B02),
    "B1.b05": dict(use_rope=True, use_alibi=False, rope_base=_BASE_B05),
    # B5 (and B10, for the 2x2 ablation) at the new spectra, and B5 with the slopes 2 mu_h scaled by 0.5 and 2:
    # is DKA's predicted slope (log h = -2 mu dt) the best one?
    "B5.b02": dict(use_rope=False, use_alibi=False, use_sc_linear_bias=True, use_SC_RoPE=True,
                  damping=0.2, rope_base=_BASE_B02),
    "B5.b02.x05": dict(use_rope=False, use_alibi=False, use_sc_linear_bias=True, use_SC_RoPE=True,
                  damping=0.2, rope_base=_BASE_B02, bias_slope_scale=0.5),
    "B5.b02.x2": dict(use_rope=False, use_alibi=False, use_sc_linear_bias=True, use_SC_RoPE=True,
                  damping=0.2, rope_base=_BASE_B02, bias_slope_scale=2.0),
    "B5.b05": dict(use_rope=False, use_alibi=False, use_sc_linear_bias=True, use_SC_RoPE=True,
                  damping=0.5, rope_base=_BASE_B05),
    "B5.b05.x05": dict(use_rope=False, use_alibi=False, use_sc_linear_bias=True, use_SC_RoPE=True,
                  damping=0.5, rope_base=_BASE_B05, bias_slope_scale=0.5),
    "B5.b05.x2": dict(use_rope=False, use_alibi=False, use_sc_linear_bias=True, use_SC_RoPE=True,
                  damping=0.5, rope_base=_BASE_B05, bias_slope_scale=2.0),
    "B10.b02": dict(use_rope=False, use_alibi=False, use_sc_linear_bias=True, use_SC_RoPE=True,
                  damping=0.2, rope_base=_BASE_B02, use_softmax_one=True),
    "B10.b05": dict(use_rope=False, use_alibi=False, use_sc_linear_bias=True, use_SC_RoPE=True,
                  damping=0.5, rope_base=_BASE_B05, use_softmax_one=True),
}


def make_args(preset, **overrides):
    """Namespace of BASE_ARGS, updated with PRESETS[preset], then with `overrides`."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    d = dict(BASE_ARGS)
    d.update(PRESETS[preset])
    d.update(overrides)
    d["preset"] = preset
    return Namespace(**d)


def is_baseline(args):
    return args.preset.startswith("B")


def backbone_config(args):
    """Typed BackboneConfig of the model described by `args` (paper layout: LayerNorm, FFN width 4 d)."""
    from dka.model import BackboneConfig
    from dka.model.compat import attention_config_from_args, rfa_config_from_args, rfa_backbone_config
    if is_baseline(args):
        attention = attention_config_from_args(args, args.d_e, args.n_heads, 2 * args.d_e)
        return BackboneConfig(attention=attention, n_blocks=args.num_blocks, ffn_dim=4 * args.d_e, norm="layer",
                              io_projections=args.io_projections)
    attention = rfa_config_from_args(args, args.n_heads, args.d_e,
                                     query_key_dim_total=args.d_k_total, value_dim_total=args.d_v_total)
    return rfa_backbone_config(args, attention, args.num_blocks, 4 * args.d_v_total, nn.LayerNorm, args.io_projections)


def build_backbone(args):
    from dka.model import Backbone
    return Backbone(backbone_config(args), grad_checkpointing=args.grad_checkpointing)


def build_language_model(args, embed_init_std=None):
    """embed_init_std=None is the published runs' N(0, 1) embedding init; 0.02 is the usual choice."""
    from dka.model import LanguageModel
    return LanguageModel(build_backbone(args), vocab_size=args.vocab_size, embed_dim=args.d_e,
                         embed_init_std=embed_init_std)
