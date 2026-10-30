from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class RFAConfig:
    """
    Configuration of a multi-head Isotropic RFA layer.

    Shapes: d_model is the real input/output width. Each head works in a complex
    space of dimension head_dim (stored as head_dim real/imaginary pairs), i.e. the
    projections map d_model -> 2 * n_heads * head_dim.
    """

    d_model: int
    n_heads: int
    head_dim: int

    # --- Dynamics: eigenvalues lambda_k = -mu + i*omega_k (per head) ---
    frequency_init: Literal["rope", "spectral"] = "rope"  # "spectral" = SC-RFA: spectrum partitioned across heads
    decay_init: Literal["spectral", "log_linear"] = "spectral"  # "spectral": mu_h = damping * omega_h,max
    damping: float = 0.05            # SC-RFA damping coefficient b
    zero_decay_frac: float = 0.25    # Fraction of (lowest-frequency) heads with mu = 0
    decay_scale: float = 1.0         # Multiplies the initial decay rates
    omega_rescale: float = 1.0       # Multiplies all rotation frequencies
    learn_rotations: bool = False

    # --- Noise model ---
    # "variance":     learn the process noise variance sigma^2 (paper experiments)
    # "steady_state": learn the steady-state variance sigma^2 / (2 mu) (better conditioned)
    process_noise: Literal["variance", "steady_state"] = "variance"
    init_steady_state_var: float = 0.1
    init_measurement_var: float = 1.0
    init_nu_over_d: float = 4.0

    # --- Attention weights ---
    robust_weight: bool = True       # Student-t (power-law) weight; False = exponential weight

    # --- Ablations ---
    dle_bias: bool = True            # Additive log-precision bias log(P_dt); False = no bias
    dle_gate: bool = True            # Multiplicative precision in the Mahalanobis distance; False = 1/sqrt(d)
    full_residual_norm: bool = True  # False = use only the cross term (dot-product attention)
    rotate_values: bool = True       # False = aggregate values without rotation or counter-rotation (M2.4)

    # --- Output: "innovation" = W_o (V_bar - V), as in the paper's Algorithm 1 (comparable with DKA);
    # "estimate" = W_o V_bar, as in the paper's experiments and all checkpoints trained before DKA ---
    output: Literal["innovation", "estimate"] = "innovation"
    zero_rotations: bool = False
    zero_decay: bool = False
    zero_process_noise: bool = False
    zero_key_measurement_noise: bool = False

    # --- Numerics ---
    epsilon: float = 1e-5
    cov_epsilon: float = 1e-2
    min_exponent: float = -10.0
    max_exponent: float = 0.0

    def __post_init__(self):
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even (conjugate frequency pairs), got {self.head_dim}")

    @classmethod
    def from_args(cls, args, d_model, n_heads, head_dim):
        """Build a config from the argparse-style `args` used by the training notebooks."""
        removed = {
            "learn_t_shift": True, "learn_decay": True, "sep_params": True, "use_colored_prior": True,
            "use_inner_residual": True, "add_gaussian_noise": True, "use_complex_conj_constraint": False,
            "compute_pulled_forward_estimates": True,
        }
        for name, bad in removed.items():
            if getattr(args, name, not bad) == bad:
                raise ValueError(f"args.{name}={bad} is no longer supported (feature removed in the refactor).")
        if getattr(args, "use_complex_input_norm", 0) or getattr(args, "use_complex_output_norm", False):
            raise ValueError("args.use_complex_input_norm / use_complex_output_norm are no longer supported (feature removed).")
        if getattr(args, "additive_bias_type", 1) not in (0, 1) or getattr(args, "multiplicative_bias_type", 1) not in (0, 1):
            raise ValueError("Only additive/multiplicative_bias_type 0 (off) or 1 (DLE) are supported.")

        if getattr(args, "scale_decay_by_time_interval", False):
            decay_scale = args.max_fixed_decay / (args.seq_len - 1)
        else:
            decay_scale = 1.0

        use_sc = args.use_SC_RoPE
        return cls(
            d_model=d_model, n_heads=n_heads, head_dim=head_dim,
            frequency_init="spectral" if use_sc else "rope",
            decay_init="log_linear" if (args.use_log_linear_decay and not use_sc) else "spectral",
            damping=args.damping,
            zero_decay_frac=args.zero_frac,
            decay_scale=decay_scale,
            omega_rescale=args.omega_rescale,
            learn_rotations=args.learn_rotations,
            process_noise="steady_state" if args.use_ss_process_noise else "variance",
            init_steady_state_var=args.target_ss_process_var,
            init_measurement_var=args.target_measurement_var,
            robust_weight=args.use_robust_weight,
            dle_bias=args.additive_bias_type == 1,
            dle_gate=args.multiplicative_bias_type == 1,
            full_residual_norm=bool(args.use_full_residual_norm),
            rotate_values=args.rotate_values,
            output=getattr(args, "rfa_output", "estimate"),
            zero_rotations=args.zero_rotations,
            zero_decay=args.lambda_real_zero,
            zero_process_noise=args.zero_process_noise,
            zero_key_measurement_noise=args.zero_key_measurement_noise,
            epsilon=args.epsilon,
            min_exponent=args.min_exponent,
            max_exponent=args.max_exponent,
        )


@dataclass(frozen=True)
class DKAConfig:
    """
    Configuration of a multi-head Isotropic DKA layer (Dynamical_Kernel_Attention.tex, Algorithm 1).

    Shapes as in RFAConfig: each head works in a complex space of dimension head_dim, so the projections
    map d_model -> 2 * n_heads * head_dim.
    """

    d_model: int
    n_heads: int
    head_dim: int

    # --- Dynamics: eigenvalues lambda_k = -mu + i*omega_k (per head); as in RFAConfig ---
    frequency_init: Literal["rope", "spectral"] = "spectral"  # "spectral" = SC-DKA
    decay_init: Literal["spectral", "log_linear"] = "spectral"  # "spectral": mu_h = damping * omega_h,max
    damping: float = 0.05
    zero_decay_frac: float = 0.0     # DKA uses slow heads rather than mu = 0 (the mu = 0 path is supported)
    decay_scale: float = 1.0
    omega_rescale: float = 1.0
    learn_rotations: bool = False

    # --- Noise model ---
    # "variance":     learn the forcing per unit time sigma^2 (v_inf = sigma^2 / (2 mu))
    # "steady_state": learn the stationary variance v_inf (sigma^2 for mu = 0 heads), as in Algorithm 1
    process_noise: Literal["variance", "steady_state"] = "variance"
    init_steady_state_var: float = 0.1   # initial v_inf (in "variance" mode sigma^2 = 2 mu v_inf, equal rho across heads)
    init_measurement_var: float = 1.0
    init_nu_over_d: float = 4.0

    # --- Attention weights ---
    robust_weight: bool = True       # Student-t (power-law) weight; False = exponential weight

    # --- Proximal prior on the current estimate (the null slot) ---
    # "tokens": gamma^-2 = c_h I(0) = c_h / eta^2, the prior is worth c_h agreeing tokens at lag 0, so a single
    #           agreeing token takes the step 1 / (1 + c_h) in every head, whatever its noise and decay
    # "window": gamma^-2 = c_h S_bar_h, S_bar_h the information of a whole context of agreeing tokens (Appendix G);
    #           slow heads then need many agreeing tokens to update (kept for comparison)
    # "direct": learn gamma^2
    prior: Literal["tokens", "window", "direct"] = "tokens"
    init_prior_weight: float = 1.0   # initial c_h ("tokens", "window"); c_h = 1 with "tokens" is softmax-plus-one-like
    calibration_length: int = 512    # prior="window": context length over which S_bar is summed (the training length)
    init_prior_var: float = 1.0      # initial gamma^2 for prior="direct"

    # --- Variance inflation (Appendix F): None = DKA's stationary prior; zeta in [0, 1) = initial inflation ---
    inflation: Optional[float] = None
    learn_inflation: bool = True

    # --- Numerics ---
    epsilon: float = 1e-5

    def __post_init__(self):
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even (conjugate frequency pairs), got {self.head_dim}")
        if self.inflation is not None and not 0.0 <= self.inflation < 1.0:
            raise ValueError(f"inflation must be in [0, 1), got {self.inflation}")
