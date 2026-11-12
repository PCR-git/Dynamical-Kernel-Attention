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
    rope_base: float = 10000.0       # base of the RoPE spectrum the frequencies are taken from
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
    # "robust": w from robust_weight (scale mixture). "mixture": w = P(relevant | key) in a contamination model whose
    # background is the stationary distribution of a key, N(0, Sigma_inf), Sigma_inf = v_inf + eta^2 (Gaussian
    # components): w = sigmoid(lambda + beta_h), lambda = log(Sigma_inf / Sigma_n) + (|K|^2 / Sigma_inf - R^2 / Sigma_n) / d,
    # beta_h = logit pi_h learned. Far keys (E -> 0) get w -> pi_h whatever their content. temperature=True multiplies
    # the content term (|K|^2 / Sigma_inf - R^2 / Sigma_n) by tau_h.
    weight: Literal["robust", "mixture"] = "robust"
    init_relevance_logit: float = 0.0  # initial beta_h (weight="mixture")

    # --- Ablations ---
    dle_bias: bool = True            # Additive log-precision bias log(P_dt); False = no bias
    dle_gate: bool = True            # Multiplicative precision in the Mahalanobis distance; False = 1/sqrt(d)
    full_residual_norm: bool = True  # False = use only the cross term (dot-product attention)
    rotate_values: bool = True       # False = aggregate values without rotation or counter-rotation (M2.4)

    # --- Output: "innovation" = W_o (V_bar - V), as in the paper's Algorithm 1 (comparable with DKA);
    # "estimate" = W_o V_bar, as in the paper's experiments and all checkpoints trained before DKA ---
    output: Literal["innovation", "estimate"] = "innovation"
    # --- Prior on the current state: "flat" = maximum likelihood (the paper's RFA); "tokens" = DKA's proximal prior,
    # a null slot gamma^-2 worth c_h agreeing lag-0 tokens, giving dV = (M - S V) / (gamma^-2 + S) with S = sum a ---
    prior: Literal["flat", "tokens"] = "flat"
    init_prior_weight: float = 1.0
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
            rope_base=getattr(args, "rope_base", 10000.0),
            learn_rotations=args.learn_rotations,
            process_noise="steady_state" if args.use_ss_process_noise else "variance",
            init_steady_state_var=args.target_ss_process_var,
            init_measurement_var=args.target_measurement_var,
            robust_weight=args.use_robust_weight,
            weight=getattr(args, "dka_weight", "robust"),
            init_relevance_logit=getattr(args, "dka_init_relevance_logit", 0.0),
            dle_bias=args.additive_bias_type == 1,
            dle_gate=args.multiplicative_bias_type == 1,
            full_residual_norm=bool(args.use_full_residual_norm),
            rotate_values=args.rotate_values,
            output=getattr(args, "rfa_output", "estimate"),
            prior=getattr(args, "rfa_prior", "flat"),
            init_prior_weight=getattr(args, "rfa_init_prior_weight", 1.0),
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
    rope_base: float = 10000.0
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
    # "robust": w from robust_weight (scale mixture). "mixture": w = P(relevant | key) in a contamination model whose
    # background is the stationary distribution of a key, N(0, Sigma_inf), Sigma_inf = v_inf + eta^2 (Gaussian
    # components): w = sigmoid(lambda + beta_h), lambda = log(Sigma_inf / Sigma_n) + (|K|^2 / Sigma_inf - R^2 / Sigma_n) / d,
    # beta_h = logit pi_h learned. Far keys (E -> 0) get w -> pi_h whatever their content. temperature=True multiplies
    # the content term (|K|^2 / Sigma_inf - R^2 / Sigma_n) by tau_h.
    weight: Literal["robust", "mixture"] = "robust"
    init_relevance_logit: float = 0.0  # initial beta_h (weight="mixture")

    # --- Proximal prior on the current estimate (the null slot) ---
    # "tokens": gamma^-2 = c_h I(0) = c_h / eta^2, the prior is worth c_h agreeing tokens at lag 0, so a single
    #           agreeing token takes the step 1 / (1 + c_h) in every head, whatever its noise and decay
    # "window": gamma^-2 = c_h S_bar_h, S_bar_h the information of a whole context of agreeing tokens (Appendix G);
    #           slow heads then need many agreeing tokens to update (kept for comparison)
    # "direct": learn gamma^2
    # "none":   no null slot (gamma^-2 = 0): every step is a full step to the context's estimate. Ablation that keeps the
    #           stationary measurement model (h = E^2); with inflation zeta -> 1 as well it is RFA (innovation output)
    prior: Literal["tokens", "window", "direct", "none"] = "tokens"
    init_prior_weight: float = 1.0   # initial c_h ("tokens", "window"); c_h = 1 with "tokens" is softmax-plus-one-like
    calibration_length: int = 512    # prior="window": context length over which S_bar is summed (the training length)
    init_prior_var: float = 1.0      # initial gamma^2 for prior="direct"

    # --- Variance inflation (Appendix F): None = DKA's stationary prior; zeta in [0, 1) = initial inflation ---
    inflation: Optional[float] = None
    learn_inflation: bool = True

    # --- Evidence saturation (R12): None = tokens are independent measurements, S_i = sum_j a_ij h_ij ---
    # A float n_sat > 0 models a measurement error shared by all tokens (equicorrelation rho ~ 1 / n_sat), so the
    # information saturates, S_eff = S / (1 + S / S_max) with S_max = n_sat / eta^2 (n_sat agreeing tokens at lag 0).
    # The estimate M / S is unchanged; the step tends to n_sat / (c_h + n_sat) instead of 1 as evidence piles up.
    saturation: Optional[float] = None
    learn_saturation: bool = True

    # --- Inverse temperature: False = none; True = a learned tau_h > 0 per head (initialized at 1) multiplies the
    # content term of the logits, -kappa log(1 + R^2 / (nu Sigma)), i.e. frees the exponent kappa from nu (a Pearson
    # type VII likelihood). -log Sigma and the null slot are not tempered, so S stays the information, a <= 1/Sigma,
    # and one agreeing token still takes the step 1 / (1 + c_h).
    temperature: bool = False

    # --- Output: "innovation" = W_o dV (the paper); "estimate" = W_o (V + dV), the new estimate itself, as in RFA's
    # published runs (an abstaining head then passes its own value through) ---
    output: Literal["innovation", "estimate"] = "innovation"

    # --- Ablations (as RFA's M2.4 / M2.5) and QK normalization ---
    rotate_values: bool = True       # False: aggregate the values in their own frames (no Phi^- / Phi^+ on V)
    zero_rotations: bool = False     # True: omega = 0 (decay only; mu keeps its spectral initialization)
    qk_norm: bool = False            # True: Q and K normalized per head to a learned length s_h (shared by Q and K)
    # p-RoPE (Barbero et al.): in heads whose horizon 1/mu exceeds this length (the training length), zero the
    # frequencies too low to complete a cycle within it (period 2 pi / omega > length): position-free channels
    prope_length: Optional[float] = None
    # Decay ceiling: mu_h = min(mu_h, decay_cap). With decay_cap = c / L the fast SC heads keep their high frequencies
    # alive over the training length L (E(L) = e^-c) instead of dying within a few tokens; slower heads are unchanged
    decay_cap: Optional[float] = None
    init_qk_scale_sq: float = 4.0    # initial s_h^2 / head_dim
    # Content residual: "euclidean" = ||K~ - E Q~||^2 (the paper); "polar" = the polar model's directional residual
    # E^2 |K|^2 2 (1 - cos(Q~, K~)): the key against the query direction, lag-invariant match (with qk_norm, |K| = s_h
    # and the weight stays <= 1 / Sigma_n, so the eviction bound holds); "dot" = the cross term -2 E Re<Q~, K~> alone
    # (the |K|^2 and E^2 |Q|^2 terms dropped: dot-product content, needs the exponential weight); "dot_noE" = the
    # same without the attenuation, -2 Re<Q~, K~>: the content term keeps its strength at every lag
    # "polar_obs" = the polar appendix with the test and the information separated: d^2 = 2 kappa^obs (1 - cos), with
    # kappa^obs = m_j^2 / Sigma_{n,t} in the key's own units (no E), information E^2 kappa^obs, null c_h kappa_ii
    residual: Literal["euclidean", "polar", "dot", "dot_noE", "polar_obs"] = "euclidean"
    # polar_obs forcing in Sigma_{n,t}: "constant" v_inf (1 - E^2) as in DKA, or "magnitude" m_j^2 (1 - E^2) / 2d
    polar_forcing: Literal["constant", "magnitude"] = "constant"
    init_rho_sq: float = 1e-3        # polar_obs: initial rho^2 (measurement noise rho^2 m^2 next to eta^2)
    # Flat precision (as RFA's flat-prior ablation): Sigma_n = eta^2 at every lag, no accumulated forcing, so the
    # -log Sigma_n bias is constant and the residual is scaled the same at every lag; decay still enters through E
    flat_precision: bool = False
    # Zero decay: mu = 0 in every head (rotation only, E = h = 1), keeping the forcing sigma^2 = 2 mu_h v_inf of each
    # head's coupled decay, so Sigma_n = eta^2 + sigma^2 dt grows linearly with lag (a random walk, not stationary)
    zero_decay: bool = False
    # Plain values: each key's estimate is U v_j itself instead of the unbiased U v_j / E (weight A E^2 on U v_j
    # instead of A E), so the weights on the values sum to at most one, as in softmax attention (the polar value path)
    plain_values: bool = False
    # Key evidence: each key's measurement noise takes the previous layer's evidence share s_j = S_j / (gamma^-2 +
    # S_j) of the head with the same index (same horizon), eta_j^2 = eta^2 / (1 + a_h s_j), a_h = e^{alpha_h} - 1
    # learned from alpha_h = 0 (exactly DKA at the start; a_h > -1 keeps the noise positive). First layer: s = 0
    key_evidence: bool = False
    # Test scale: "sigma_n" tests the residual against its full variance Sigma_n (the paper); "measurement" tests it
    # against the measurement noise eta^2 alone, w(R^2 / (nu eta^2)), while the influence keeps 1/Sigma_n; "split"
    # tests it against Sigma_test = v_inf,test (1 - E^2) + eta^2 with its own forcing, initialized equal to the
    # influence's (exactly DKA at the start; v_inf,test = 0 is "measurement"). A selective (redescending) weight with a
    # test scale that grows with lag favors distant keys with large residuals
    test_scale: Literal["sigma_n", "measurement", "split"] = "sigma_n"
    # Learned decay: mu_h = omega_max,h (b_floor,h + softplus(beta_h / 10)), starting at the coupled mu. Heads whose
    # slowest period fits in calibration_length (every phase seen in training) may only keep horizon <= that length;
    # the others may only decay faster than the coupling (b >= its initial value), since slower decay helps in the
    # window at a cost beyond it that training cannot see. The /10 slows beta's steps under Adam
    learn_decay: bool = False

    # --- Numerics ---
    epsilon: float = 1e-5

    def __post_init__(self):
        if self.saturation is not None and not self.saturation > 0:
            raise ValueError(f"saturation must be positive, got {self.saturation}")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even (conjugate frequency pairs), got {self.head_dim}")
        if self.weight == "mixture" and (self.inflation is not None or self.zero_decay_frac > 0):
            raise ValueError("weight='mixture' needs the stationary prior (no inflation) and mu > 0 in every head")
        if self.inflation is not None and not 0.0 <= self.inflation < 1.0:
            raise ValueError(f"inflation must be in [0, 1), got {self.inflation}")
        if self.residual == "polar" and (self.inflation is not None or self.weight != "robust"):
            raise ValueError("residual='polar' needs h = E^2 (no inflation) and weight='robust'")
        if self.learn_decay and (self.zero_decay or self.zero_rotations or self.decay_init != "spectral"):
            raise ValueError("learn_decay needs the coupled decay (no zero_decay, zero_rotations or log-linear init)")
        if self.test_scale != "sigma_n" and (self.weight != "robust" or self.residual in ("polar", "polar_obs")
                                             or self.inflation is not None or self.flat_precision):
            raise ValueError(f"test_scale={self.test_scale!r} needs weight='robust', a Euclidean or dot-product "
                             "residual, no inflation and no flat precision")
        if self.key_evidence and (self.residual == "polar_obs" or self.inflation is not None):
            raise ValueError("key_evidence needs Sigma_n = forcing + eta^2 (no polar_obs, no inflation)")
        if self.residual == "polar_obs" and (self.inflation is not None or self.weight != "robust"
                                             or self.prior not in ("tokens", "none")):
            raise ValueError("residual='polar_obs' needs no inflation, weight='robust' and prior 'tokens' or 'none'")
        if self.residual in ("dot", "dot_noE") and (self.inflation is not None or self.weight != "robust" or self.robust_weight):
            raise ValueError(f"residual={self.residual!r} needs no inflation and the exponential weight (robust_weight=False)")
        if self.flat_precision and (self.inflation is not None or self.weight != "robust"):
            raise ValueError("flat_precision needs no inflation and weight='robust'")

    @classmethod
    def from_args(cls, args, d_model, n_heads, head_dim):
        """
        Build a config from the argparse-style `args` of the presets and notebooks. The dynamics and noise
        flags are shared with RFAConfig.from_args; DKA's own are dka_prior, dka_init_prior_weight,
        dka_init_prior_var, dka_inflation, dka_learn_inflation, dka_saturation and dka_learn_saturation. The S_bar calibration ("window") uses
        the training length args.seq_len.
        """
        for name, default in _RFA_ONLY_ARGS.items():
            if getattr(args, name, default) != default:
                raise ValueError(f"args.{name}={getattr(args, name)!r} is an RFA ablation with no DKA counterpart.")
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
            rope_base=getattr(args, "rope_base", 10000.0),
            learn_rotations=args.learn_rotations,
            process_noise="steady_state" if args.use_ss_process_noise else "variance",
            init_steady_state_var=args.target_ss_process_var,
            init_measurement_var=args.target_measurement_var,
            robust_weight=args.use_robust_weight,
            weight=getattr(args, "dka_weight", "robust"),
            init_relevance_logit=getattr(args, "dka_init_relevance_logit", 0.0),
            prior=getattr(args, "dka_prior", "tokens"),
            init_prior_weight=getattr(args, "dka_init_prior_weight", 1.0),
            calibration_length=args.seq_len,
            init_prior_var=getattr(args, "dka_init_prior_var", 1.0),
            inflation=getattr(args, "dka_inflation", None),
            learn_inflation=getattr(args, "dka_learn_inflation", True),
            saturation=getattr(args, "dka_saturation", None),
            learn_saturation=getattr(args, "dka_learn_saturation", True),
            temperature=getattr(args, "dka_temperature", False),
            output=getattr(args, "dka_output", "innovation"),
            rotate_values=getattr(args, "rotate_values", True),
            zero_rotations=getattr(args, "zero_rotations", False),
            qk_norm=getattr(args, "dka_qk_norm", False),
            prope_length=getattr(args, "prope_length", None),
            decay_cap=getattr(args, "dka_decay_cap", None),
            residual=getattr(args, "dka_residual", "euclidean"),
            polar_forcing=getattr(args, "dka_polar_forcing", "constant"),
            flat_precision=getattr(args, "dka_flat_precision", False),
            zero_decay=getattr(args, "dka_zero_decay", False),
            plain_values=getattr(args, "dka_plain_values", False),
            key_evidence=getattr(args, "dka_key_evidence", False),
            test_scale=getattr(args, "dka_test_scale", "sigma_n"),
            learn_decay=getattr(args, "dka_learn_decay", False),
            epsilon=args.epsilon,
        )


# RFA ablation flags (args name -> RFA default) that have no DKA counterpart
_RFA_ONLY_ARGS = {"additive_bias_type": 1, "multiplicative_bias_type": 1, "use_full_residual_norm": 1,
                  "lambda_real_zero": False,
                  "zero_process_noise": False, "zero_key_measurement_noise": False}
