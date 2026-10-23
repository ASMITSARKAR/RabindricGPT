"""
src/rgpt/preflight/memory.py : Analytic GPU VRAM memory estimator for model parameters, AdamW states, and activations.
Backlog/Phase : Phase P      Docs : 15_BUILD_DIRECTIVE.md Phase P (Level P6); DR-14
Contract      : estimate_vram_usage_gb(model_config, batch_size, s_max, precision) -> dict
Invariants    : Microbatch allocation must fit target GPU with <=80% memory ceiling
Failure modes : Underestimated activation memory causing GPU OOM in production
Status        : TESTED
"""

from typing import Dict

from src.rgpt.model.config import ModelConfig
from src.rgpt.model.params import count_parameters_analytic


def estimate_vram_usage_gb(
    config: ModelConfig,
    batch_size: int,
    s_max: int,
    precision: str = "bf16",
    optimizer: str = "adamw",
) -> Dict[str, float]:
    """Analytically estimate peak VRAM consumption in gigabytes.

    Components estimated:
        1. Model parameters (fp16/bf16: 2 bytes/param)
        2. Optimizer states (AdamW: fp32 master weights 4B + momentum 4B + variance 4B = 12 bytes/param)
        3. Gradient buffers (2 bytes/param)
        4. Activations per microbatch and s_max

    Returns:
        Dictionary of memory component breakdowns in GB.
    """
    total_params, _ = count_parameters_analytic(config)

    # 1. Parameter memory
    param_bytes_per_elem = 2 if precision in ("bf16", "fp16") else 4
    params_bytes = total_params * param_bytes_per_elem

    # 2. Optimizer memory
    opt_bytes = total_params * 12 if optimizer == "adamw" else total_params * 8

    # 3. Gradients
    grads_bytes = total_params * param_bytes_per_elem

    # 4. Activations (standard transformer backward activations per layer)
    # Approx 34 * b * s * d * L bytes in mixed precision
    b, s, d, n_layers = batch_size, s_max, config.d_model, config.n_layers
    activations_bytes = 34 * b * s * d * n_layers * param_bytes_per_elem

    gb = 1024.0 ** 3
    params_gb = params_bytes / gb
    opt_gb = opt_bytes / gb
    grads_gb = grads_bytes / gb
    act_gb = activations_bytes / gb

    total_gb = params_gb + opt_gb + grads_gb + act_gb

    # Check against L4 (24GB) and T4 (16GB) ceilings (<= 80%)
    fits_l4 = total_gb <= (24.0 * 0.80)
    fits_t4 = total_gb <= (16.0 * 0.80)

    return {
        "params_gb": round(params_gb, 4),
        "optimizer_gb": round(opt_gb, 4),
        "gradients_gb": round(grads_gb, 4),
        "activations_gb": round(act_gb, 4),
        "total_gb": round(total_gb, 4),
        "fits_l4_80pct": fits_l4,
        "fits_t4_80pct": fits_t4,
    }
