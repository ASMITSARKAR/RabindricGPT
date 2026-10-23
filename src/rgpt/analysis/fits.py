"""
src/rgpt/analysis/fits.py : Power-law scaling fits for BPC/BPB versus non-embedding parameter count.
Backlog/Phase : I11 / Phase J      Docs : 15_BUILD_DIRECTIVE.md Phase J; 01_CHARTER.md; 08 §2
Contract      : sizes: list[float], losses: list[float] -> FitResult
Invariants    : 2-parameter power law when sizes < 4; 3-parameter only when sizes >= 4; labels weak fits
Failure modes : Fewer than 3 data points; non-convergent optimization
Status        : TESTED
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class FitResult:
    """Dataclass holding power-law scaling curve parameters and error bounds."""
    alpha: float
    beta: float  # Scale factor A
    r_squared: float
    formula: str
    is_weak: bool = False
    l_inf: Optional[float] = None


def fit_power_law(sizes: List[float], losses: List[float]) -> FitResult:
    """Fit power-law curve: Loss = A * N^(-alpha) or Loss = A * N^(-alpha) + L_inf.

    Args:
        sizes: Non-embedding parameter counts.
        losses: Measured validation BPC or BPB values.

    Returns:
        FitResult containing fitted exponents and goodness-of-fit R^2.
    """
    if len(sizes) < 2 or len(losses) < 2:
        raise ValueError("Fitting requires at least 2 data points.")
    if len(sizes) != len(losses):
        raise ValueError(f"Mismatch: {len(sizes)} sizes vs {len(losses)} losses.")

    x = np.array(sizes, dtype=np.float64)
    y = np.array(losses, dtype=np.float64)

    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Sizes and losses must be strictly positive.")

    # Always start with 2-parameter log-linear fit: ln(y) = ln(A) - alpha * ln(x)
    log_x = np.log(x)
    log_y = np.log(y)

    poly = np.polyfit(log_x, log_y, 1)
    alpha_2p = float(-poly[0])
    beta_2p = float(math.exp(poly[1]))

    # Predictions and R^2 for 2-parameter
    y_pred_2p = beta_2p * (x ** (-alpha_2p))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res_2p = float(np.sum((y - y_pred_2p) ** 2))
    r2_2p = 1.0 - (ss_res_2p / ss_tot) if ss_tot > 0 else 1.0

    # If < 4 points, use 2-parameter per contract
    if len(sizes) < 4:
        is_weak = r2_2p < 0.90
        formula = f"L(N) = {beta_2p:.4f} * N^(-{alpha_2p:.4f})"
        if is_weak:
            formula += " [WEAK FIT]"
        return FitResult(
            alpha=alpha_2p,
            beta=beta_2p,
            r_squared=r2_2p,
            formula=formula,
            is_weak=is_weak,
            l_inf=None,
        )

    # 3-parameter fit: L(N) = A * N^(-alpha) + L_inf
    try:
        from scipy.optimize import curve_fit

        def power_law_3p(n, a, alpha, l_inf):
            return a * (n ** (-alpha)) + l_inf

        p0 = [beta_2p, alpha_2p, float(np.min(y) * 0.5)]
        bounds = ([0.0, 0.0, 0.0], [np.inf, 5.0, np.min(y)])

        popt, _ = curve_fit(power_law_3p, x, y, p0=p0, bounds=bounds, maxfev=10000)
        a_3p, alpha_3p, l_inf_3p = popt

        y_pred_3p = power_law_3p(x, *popt)
        ss_res_3p = float(np.sum((y - y_pred_3p) ** 2))
        r2_3p = 1.0 - (ss_res_3p / ss_tot) if ss_tot > 0 else 1.0

        is_weak = r2_3p < 0.90
        formula = f"L(N) = {a_3p:.4f} * N^(-{alpha_3p:.4f}) + {l_inf_3p:.4f}"
        if is_weak:
            formula += " [WEAK FIT]"

        return FitResult(
            alpha=float(alpha_3p),
            beta=float(a_3p),
            r_squared=float(r2_3p),
            formula=formula,
            is_weak=is_weak,
            l_inf=float(l_inf_3p),
        )
    except Exception:
        # Fall back to 2-parameter if 3-parameter fails to converge
        is_weak = r2_2p < 0.90
        formula = f"L(N) = {beta_2p:.4f} * N^(-{alpha_2p:.4f})"
        if is_weak:
            formula += " [WEAK FIT]"
        return FitResult(
            alpha=alpha_2p,
            beta=beta_2p,
            r_squared=r2_2p,
            formula=formula,
            is_weak=is_weak,
            l_inf=None,
        )
