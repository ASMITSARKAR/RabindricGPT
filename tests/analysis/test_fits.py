"""
tests/analysis/test_fits.py : Tests for power-law scaling curve fitting and parameter recovery.
"""

import pytest

from src.rgpt.analysis.fits import fit_power_law


def test_power_law_parameter_recovery() -> None:
    """Fitting a synthetic power-law dataset must recover true exponent within tolerance."""
    true_alpha = 0.35
    true_beta = 5.0

    sizes = [1e6, 5e6, 25e6]
    # y = beta * x^(-alpha)
    losses = [true_beta * (s ** (-true_alpha)) for s in sizes]

    res = fit_power_law(sizes, losses)

    assert pytest.approx(res.alpha, rel=1e-3) == true_alpha
    assert pytest.approx(res.beta, rel=1e-3) == true_beta
    assert res.r_squared > 0.99
    assert not res.is_weak


def test_power_law_3p_fit() -> None:
    """Fitting 4 or more points triggers 3-parameter fit."""
    true_alpha = 0.40
    true_a = 8.0
    true_linf = 0.80

    sizes = [1e6, 5e6, 25e6, 100e6]
    losses = [true_a * (s ** (-true_alpha)) + true_linf for s in sizes]

    res = fit_power_law(sizes, losses)
    assert res.r_squared > 0.95
