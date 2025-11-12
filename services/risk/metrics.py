"""
Lightweight risk metrics used when OS-Climate physrisk is unavailable.

Provides Value-at-Risk (VaR) and Expected Shortfall (ES) for a vector of
scenario/sample losses. Designed to be robust for small samples.
"""

from __future__ import annotations

import numpy as np


def value_at_risk(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Compute the alpha-quantile (VaR) of losses.

    Args:
        losses: 1D array of non-negative loss realizations.
        alpha: confidence level in (0,1).

    Returns:
        The VaR at level alpha.
    """
    if losses is None:
        return 0.0
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        return 0.0
    arr = np.clip(arr, 0.0, np.inf)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return float(np.min(arr))
    if alpha >= 1.0:
        return float(np.max(arr))
    return float(np.quantile(arr, alpha, method="nearest"))


def expected_shortfall(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Compute the Expected Shortfall (CVaR) at level alpha.

    ES is the conditional mean of losses beyond VaR(alpha).
    Uses a simple tail average when sample is small.
    """
    if losses is None:
        return 0.0
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        return 0.0
    arr = np.clip(arr, 0.0, np.inf)
    var = value_at_risk(arr, alpha)
    tail = arr[arr >= var]
    if tail.size == 0:
        return float(var)
    return float(np.mean(tail))

