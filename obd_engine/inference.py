"""Normal-approximation inference helpers.

Ports R ``2 * pnorm(-abs(stat), sd = se)`` (census/03_count-flips.R:47) and the ICU
significance-star helper.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def pval(est: float, se: float) -> float:
    """Two-sided p-value under a normal approximation with bootstrap SE."""
    if est is None or se is None or np.isnan(est) or np.isnan(se) or se <= 0:
        return np.nan
    return float(2 * norm.cdf(-abs(est / se)))


def stars(p: float) -> str:
    """Significance stars at 1% / 5% / 10% (ICU notebook helper)."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""
