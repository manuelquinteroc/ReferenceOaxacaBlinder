"""The nonlinear Oaxaca-Blinder decomposition with both reference groups.

Python port of ``decompose_mean`` in ``census/hlp_nonlinear.R`` (lines 126-149).

For each group ``g in {0, 1}`` a model ``fit_g`` is trained on that group's rows. With

    mu(a, b) = mean prediction of the group-``b`` model on the group-``a`` covariates,

the decomposition returns, for both reference choices,

    delta_y        = mean(y[pop==1]) - mean(y[pop==0])
    explained_1    = mu(1, 1) - mu(0, 1)      # reference = group 1
    unexplained_1  = mu(0, 1) - mu(0, 0)
    explained_0    = mu(1, 0) - mu(0, 0)      # reference = group 0
    unexplained_0  = mu(1, 1) - mu(1, 0)

A *sign flip* occurs when ``explained_0 * explained_1 < 0`` (or the same for the
unexplained components). This generalizes the linear two-fold decomposition used in the
ICU notebook (which is the OLS special case).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .builders import get_builder, outcome_type, predict_mean
from .features import PreparedData, prepare_features

_NAN_ROW = {
    "delta_y": np.nan,
    "explained_1": np.nan, "unexplained_1": np.nan,
    "explained_0": np.nan, "unexplained_0": np.nan,
    "n_1": 0, "n_0": 0,
}


def decompose_mean(prepped: PreparedData, algo_name: str) -> dict:
    """Run the decomposition on an already-prepared subset."""
    pop = np.asarray(prepped.pop)
    y = np.asarray(prepped.y, dtype=float)
    X = prepped.features

    idx0 = np.where(pop == 0)[0]
    idx1 = np.where(pop == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return dict(_NAN_ROW)

    builder = get_builder(algo_name, outcome_type(y))

    X0, y0 = X.iloc[idx0], y[idx0]
    X1, y1 = X.iloc[idx1], y[idx1]

    fit_0 = builder().fit(X0, y0)
    fit_1 = builder().fit(X1, y1)

    # mu(a, b): group-b model evaluated on group-a covariates.
    mu_00 = predict_mean(fit_0, X0)
    mu_10 = predict_mean(fit_0, X1)
    mu_01 = predict_mean(fit_1, X0)
    mu_11 = predict_mean(fit_1, X1)

    return {
        "delta_y": float(np.mean(y1) - np.mean(y0)),
        "explained_1": mu_11 - mu_01,
        "unexplained_1": mu_01 - mu_00,
        "explained_0": mu_10 - mu_00,
        "unexplained_0": mu_11 - mu_10,
        "n_1": int(len(idx1)),
        "n_0": int(len(idx0)),
    }


def decompose_from_data(data: pd.DataFrame, y_name: str, pop_name: str,
                        algo_name: str) -> dict:
    """Restrict to non-missing group rows, prepare features, and decompose.

    Mirrors the per-call work in R ``twoway_subset`` (hlp_nonlinear.R:21-33): drop rows
    with missing group membership, build the design matrix, then decompose. The caller is
    responsible for having already restricted ``data`` to the subset of interest.
    """
    data = data[data[pop_name].notna()]
    if len(data) == 0:
        return dict(_NAN_ROW)
    prepped = prepare_features(data, pop_name, y_name)
    return decompose_mean(prepped, algo_name)
