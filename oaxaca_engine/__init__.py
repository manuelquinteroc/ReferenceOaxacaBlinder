"""Shared Oaxaca-Blinder decomposition engine.

Used by both the ICU analysis (``Real-data example/``) and the Python census pipeline
(``census/py/``). Public API:

    prepare_features          - build the numeric design matrix (one-hot + missing flags)
    outcome_type              - 'binary' vs 'continuous'
    get_builder, predict_mean - ICU model builders + uniform mean prediction
    decompose_mean            - decomposition on a prepared subset
    decompose_from_data       - restrict + prepare + decompose
    resample                  - within-group bootstrap resampling
    bootstrap_decomposition   - bootstrap one design cell
    pval, stars               - normal-approx inference helpers
"""

from .features import PreparedData, prepare_features, CATEGORICAL_COLS
from .builders import (
    outcome_type,
    get_builder,
    predict_mean,
    OLSModel,
    build_ols,
    build_logistic,
    build_nn_clf,
    build_nn_reg,
    build_xgb_clf,
    build_xgb_reg,
)
from .decomposition import decompose_mean, decompose_from_data
from .bootstrap import resample, bootstrap_decomposition
from .inference import pval, stars

__all__ = [
    "PreparedData", "prepare_features", "CATEGORICAL_COLS",
    "outcome_type", "get_builder", "predict_mean", "OLSModel",
    "build_ols", "build_logistic", "build_nn_clf", "build_nn_reg",
    "build_xgb_clf", "build_xgb_reg",
    "decompose_mean", "decompose_from_data",
    "resample", "bootstrap_decomposition",
    "pval", "stars",
]
