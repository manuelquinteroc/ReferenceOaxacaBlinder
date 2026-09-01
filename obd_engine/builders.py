"""Model builders and the algorithm registry.

These are the *same* estimators used by the ICU analysis in
``Real-data example/icu_139_final_models.ipynb`` (classification builders copied
verbatim), extended here with regression twins so that the continuous census outcome
(``pincp`` = log income) can be decomposed. Hyperparameters are fixed; there is no
cross-validation / early-stopping tuning (the R ``lgb.cv`` path is intentionally dropped).

Algorithm names follow the R census convention so that downstream output schemas match:

    ols  -> statsmodels OLS  (linear; a linear probability model when y is binary)
    glm  -> sklearn LogisticRegression   (binary outcomes only)
    gbt  -> XGBoost          (XGBClassifier / XGBRegressor)
    net  -> sklearn MLP      (MLPClassifier / MLPRegressor)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor


# --- outcome type -----------------------------------------------------------------

def outcome_type(y) -> str:
    """'binary' if every value is in {0, 1}, else 'continuous' (R hlp_nonlinear.R:206)."""
    arr = np.asarray(y, dtype=float)
    arr = arr[~np.isnan(arr)]
    return "binary" if np.isin(arr, (0.0, 1.0)).all() else "continuous"


# --- statsmodels OLS wrapper ------------------------------------------------------
# Wrapped so it presents a uniform .fit(X, y) / .predict(X) interface and bakes in the
# intercept. Has no predict_proba, so predict_mean treats it as a (linear) regressor —
# i.e. a linear probability model for binary outcomes, matching R's gaussian glm.fit.

class OLSModel:
    def __init__(self):
        self.results_ = None
        self.columns_ = None

    @staticmethod
    def _design(X) -> pd.DataFrame:
        X = pd.DataFrame(X).reset_index(drop=True)
        return sm.add_constant(X, has_constant="add")

    def fit(self, X, y):
        Xd = self._design(X)
        self.columns_ = list(Xd.columns)
        self.results_ = sm.OLS(np.asarray(y, dtype=float), Xd, missing="drop").fit()
        return self

    def predict(self, X):
        Xd = self._design(X).reindex(columns=self.columns_, fill_value=0.0)
        return np.asarray(self.results_.predict(Xd), dtype=float)

    @property
    def intercept(self) -> float:
        """Intercept term, used by the aligned-slope diagnostic (step 04)."""
        return float(self.results_.params["const"])


def build_ols() -> OLSModel:
    return OLSModel()


# --- classification builders (copied from the ICU notebook) -----------------------

class LogisticModel:
    """LogisticRegression that degrades gracefully on a single-class group.

    R's ``glm(..., family = binomial)`` still returns a fit (with a warning) when the
    outcome is constant within a group, predicting that constant everywhere. sklearn's
    solver raises instead, so mirror R: if only one class is present, predict its value.
    """

    def __init__(self):
        self.pipe_ = Pipeline([("scaler", StandardScaler()),
                               ("clf", LogisticRegression(max_iter=2000, random_state=0))])
        self.const_ = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if np.unique(y).size < 2:
            self.const_ = float(y[0]) if y.size else np.nan
        else:
            self.pipe_.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.const_ is not None:
            p = np.full(len(X), self.const_, dtype=float)
            return np.column_stack([1.0 - p, p])
        return self.pipe_.predict_proba(X)


def build_logistic() -> LogisticModel:
    return LogisticModel()


def build_nn_clf() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=(32, 16),
                                           activation="relu", alpha=1e-4,
                                           max_iter=2000, solver="adam",
                                           random_state=0))])


def build_xgb_clf() -> XGBClassifier:
    return XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                         eval_metric="logloss", random_state=0, n_jobs=1,
                         tree_method="hist", verbosity=0)


# --- regression twins (same architecture / hyperparameters) -----------------------

def build_nn_reg() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()),
                     ("reg", MLPRegressor(hidden_layer_sizes=(32, 16),
                                          activation="relu", alpha=1e-4,
                                          max_iter=2000, solver="adam",
                                          random_state=0))])


def build_xgb_reg() -> XGBRegressor:
    return XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                        objective="reg:squarederror", random_state=0, n_jobs=1,
                        tree_method="hist", verbosity=0)


# --- registry: (algo_name, outcome_type) -> builder -------------------------------
# A missing (algo, type) pair is invalid by construction; the design grid in
# 01_fit_decomposition only pairs ``glm`` with the binary outcome.

_REGISTRY: dict[str, dict[str, Callable]] = {
    "ols": {"binary": build_ols,       "continuous": build_ols},
    "glm": {"binary": build_logistic},
    "gbt": {"binary": build_xgb_clf,   "continuous": build_xgb_reg},
    "net": {"binary": build_nn_clf,    "continuous": build_nn_reg},
}


def get_builder(algo_name: str, y_type: str) -> Callable:
    try:
        return _REGISTRY[algo_name][y_type]
    except KeyError as exc:
        raise ValueError(
            f"No builder for algo_name={algo_name!r} with outcome type {y_type!r}"
        ) from exc


# --- uniform mean prediction ------------------------------------------------------

def predict_mean(model, X) -> float:
    """Mean predicted outcome (R ``predict_mean``, hlp_nonlinear.R:185-194).

    Classifiers expose ``predict_proba``; use P(Y=1) (ICU ``predict_positive_class``).
    Regressors and the OLS/LPM wrapper use ``predict`` directly.
    """
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        yhat = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
    else:
        yhat = np.asarray(model.predict(X), dtype=float).ravel()
    return float(np.mean(yhat))
