"""Model builders and the algorithm registry.

These are the *same* estimators used by the ICU analysis in
``Real-data example/icu_139_final_models.ipynb`` (classification builders copied
verbatim), extended here with regression twins so that the continuous census outcome
(``pincp`` = log income) can be decomposed. Hyperparameters are fixed; there is no
cross-validation / early-stopping tuning (the R ``lgb.cv`` path is intentionally dropped).

Algorithm names follow the R census convention so that downstream output schemas match:

    ols  -> statsmodels OLS  (linear; a linear probability model when y is binary)
    glm  -> sklearn LogisticRegression, L2 (binary outcomes only; ICU notebook)
    glm_r -> unpenalized logistic, R glm-faithful (binary outcomes only; census paper)
    gbt  -> XGBoost          (XGBClassifier / XGBRegressor)
    net  -> sklearn MLP      (MLPClassifier / MLPRegressor)
    tabpfn -> TabPFN v2 (TabPFNClassifier / TabPFNRegressor), optional dependency
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
# xgboost is imported inside its builders (not here): on macOS, loading xgboost's OpenMP
# runtime before torch's makes TabPFN segfault, so a tabpfn-only run must never import it.


# --- outcome type -----------------------------------------------------------------

def outcome_type(y) -> str:
    """'binary' if every value is in {0, 1}, else 'continuous' (R hlp_nonlinear.R:206)."""
    arr = np.asarray(y, dtype=float)
    arr = arr[~np.isnan(arr)]
    return "binary" if np.isin(arr, (0.0, 1.0)).all() else "continuous"


# --- rank deficiency, handled the way R does ------------------------------------------
# Subgroups can be small (n ~ 50-200) while the design has ~150 columns (state and industry
# dummies plus missing-value flags), so exact collinearity is routine. R's lm.fit / glm.fit
# *alias* a column that is a linear combination of the columns before it (coefficient NA,
# dropped at prediction). statsmodels' pseudo-inverse instead spreads the weight across the
# collinear columns: the same fitted values in-sample, but different predictions on the
# *other* group -- which is exactly what the decomposition uses. That difference alone
# tripled the census sign-flip counts relative to the R pipeline, so the linear and
# unpenalized logistic fits below reproduce R's column selection.

def aliased_free_columns(X: np.ndarray, rtol: float = 1e-9) -> list[int]:
    """Indices of the columns R's QR (order-preserving, limited pivoting) would keep.

    Works on the Gram matrix with an incremental Cholesky factor, so the cost is O(n p^2)
    in BLAS plus O(p^3) in Python -- cheap enough to run inside a 1000-replicate bootstrap.
    """
    G = X.T @ X
    diag = np.diag(G)
    kept: list[int] = []
    L = np.zeros((0, 0))
    for j in range(X.shape[1]):
        if kept:
            w = np.linalg.solve(L, G[kept, j])           # L w = g_Kj  (L lower-triangular)
            resid = diag[j] - w @ w                       # squared norm after projecting on kept
        else:
            w, resid = np.zeros(0), diag[j]
        if resid > rtol * max(diag[j], 1.0):
            d = np.sqrt(resid)
            L = np.block([[L, np.zeros((len(kept), 1))], [w[None, :], np.array([[d]])]])
            kept.append(j)
    return kept


class OLSModel:
    """Least-squares linear model with R's aliasing of collinear columns (see above).

    Presents .fit(X, y) / .predict(X); no predict_proba, so predict_mean treats it as a
    regressor -- i.e. a linear probability model for binary outcomes, matching R's gaussian
    glm.fit.
    """

    def __init__(self):
        self.kept_ = None
        self.beta_ = None

    @staticmethod
    def _design(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X, y):
        Xa = self._design(X)
        self.kept_ = aliased_free_columns(Xa)
        self.beta_ = np.linalg.lstsq(Xa[:, self.kept_], np.asarray(y, dtype=float), rcond=None)[0]
        return self

    def predict(self, X):
        return self._design(X)[:, self.kept_] @ self.beta_

    @property
    def intercept(self) -> float:
        """Intercept term, used by the aligned-slope diagnostic (step 04)."""
        return float(self.beta_[0])       # the constant column is never aliased


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


class LogisticUnpenalizedModel:
    """Unpenalized logistic regression with R's aliasing of collinear columns.

    This is R's ``glm(..., family = binomial)`` -- the census paper's "Logistic (unpenalized)"
    -- as opposed to ``LogisticModel`` above, the ICU notebook's sklearn default (L2, C = 1)
    on standardized inputs. Fit by IRLS with R's iteration cap; degrades to a constant on a single-class group.
    """

    def __init__(self):
        self.kept_ = None
        self.beta_ = None
        self.const_ = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if np.unique(y).size < 2:
            self.const_ = float(y[0]) if y.size else np.nan
            return self
        Xa = OLSModel._design(X)
        self.kept_ = aliased_free_columns(Xa)
        Xk = Xa[:, self.kept_]
        # IRLS with R's glm.control() defaults (maxit = 25). On quasi-separated groups --
        # common here, since insurance coverage is 94% and small groups contain dummies
        # that predict it perfectly -- the likelihood has no finite maximum; R stops after
        # 25 iterations with large coefficients, and so does this. (Newton with a high
        # iteration cap gives the same fitted probabilities but ~10x slower.)
        self.beta_ = np.asarray(sm.GLM(y, Xk, family=sm.families.Binomial()).fit(maxiter=25).params)
        return self

    def predict_proba(self, X):
        if self.const_ is not None:
            p = np.full(len(X), self.const_, dtype=float)
        else:
            eta = OLSModel._design(X)[:, self.kept_] @ self.beta_
            p = expit(eta)                      # overflow-safe sigmoid
        return np.column_stack([1.0 - p, p])


def build_logistic_unpenalized() -> LogisticUnpenalizedModel:
    return LogisticUnpenalizedModel()


# The paper's settings (Appendix B.1). The keyword arguments exist only for the
# complexity sweeps (census/py/05_complexity_sweep.py); the registry calls these with
# no arguments, so the defaults ARE the paper's models.
NN_HIDDEN, NN_SEED = (32, 16), 0
XGB_TREES, XGB_DEPTH = 250, 3


def build_nn_clf(hidden=NN_HIDDEN, seed=NN_SEED) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=hidden,
                                           activation="relu", alpha=1e-4,
                                           max_iter=2000, solver="adam",
                                           random_state=seed))])


def build_xgb_clf(n_estimators=XGB_TREES, max_depth=XGB_DEPTH):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                         eval_metric="logloss", random_state=0, n_jobs=1,
                         tree_method="hist", verbosity=0)


# --- regression twins (same architecture / hyperparameters) -----------------------

def build_nn_reg(hidden=NN_HIDDEN, seed=NN_SEED) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()),
                     ("reg", MLPRegressor(hidden_layer_sizes=hidden,
                                          activation="relu", alpha=1e-4,
                                          max_iter=2000, solver="adam",
                                          random_state=seed))])


def build_xgb_reg(n_estimators=XGB_TREES, max_depth=XGB_DEPTH):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                        objective="reg:squarederror", random_state=0, n_jobs=1,
                        tree_method="hist", verbosity=0)


# --- TabPFN (pre-trained transformer, in-context learner) --------------------------
# Same model family as the ICU analysis (``Real-data example/icu_139_tabpfn_local.ipynb``:
# ``TabPFNClassifier(device=..., ignore_pretraining_limits=True)``), plus a regressor twin
# for log income. ``tabpfn`` is imported lazily so the engine works without it installed.
#
# TabPFN's pretraining context is ~10k rows; the largest census cells have ~100k. Groups
# larger than OBD_TABPFN_MAX_ROWS (default 10000) are fit on a fixed random subsample
# but *predicted* on every row, so the counterfactual means still average over the full
# group. Environment knobs (all optional):
#   OBD_TABPFN_DEVICE      cuda | mps | cpu | auto (default: auto = cuda if present, else cpu)
#   OBD_TABPFN_CLF_PATH    local classifier .ckpt  (default: TabPFN's own cache / download)
#   OBD_TABPFN_REG_PATH    local regressor  .ckpt  (default: TabPFN's own cache / download)
#   OBD_TABPFN_MAX_ROWS    training-row cap per group (default 10000)
# Weights: TabPFN >= 7 downloads them only after the license is accepted and TABPFN_TOKEN
# is set (https://ux.priorlabs.ai). The ICU run used the v2.6 checkpoints; point the
# *_PATH variables at copies of those on an air-gapped node.

def _tabpfn_device() -> str:
    dev = os.environ.get("OBD_TABPFN_DEVICE", "auto")
    if dev != "auto":
        return dev
    import torch
    if torch.cuda.is_available():
        return "cuda"
    # Apple MPS is deliberately *not* auto-selected: TabPFN 7.1 / torch 2.8 segfault on it
    # (exit 139) on the census design matrices. Opt in with OBD_TABPFN_DEVICE=mps.
    return "cpu"


class _TabPFNBase:
    max_rows = int(os.environ.get("OBD_TABPFN_MAX_ROWS", 10000))

    path_env = "OBD_TABPFN_CLF_PATH"          # overridden by the regressor subclass

    def __init__(self, estimator_cls):
        kwargs = dict(device=_tabpfn_device(), ignore_pretraining_limits=True, random_state=0)
        path = os.environ.get(self.path_env)
        if path:
            kwargs["model_path"] = os.path.expanduser(path)
        self.model_ = estimator_cls(**kwargs)

    def _subsample(self, X, y):
        n = len(X)
        if n <= self.max_rows:
            return X, y
        idx = np.random.default_rng(0).choice(n, size=self.max_rows, replace=False)
        return X.iloc[idx], np.asarray(y)[idx]

    def fit(self, X, y):
        Xs, ys = self._subsample(X, y)
        self.model_.fit(Xs, ys)
        return self


class TabPFNClf(_TabPFNBase):
    """Classifier; degrades to a constant on a single-class group (mirrors LogisticModel)."""

    def __init__(self):
        from tabpfn import TabPFNClassifier
        super().__init__(TabPFNClassifier)
        self.const_ = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if np.unique(y).size < 2:
            self.const_ = float(y[0]) if y.size else np.nan
            return self
        return super().fit(X, y.astype(int))

    def predict_proba(self, X):
        if self.const_ is not None:
            p = np.full(len(X), self.const_, dtype=float)
            return np.column_stack([1.0 - p, p])
        return np.asarray(self.model_.predict_proba(X))


class TabPFNReg(_TabPFNBase):
    path_env = "OBD_TABPFN_REG_PATH"

    def __init__(self):
        from tabpfn import TabPFNRegressor
        super().__init__(TabPFNRegressor)

    def predict(self, X):
        return np.asarray(self.model_.predict(X), dtype=float)


def build_tabpfn_clf() -> TabPFNClf:
    return TabPFNClf()


def build_tabpfn_reg() -> TabPFNReg:
    return TabPFNReg()


# --- lightgbm, exactly as the paper's R pipeline tuned it -------------------------
# ``gbt`` in census/hlp_nonlinear.R: learning rate 0.25, 7 leaves, number of trees chosen per
# group by 3-fold CV (stratified on the outcome; quintiles if continuous) with early stopping
# after 10 rounds, then refit on the whole group with that many trees. Registered as ``lgb_r``
# so the draft's lightgbm rows can be compared with the fixed-hyperparameter XGBoost (``gbt``).

class LGBMRLike:
    LR, LEAVES, MAX_ROUNDS, PATIENCE, FOLDS, SEED = 0.25, 7, 10_000, 10, 3, 1

    def __init__(self, binary: bool):
        self.binary = binary
        self.booster_ = None
        self.const_ = None
        self.best_iter_ = None

    def fit(self, X, y):
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        if self.binary and np.unique(y).size < 2:
            self.const_ = float(y[0]); return self
        strata = y.astype(int) if self.binary else pd.qcut(pd.Series(y).rank(method="first"), 5, labels=False).to_numpy()
        params = {"objective": "binary" if self.binary else "regression", "learning_rate": self.LR,
                  "num_leaves": self.LEAVES, "verbose": -1, "num_threads": 1, "seed": self.SEED}
        folds = list(StratifiedKFold(self.FOLDS, shuffle=True, random_state=self.SEED).split(X, strata))
        cv = lgb.cv(params, lgb.Dataset(X, y), num_boost_round=self.MAX_ROUNDS, folds=folds,
                    callbacks=[lgb.early_stopping(self.PATIENCE, verbose=False)])
        self.best_iter_ = len(next(iter(cv.values())))
        self.booster_ = lgb.train(params, lgb.Dataset(X, y), num_boost_round=self.best_iter_)
        return self

    def _raw(self, X):
        return np.asarray(self.booster_.predict(np.asarray(X, dtype=float)), dtype=float)

    def predict(self, X):
        return self._raw(X)

    def predict_proba(self, X):
        p = np.full(len(X), self.const_) if self.const_ is not None else self._raw(X)
        return np.column_stack([1.0 - p, p])


def build_lgb_r_clf(): return LGBMRLike(binary=True)
def build_lgb_r_reg(): return LGBMRLike(binary=False)


# --- registry: (algo_name, outcome_type) -> builder -------------------------------
# A missing (algo, type) pair is invalid by construction; the design grid in
# 01_fit_decomposition only pairs ``glm`` with the binary outcome.

_REGISTRY: dict[str, dict[str, Callable]] = {
    "ols": {"binary": build_ols,       "continuous": build_ols},
    "glm": {"binary": build_logistic},                 # ICU-style: sklearn, L2 (C=1)
    "glm_r": {"binary": build_logistic_unpenalized},   # R-style: unpenalized (census paper)
    "gbt": {"binary": build_xgb_clf,   "continuous": build_xgb_reg},
    "net": {"binary": build_nn_clf,    "continuous": build_nn_reg},
    "tabpfn": {"binary": build_tabpfn_clf, "continuous": build_tabpfn_reg},
    "lgb_r": {"binary": build_lgb_r_clf, "continuous": build_lgb_r_reg},   # paper's lightgbm (R gbt)
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
