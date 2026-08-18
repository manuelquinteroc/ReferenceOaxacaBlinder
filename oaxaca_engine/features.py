"""Feature preparation for the nonlinear Oaxaca-Blinder decomposition.

Python port of ``prepare_data`` in ``census/hlp_nonlinear.R`` (lines 66-98), minus the
cross-validation fold machinery (no hyperparameter tuning is used in this pipeline).

The two transformations that matter for fidelity with the R code are:

1. One-hot encoding of the high-cardinality categoricals ``st`` and ``indp_2``
   (R uses ``fastDummies::dummy_cols(remove_selected_columns = TRUE)``).
2. The "missing-indicator trick": for every column that contains a missing value, append
   a 0/1 ``is_na_<i>`` flag column and zero out the NAs in the original, so tree/linear
   models train without NA propagation (R lines 85-91).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Columns that are one-hot encoded when present. ``naics_2`` is excluded from features
# entirely: it is only a subset identifier (R drops it alongside pop/outcome).
CATEGORICAL_COLS = ["st", "indp_2"]
EXCLUDE_FROM_FEATURES = ["naics_2"]


@dataclass
class PreparedData:
    """Result of :func:`prepare_features`."""

    pop: np.ndarray          # group membership (0/1), one entry per row
    y: np.ndarray            # outcome, one entry per row
    features: pd.DataFrame   # numeric design matrix (one-hot + missing indicators)


def prepare_features(data: pd.DataFrame, pop_name: str, y_name: str) -> PreparedData:
    """Build the numeric design matrix for one subset.

    Parameters
    ----------
    data : DataFrame
        Rows of a single subset (already restricted to the subset and to non-missing
        ``pop_name``).
    pop_name : str
        Name of the binary population/group column.
    y_name : str
        Name of the outcome column.
    """
    # Feature columns: everything except pop, outcome, and the subset identifier.
    drop_cols = [pop_name, y_name, *EXCLUDE_FROM_FEATURES]
    feature_cols = [c for c in data.columns if c not in drop_cols]
    features = data[feature_cols].copy()

    # One-hot encode the categoricals that are actually present among the features.
    cat_present = [c for c in CATEGORICAL_COLS if c in features.columns]
    if cat_present:
        features = pd.get_dummies(features, columns=cat_present, dtype=float)

    # Ensure a plain float matrix (booleans -> float, etc.).
    features = features.astype(float)

    # Missing-indicator trick: iterate over the columns as they currently stand so the
    # appended ``is_na_<i>`` flags line up with R's 1-based column loop.
    original_cols = list(features.columns)
    for i, col in enumerate(original_cols, start=1):
        if features[col].isna().any():
            flag = features[col].isna().astype(float)
            features[col] = features[col].fillna(0.0)
            features[f"is_na_{i}"] = flag.to_numpy()

    return PreparedData(
        pop=data[pop_name].to_numpy(),
        y=data[y_name].to_numpy(),
        features=features.reset_index(drop=True),
    )
