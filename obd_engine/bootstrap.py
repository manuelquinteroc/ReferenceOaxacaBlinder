"""Stratified (within-group) bootstrap of the decomposition.

Port of the resampling and bootstrap branch of ``census/hlp_nonlinear.R``
(``resample`` lines 311-322; the ``twoway_subset`` bootstrap loop lines 37-57), using the
ICU notebook's paired-bootstrap idiom and ``joblib`` for parallelism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .decomposition import decompose_from_data


def resample(data: pd.DataFrame, strata_name: str, rng: np.random.Generator) -> pd.DataFrame:
    """Resample rows with replacement *within* each level of ``strata_name``.

    Singleton strata are returned unchanged (R hlp_nonlinear.R:316-317).
    """
    parts = []
    for _, grp in data.groupby(strata_name, sort=False, observed=True):
        n = len(grp)
        if n == 1:
            parts.append(grp)
        else:
            take = rng.integers(0, n, size=n)
            parts.append(grp.iloc[take])
    return pd.concat(parts, axis=0)


def bootstrap_decomposition(data: pd.DataFrame, y_name: str, pop_name: str,
                            algo_name: str, B: int = 1000, random_state: int = 1,
                            n_jobs: int = -1) -> pd.DataFrame:
    """Bootstrap one design cell.

    Returns a DataFrame with ``B`` rows (one per replicate, indexed by ``boot_iter``) of
    the decomposition statistics. ``data`` must already be restricted to the subset.
    """
    data = data[data[pop_name].notna()].reset_index(drop=True)

    # Deterministic per-replicate seeds from one parent generator.
    seeds = np.random.default_rng(random_state).integers(1_000_000_000, size=B)

    def one(b: int, seed: int) -> dict:
        rng = np.random.default_rng(int(seed))
        boot = resample(data, pop_name, rng)
        row = decompose_from_data(boot, y_name, pop_name, algo_name)
        row["boot_iter"] = b + 1
        return row

    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(one)(b, int(s)) for b, s in enumerate(seeds)
    )
    return pd.DataFrame(rows)
