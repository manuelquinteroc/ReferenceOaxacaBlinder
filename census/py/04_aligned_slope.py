"""04 - Assumption 5.1 ("aligned slope-intercept") diagnostic.

Python port of ``census/04_aligned-slope-intercept.R``, made self-contained: the intercept
gap that R sourced from the un-orchestrated ``xx_check-aligned-slope.R`` (which produced
``temp/ols_aligned.fst``) is recomputed here, so the step runs from a clean checkout.

Inputs : census/temp/nonlinear_fits.parquet
         census/temp/acs16_workforce.parquet
Output : census/out_py/aligned_stats.csv

Run from the repo root:  python census/py/04_aligned_slope.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyprojroot.here import here

sys.path.insert(0, str(here()))
from oaxaca_engine import build_ols, prepare_features  # noqa: E402

N_JOBS = 8


def intercept_gap(sub: pd.DataFrame, pop_name: str, y_name: str) -> float:
    """delta_alpha = intercept(group 1) - intercept(group 0) from OLS (R hlp:175)."""
    sub = sub[sub[pop_name].notna()]
    prepped = prepare_features(sub, pop_name, y_name)
    pop = np.asarray(prepped.pop)
    X, y = prepped.features, np.asarray(prepped.y, dtype=float)
    fit0 = build_ols().fit(X.iloc[np.where(pop == 0)[0]], y[pop == 0])
    fit1 = build_ols().fit(X.iloc[np.where(pop == 1)[0]], y[pop == 1])
    return fit1.intercept - fit0.intercept


def is_aligned(delta_alpha, explained_0, explained_1) -> bool:
    """Assumption 5.1 predicate (R 04_...R:20-26)."""
    return bool(
        (explained_0 * explained_1 > 0)
        and (np.sign(delta_alpha) == np.sign(explained_0))
        and (np.sign(delta_alpha) == np.sign(explained_1))
    )


def main() -> None:
    fits = pd.read_parquet(here() / "census" / "temp" / "nonlinear_fits.parquet")
    acs = pd.read_parquet(here() / "census" / "temp" / "acs16_workforce.parquet")
    acs = acs.drop(columns=["naics_3"])

    # Same base of cases as 02/03, restricted to OLS (R lines 34-37).
    cells = fits[(fits["algo_name"] == "ols")
                 & (fits["n_0"] > 50) & (fits["n_1"] > 50)
                 & (fits["delta_y"].abs() > 0.01)].copy()

    def task(row) -> bool:
        sub = acs[acs[row.subset_name].astype(str) == row.subset_value]
        da = intercept_gap(sub, row.pop_name, row.y_name)
        return is_aligned(da, row.explained_0, row.explained_1)

    aligned = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(task)(row) for row in cells.itertuples(index=False)
    )
    cells = cells.assign(aligned=aligned)

    # Summary stats quoted in Section 5.4 (R lines 41-45).
    align_tb = (cells.groupby("y_name", as_index=False)
                     .agg(pct_aligned=("aligned", "mean"),
                          n_aligned=("aligned", "sum"),
                          n_total=("aligned", "size")))

    out_path = here() / "census" / "out_py" / "aligned_stats.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    align_tb.to_csv(out_path, index=False)
    print(f"Saved aligned stats -> {out_path}")
    print(align_tb.to_string(index=False))


if __name__ == "__main__":
    main()
