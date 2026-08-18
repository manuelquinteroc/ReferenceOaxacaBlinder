"""01 - Point-estimate the OBD across the full design grid (no bootstrap).

Python port of ``census/01_fit-nonlinear-decomposition.R``.

Inputs : census/temp/acs16_workforce.parquet
Output : census/temp/nonlinear_fits.parquet
         one row per (subset_name, subset_value, pop_name, y_name, algo_name) with
         delta_y, explained_0/1, unexplained_0/1, n_0, n_1.

Run from the repo root:  python census/py/01_fit_decomposition.py
"""

from __future__ import annotations

import sys

import pandas as pd
from joblib import Parallel, delayed
from pyprojroot.here import here

sys.path.insert(0, str(here()))
from oaxaca_engine import decompose_from_data  # noqa: E402

# Design grid (R 01_fit-nonlinear-decomposition.R:34-52).
SUBSET_COLS = ["st", "naics_2"]
POP_NAMES = ["sex_female", "race_bw", "immigrant"]
# Algorithms per outcome. R fits ols/gbt for pincp and ols/glm/gbt for hicov; we add the
# neural net (``net``). ``glm`` (logistic) is binary-only.
ALGOS_BY_OUTCOME = {
    "pincp": ["ols", "gbt", "net"],
    "hicov": ["ols", "glm", "gbt", "net"],
}

N_JOBS = 8  # R uses plan(multicore, workers = 8)


def build_subsets(acs: pd.DataFrame) -> pd.DataFrame:
    """Distinct (subset_name, subset_value) pairs over st and naics_2, as strings."""
    long = acs[SUBSET_COLS].astype(str).melt(
        var_name="subset_name", value_name="subset_value")
    return (long.drop_duplicates()
                .sort_values(["subset_name", "subset_value"])
                .reset_index(drop=True))


def fit_subset(subset_name: str, subset_value: str, sub: pd.DataFrame) -> list[dict]:
    """All pop x outcome x algo decompositions for a single subset."""
    rows = []
    for pop_name in POP_NAMES:
        for y_name, algos in ALGOS_BY_OUTCOME.items():
            for algo_name in algos:
                res = decompose_from_data(sub, y_name, pop_name, algo_name)
                rows.append({
                    "subset_name": subset_name,
                    "subset_value": subset_value,
                    "pop_name": pop_name,
                    "y_name": y_name,
                    "algo_name": algo_name,
                    **res,
                })
    return rows


def main() -> None:
    acs = pd.read_parquet(here() / "census" / "temp" / "acs16_workforce.parquet")
    # Drop NAICS3: too many factor levels, slows ML fits (R line 28).
    acs = acs.drop(columns=["naics_3"])

    subsets = build_subsets(acs)
    print(f"Fitting {len(subsets)} subsets x {len(POP_NAMES)} pops x outcomes/algos...")

    # Parallelize across subsets; each worker pickles only its own slice.
    def task(subset_name: str, subset_value: str):
        sub = acs[acs[subset_name].astype(str) == subset_value]
        return fit_subset(subset_name, subset_value, sub)

    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(task)(r.subset_name, r.subset_value)
        for r in subsets.itertuples(index=False)
    )

    fits = pd.DataFrame([row for chunk in results for row in chunk])
    out_path = here() / "census" / "temp" / "nonlinear_fits.parquet"
    fits.to_parquet(out_path, index=False)
    print(f"Saved {len(fits):,} fit rows -> {out_path}")


if __name__ == "__main__":
    main()
