"""01 - Point-estimate the OBD across the full design grid (no bootstrap).

Python port of ``census/01_fit-nonlinear-decomposition.R``.

Inputs : census/temp/acs16_workforce.parquet
Output : census/temp/nonlinear_fits.parquet
         one row per (subset_name, subset_value, pop_name, y_name, algo_name) with
         delta_y, explained_0/1, unexplained_0/1, n_0, n_1.

Run from the repo root:  python census/py/01_fit_decomposition.py
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
from joblib import Parallel, delayed
from pyprojroot.here import here

sys.path.insert(0, str(here()))
from obd_engine import decompose_from_data

# Design grid. Each setting can be overridden from the environment so this one script
# serves quick tests and the TabPFN run on the cluster without editing code, e.g.
#   OBD_SUBSET_COLS=st OBD_POP_NAMES=sex_female OBD_ALGOS='{"pincp": ["gbt"]}' \
#       OBD_OUT_SUFFIX=_test python census/py/01_fit_decomposition.py
SUBSET_COLS = os.environ.get("OBD_SUBSET_COLS", "st,naics_2").split(",")
POP_NAMES = os.environ.get("OBD_POP_NAMES", "sex_female,race_bw,immigrant").split(",")
ALGOS_BY_OUTCOME = json.loads(os.environ.get("OBD_ALGOS", json.dumps({
    "pincp": ["ols", "gbt", "net"],          # continuous outcome: no logistic
    "hicov": ["ols", "glm", "glm_r", "gbt", "net"],
})))
# "tabpfn" is registered in obd_engine but left out of the default grid (needs a GPU):
#   OBD_ALGOS='{"pincp": ["tabpfn"], "hicov": ["tabpfn"]}' OBD_OUT_SUFFIX=_tabpfn ...

N_JOBS = int(os.environ.get("OBD_N_JOBS", 8))
OUT_SUFFIX = os.environ.get("OBD_OUT_SUFFIX", "")   # e.g. "_test", "_tabpfn"

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
    # Drop NAICS3 (too many factor levels)
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
    out_path = here() / "census" / "temp" / f"nonlinear_fits{OUT_SUFFIX}.parquet"
    fits.to_parquet(out_path, index=False)
    print(f"Saved {len(fits):,} fit rows -> {out_path}")


if __name__ == "__main__":
    main()
