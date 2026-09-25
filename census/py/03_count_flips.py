"""03 - Bootstrap SEs, normal-approx p-values, and the flip-count table.

Inputs : census/temp/nonlinear_fits.parquet
         census/temp/nonlinear_boots.parquet
Output : census/out_py/flip_counts.csv

Run from the repo root:  python census/py/03_count_flips.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from pyprojroot.here import here

sys.path.insert(0, str(here()))
from obd_engine import pval  # noqa: E402

OUT_SUFFIX = os.environ.get("OBD_OUT_SUFFIX", "")
GROUP_KEYS = ["subset_name", "subset_value", "pop_name", "algo_name", "y_name"]
STAT_COLS = ["explained_0", "explained_1", "unexplained_0", "unexplained_1"]


def bootstrap_se(boots: pd.DataFrame) -> pd.DataFrame:
    """SD of each statistic across bootstrap replicates (R 03_...R:23-32)."""
    value_cols = [c for c in boots.columns
                  if c.startswith("delta") or c.endswith("_0") or c.endswith("_1")]
    long = boots.melt(id_vars=GROUP_KEYS, value_vars=value_cols,
                      var_name="stat_name", value_name="stat_value_boot")
    return (long.groupby(GROUP_KEYS + ["stat_name"], as_index=False)
                .agg(stat_se=("stat_value_boot", lambda s: s.std(ddof=1))))


def main() -> None:
    fits = pd.read_parquet(here() / "census" / "temp" / f"nonlinear_fits{OUT_SUFFIX}.parquet")
    boots_path = here() / "census" / "temp" / f"nonlinear_boots{OUT_SUFFIX}.parquet"
    if boots_path.exists():
        boot_se = bootstrap_se(pd.read_parquet(boots_path))
    else:
        # No bootstrap yet (or none for these algos): SEs are missing, p-values NaN, and
        # the reject_* columns below count 0 -- n_with_se makes that visible.
        print(f"NOTE: {boots_path.name} not found; reporting point-estimate flips only.")
        boot_se = pd.DataFrame(columns=GROUP_KEYS + ["stat_name", "stat_se"])

    # Point estimates, same size/magnitude filter as 02 (R lines 36-38).
    flips = fits[(fits["n_0"] > 50) & (fits["n_1"] > 50)
                 & (fits["delta_y"].abs() > 0.01)].copy()

    # Long over the four explained/unexplained columns; attach SEs; split into
    # component + ref_pop; normal-approx p-value (R lines 39-47).
    long = flips.melt(id_vars=GROUP_KEYS, value_vars=STAT_COLS,
                      var_name="stat_name", value_name="stat_value")
    long = long.merge(boot_se, on=GROUP_KEYS + ["stat_name"], how="left")
    comp_ref = long["stat_name"].str.rsplit("_", n=1, expand=True)
    long["component"] = comp_ref[0]
    long["ref_pop"] = comp_ref[1]
    long["p_value"] = [pval(e, s) for e, s in zip(long["stat_value"], long["stat_se"])]

    # Per (cell, component): flip if the two ref-group estimates have opposite signs;
    # significant if the smaller p-value clears the threshold (R lines 48-58).
    by_comp = GROUP_KEYS[:-1] + ["y_name", "component"]  # subset/value/pop/algo/y/component
    flips_components = (long.groupby(GROUP_KEYS + ["component"], as_index=False)
                            .agg(is_flip=("stat_value", lambda s: float(np.prod(s)) < 0),
                                 min_p_value=("p_value", "min"),
                                 has_se=("stat_se", lambda s: bool(s.notna().all()))))
    flips_components["reject_10"] = flips_components["is_flip"] & (flips_components["min_p_value"] < 0.10)
    flips_components["reject_05"] = flips_components["is_flip"] & (flips_components["min_p_value"] < 0.05)
    flips_components["reject_01"] = flips_components["is_flip"] & (flips_components["min_p_value"] < 0.01)

    # Per-component counts (R lines 60-68).
    flip_counts_component = (
        flips_components.groupby(["y_name", "algo_name", "component"], as_index=False)
        .agg(n_fit=("is_flip", "size"),
             is_flip=("is_flip", "sum"),
             reject_10=("reject_10", "sum"),
             reject_05=("reject_05", "sum"),
             reject_01=("reject_01", "sum"),
             n_with_se=("has_se", "sum"))
    )

    # "Either component" aggregate per cell, then counts (R lines 72-89).
    flips_total = (
        flips_components.groupby(GROUP_KEYS, as_index=False)
        .agg(is_flip=("is_flip", "any"),
             reject_10=("reject_10", "any"),
             reject_05=("reject_05", "any"),
             reject_01=("reject_01", "any"),
             has_se=("has_se", "all"))
    )
    flip_counts_total = (
        flips_total.groupby(["y_name", "algo_name"], as_index=False)
        .agg(n_fit=("is_flip", "size"),
             is_flip=("is_flip", "sum"),
             reject_10=("reject_10", "sum"),
             reject_05=("reject_05", "sum"),
             reject_01=("reject_01", "sum"),
             n_with_se=("has_se", "sum"))
    )
    flip_counts_total["component"] = "either"

    flip_counts = pd.concat([flip_counts_component, flip_counts_total], ignore_index=True)
    # Order columns as in R output.
    flip_counts = flip_counts[["y_name", "algo_name", "component", "n_fit",
                               "is_flip", "reject_10", "reject_05", "reject_01", "n_with_se"]]

    out_path = here() / "census" / "out_py" / f"flip_counts{OUT_SUFFIX}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flip_counts.to_csv(out_path, index=False)
    print(f"Saved flip counts -> {out_path}")
    print(flip_counts.to_string(index=False))


if __name__ == "__main__":
    main()
