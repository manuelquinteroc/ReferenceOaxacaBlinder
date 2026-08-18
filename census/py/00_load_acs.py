"""00 - Load 2016 ACS PUMS and build the analysis table.

Python port of ``census/00_load-acs-16.R``.

Inputs : census/raw/ss16pusa.csv, census/raw/ss16pusb.csv  (2016 ACS 1-Year PUMS)
Output : census/temp/acs16_workforce.parquet

Run from the repo root:  python census/py/00_load_acs.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from pyprojroot.here import here

sys.path.insert(0, str(here()))  # make oaxaca_engine importable (unused here, kept uniform)

# Columns to read (R cols_only, 00_load-acs-16.R:18-33). All integer except NAICSP.
INT_COLS = ["ST", "AGEP", "ESR", "WKHP", "WKW", "PERNP", "PINCP",
            "NATIVITY", "HICOV", "SEX", "RAC1P", "SCHL", "MAR", "INDP"]
STR_COLS = ["NAICSP"]
USECOLS = INT_COLS + STR_COLS

# Full-time federal minimum wage floor: $7.25 x 35 hrs x 50 wks (R lines 54-55).
MIN_EARNINGS = 12687.50


def load_acs() -> pd.DataFrame:
    """Read both PUMS files, lowercase columns, drop rows with any missing value."""
    raw_dir = here() / "census" / "raw"
    frames = []
    for name in ("ss16pusa.csv", "ss16pusb.csv"):
        frames.append(pd.read_csv(
            raw_dir / name,
            usecols=USECOLS,
            dtype={c: "string" for c in STR_COLS},  # ints left to inference (allow NA)
            low_memory=False,
        ))
    acs = pd.concat(frames, ignore_index=True)
    acs.columns = acs.columns.str.lower()        # R rename_with(tolower)
    return acs.dropna()                           # R na.omit()


def main() -> None:
    acs_raw = load_acs()

    # Sample restrictions from Bach et al. (2024) (R 00_load-acs-16.R:50-55).
    mask = (
        acs_raw["agep"].between(25, 65)
        & (acs_raw["wkhp"] >= 35)
        & (acs_raw["wkw"] == 1)
        & (acs_raw["esr"] == 1)
        & (acs_raw["pernp"] >= MIN_EARNINGS)
        & (acs_raw["pincp"] >= MIN_EARNINGS)
    )
    acs = acs_raw[mask].copy()

    # Integer-coded industry; R coerces to character before substr. Cast via int to
    # avoid "170.0"-style float strings.
    indp_str = acs["indp"].astype("int64").astype(str)
    naicsp_str = acs["naicsp"].astype(str)

    out = pd.DataFrame({
        "st": acs["st"].astype("int64"),
        "indp_2": indp_str.str[:2],                      # 2-digit INDP (covariate)
        "naics_2": naicsp_str.str[:2],                   # 2-digit NAICS (subset id)
        "naics_3": naicsp_str.str[:3],
        "pincp": np.log(acs["pincp"].astype(float)),     # log income (continuous outcome)
        "sex_female": (acs["sex"] == 2).astype(float),
        "hicov": (acs["hicov"] == 1).astype(float),
        "immigrant": (acs["nativity"] == 2).astype(float),
        "mar": (acs["mar"] == 1).astype(float),
        "educ_bach": (acs["schl"] >= 21).astype(float),  # bachelor's or higher
        # NaN for non-Black/non-White; those rows are still used where race_bw isn't pop.
        "race_bw": np.select(
            [acs["rac1p"] == 1, acs["rac1p"] == 2],
            [0.0, 1.0],
            default=np.nan,
        ),
    })

    out_path = here() / "census" / "temp" / "acs16_workforce.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Saved {len(out):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
