"""02 - Bootstrap only the sign-flip cases that pass the size/magnitude filter.

Python port of ``census/02_bootstrap-nonlinear-decomposition.R``.

Inputs : census/temp/acs16_workforce.parquet
         census/temp/nonlinear_fits.parquet
Outputs: census/temp/nonlinear_boots/<i>.parquet   (resumable chunks)
         census/temp/nonlinear_boots.parquet       (combined)

Run from the repo root:  python census/py/02_bootstrap_flips.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from pyprojroot.here import here

sys.path.insert(0, str(here()))
from oaxaca_engine import bootstrap_decomposition  # noqa: E402

B = 1000          # bootstrap replicates (R n_boot = 1000)
INNER_JOBS = 6    # R plan(multicore, workers = min(6, availableCores()))
KEYS = ["subset_name", "subset_value", "pop_name", "y_name", "algo_name"]


def select_flip_cases(fits: pd.DataFrame) -> pd.DataFrame:
    """Size/magnitude filter + sign-flip criterion (R 02_...R:33-48)."""
    sized = fits[(fits["n_0"] > 50) & (fits["n_1"] > 50)
                 & (fits["delta_y"].abs() > 0.01)]
    flips = sized[
        (np.sign(sized["explained_0"] * sized["explained_1"]) != 1)
        | (np.sign(sized["unexplained_0"] * sized["unexplained_1"]) != 1)
    ]
    # Sort ols/glm first for throughput (cosmetic; R lines 47-48).
    design = flips[KEYS].drop_duplicates().copy()
    design["_o"] = (design["algo_name"] != "ols").astype(int) \
        + (design["algo_name"] != "glm").astype(int)
    return design.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def main() -> None:
    acs = pd.read_parquet(here() / "census" / "temp" / "acs16_workforce.parquet")
    acs = acs.drop(columns=["naics_3"])
    fits = pd.read_parquet(here() / "census" / "temp" / "nonlinear_fits.parquet")

    design = select_flip_cases(fits)
    print(f"{len(design)} flip cells to bootstrap (B={B}).")

    boots_dir = here() / "census" / "temp" / "nonlinear_boots"
    boots_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip cells whose chunk file already exists (R counts existing files).
    for i, cell in enumerate(design.itertuples(index=False), start=1):
        chunk_path = boots_dir / f"{i}.parquet"
        if chunk_path.exists():
            continue
        sub = acs[acs[cell.subset_name].astype(str) == cell.subset_value]
        bt = bootstrap_decomposition(
            sub, cell.y_name, cell.pop_name, cell.algo_name,
            B=B, random_state=i, n_jobs=INNER_JOBS,
        )
        for k in KEYS:
            bt[k] = getattr(cell, k)
        bt.to_parquet(chunk_path, index=False)
        print(f"  [{i}/{len(design)}] {cell.subset_name}={cell.subset_value} "
              f"{cell.pop_name} {cell.y_name} {cell.algo_name} -> {chunk_path.name}")

    # Combine all chunks.
    chunks = sorted(boots_dir.glob("*.parquet"), key=lambda p: int(p.stem))
    boots = pd.concat([pd.read_parquet(p) for p in chunks], ignore_index=True)
    out_path = here() / "census" / "temp" / "nonlinear_boots.parquet"
    boots.to_parquet(out_path, index=False)
    print(f"Saved {len(boots):,} bootstrap rows -> {out_path}")


if __name__ == "__main__":
    main()
