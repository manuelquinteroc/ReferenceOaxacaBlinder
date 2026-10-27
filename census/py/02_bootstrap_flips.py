"""02 - Bootstrap only the sign-flip cases that pass the size/magnitude filter.

Inputs : census/temp/acs16_workforce.parquet
         census/temp/nonlinear_fits.parquet
Outputs: census/temp/nonlinear_boots/<cell>__B<B>__seed<seed>.parquet   (resumable chunks)
         census/temp/nonlinear_boots.parquet       (combined)

Run from the repo root:  python census/py/02_bootstrap_flips.py
"""

from __future__ import annotations

import os
import re
import sys
import zlib

import numpy as np
import pandas as pd
from pyprojroot.here import here

sys.path.insert(0, str(here()))
from obd_engine import bootstrap_decomposition  # noqa: E402

# Overridable from the environment (see 01_fit_decomposition.py), e.g. a quick test:
#   OBD_B=50 OBD_OUT_SUFFIX=_test python census/py/02_bootstrap_flips.py
# or bootstrapping only the cheap models first (the run is resumable per cell):
#   OBD_BOOT_ALGOS=ols,glm python census/py/02_bootstrap_flips.py
B = int(os.environ.get("OBD_B", 1000))
INNER_JOBS = int(os.environ.get("OBD_INNER_JOBS", 6))
BOOT_ALGOS = os.environ.get("OBD_BOOT_ALGOS")           # comma list; None = all algos
OUT_SUFFIX = os.environ.get("OBD_OUT_SUFFIX", "")
KEYS = ["subset_name", "subset_value", "pop_name", "y_name", "algo_name"]
DATA_KEYS = KEYS[:-1]                                   # the data cell, without the model


def cell_slug(cell) -> str:
    """Stable file name for a design cell, so resuming works across algo filters."""
    raw = "__".join(str(getattr(cell, k)) for k in KEYS)
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", raw)


def cell_seed(cell) -> int:
    """Bootstrap seed from the cell's identity, not its position in the (filtered, sorted)
    design, so a cell gets the same seed in every invocation. The model is left out, so all
    algorithms in a cell are bootstrapped on the same resamples."""
    return zlib.crc32("__".join(str(getattr(cell, k)) for k in DATA_KEYS).encode())


def chunk_name(cell) -> str:
    """Chunk file keyed on everything that determines its contents (cell, B, seed)."""
    return f"{cell_slug(cell)}__B{B}__seed{cell_seed(cell)}.parquet"


def select_flip_cases(fits: pd.DataFrame) -> pd.DataFrame:
    """Size/magnitude filter + sign-flip criterion (R 02_...R:33-48)."""
    sized = fits[(fits["n_0"] >= 50) & (fits["n_1"] >= 50)
                 & (fits["delta_y"].abs() > 0.01)]
    flips = sized[
        (np.sign(sized["explained_0"] * sized["explained_1"]) != 1)
        | (np.sign(sized["unexplained_0"] * sized["unexplained_1"]) != 1)
    ]
    
    design = flips[KEYS].drop_duplicates().copy()
    cheap = design["algo_name"].isin(["ols", "glm", "glm_r"])
    design["_o"] = (~cheap).astype(int)                 # cheap linear models first
    return design.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def main() -> None:
    acs = pd.read_parquet(here() / "census" / "temp" / "acs16_workforce.parquet")
    acs = acs.drop(columns=["naics_3"])
    fits = pd.read_parquet(here() / "census" / "temp" / f"nonlinear_fits{OUT_SUFFIX}.parquet")

    full_design = select_flip_cases(fits)
    design = full_design
    if BOOT_ALGOS:
        design = design[design["algo_name"].isin(BOOT_ALGOS.split(","))]
    print(f"{len(design)} flip cells to bootstrap (B={B}).")

    boots_dir = here() / "census" / "temp" / f"nonlinear_boots{OUT_SUFFIX}"
    boots_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip cells whose chunk file already exists.
    for i, cell in enumerate(design.itertuples(index=False), start=1):
        chunk_path = boots_dir / chunk_name(cell)
        if chunk_path.exists():
            continue
        sub = acs[acs[cell.subset_name].astype(str) == cell.subset_value]
        bt = bootstrap_decomposition(
            sub, cell.y_name, cell.pop_name, cell.algo_name,
            B=B, random_state=cell_seed(cell), n_jobs=INNER_JOBS,
        )
        for k in KEYS:
            bt[k] = getattr(cell, k)
        bt.to_parquet(chunk_path, index=False)
        print(f"  [{i}/{len(design)}] {cell.subset_name}={cell.subset_value} "
              f"{cell.pop_name} {cell.y_name} {cell.algo_name} -> {chunk_path.name}")

    # Combine the chunks of the full flip design at this B (also those from earlier runs with
    # another OBD_BOOT_ALGOS); stale files from other B, seeds or designs are left out.
    expected = [boots_dir / chunk_name(c) for c in full_design.itertuples(index=False)]
    chunks = [p for p in expected if p.exists()]
    print(f"{len(chunks)} of {len(expected)} cells bootstrapped.")
    boots = pd.concat([pd.read_parquet(p) for p in chunks], ignore_index=True)
    out_path = here() / "census" / "temp" / f"nonlinear_boots{OUT_SUFFIX}.parquet"
    boots.to_parquet(out_path, index=False)
    print(f"Saved {len(boots):,} bootstrap rows -> {out_path}")


if __name__ == "__main__":
    main()
