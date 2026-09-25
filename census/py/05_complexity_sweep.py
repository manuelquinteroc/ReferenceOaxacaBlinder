"""05 - Sign flips vs. model complexity on the census design (companion to the ICU sweep
``Real-data example/icu_139_complexity_sweep.ipynb``).

Same four sweeps as the ICU notebook, on the 217 (log income) + 183 (insurance) cells that
enter Tables 8-9 (n > 50 per group and |delta_y| > 0.01, read off nonlinear_fits.parquet):

    xgb_depth   XGBoost max_depth 1..10           (paper: 3; 250 trees, lr 0.05, ...)
    xgb_trees   XGBoost n_estimators 10..10,000    (paper: 250; depth 3)
    nn_width    MLP (w, w, w), w = 8..W_MAX, 3 seeds   (paper: (32, 16))
    nn_depth    MLP (32,) * d, d = 1..16, 3 seeds     (paper: (32, 16))

Everything else is the paper's setting (obd_engine.builders defaults). Point estimates only,
no bootstrap. Results are cached per (sweep, setting) in census/complexity_sweep/<sweep>_rows.csv
and each run only fits the settings that are missing, so the script can be stopped and resumed.
Plots: census/py/census_complexity_sweep.ipynb.

Run from the repo root (hours; use tmux):
    OBD_N_JOBS=12 python census/py/05_complexity_sweep.py [xgb_depth xgb_trees nn_width nn_depth]
Environment: OBD_N_JOBS (default 8), OBD_NN_WIDTHS / OBD_NN_DEPTHS / OBD_XGB_TREES / OBD_XGB_DEPTHS
(comma lists), OBD_NN_SEEDS (default "0,1,2"), OBD_RECOMPUTE=1 to ignore the cache.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyprojroot.here import here

sys.path.insert(0, str(here()))
from obd_engine import prepare_features, decompose_mean, outcome_type  # noqa: E402
from obd_engine.builders import (build_xgb_clf, build_xgb_reg,           # noqa: E402
                                 build_nn_clf, build_nn_reg,
                                 XGB_TREES, XGB_DEPTH, NN_HIDDEN)

N_JOBS = int(os.environ.get("OBD_N_JOBS", 8))
RECOMPUTE = os.environ.get("OBD_RECOMPUTE", "0") == "1"
_ints = lambda key, default: [int(x) for x in os.environ.get(key, default).split(",")]
XGB_DEPTHS = _ints("OBD_XGB_DEPTHS", "1,2,3,4,5,6,7,8,9,10")
XGB_TREES_GRID = _ints("OBD_XGB_TREES", "10,25,50,100,250,500,1000,2000,4000,10000")
NN_WIDTHS = _ints("OBD_NN_WIDTHS", "8,16,32,64,128,256")
NN_DEPTHS = _ints("OBD_NN_DEPTHS", "1,2,3,4,6,8,12,16")
NN_SEEDS = _ints("OBD_NN_SEEDS", "0,1,2")

OUT_DIR = here() / "census" / "complexity_sweep"
OUT_DIR.mkdir(exist_ok=True)
CELL_KEYS = ["subset_name", "subset_value", "pop_name", "y_name"]


def nn_n_params(hidden, n_in, n_out=1):
    sizes = [n_in, *hidden, n_out]
    return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])))


def table_cells() -> pd.DataFrame:
    """The cells behind Tables 8-9: the 03_count_flips filter applied to the OLS rows
    (n_0, n_1 and delta_y do not depend on the model)."""
    fits = pd.read_parquet(here() / "census" / "temp" / "nonlinear_fits.parquet")
    ols = fits[fits["algo_name"] == "ols"]
    keep = ols[(ols["n_0"] > 50) & (ols["n_1"] > 50) & (ols["delta_y"].abs() > 0.01)]
    return keep[CELL_KEYS].drop_duplicates().reset_index(drop=True)


def make_builder(sweep: str, knob, y_type: str):
    if sweep == "xgb_depth":
        b = build_xgb_clf if y_type == "binary" else build_xgb_reg
        return lambda: b(max_depth=knob)
    if sweep == "xgb_trees":
        b = build_xgb_clf if y_type == "binary" else build_xgb_reg
        return lambda: b(n_estimators=knob)
    b = build_nn_clf if y_type == "binary" else build_nn_reg
    if sweep == "nn_width":
        w, s = knob
        return lambda: b(hidden=(w,) * 3, seed=s)
    if sweep == "nn_depth":
        d, s = knob
        return lambda: b(hidden=(32,) * d, seed=s)
    raise ValueError(sweep)


def fit_cell(sweep, label, knob, cell, sub):
    """One (setting, cell) decomposition; returns a flat row."""
    pop_name, y_name = cell["pop_name"], cell["y_name"]
    data = sub[sub[pop_name].notna()]
    prepped = prepare_features(data, pop_name, y_name)
    y_type = outcome_type(prepped.y)
    t0 = time.time()
    try:
        res = decompose_mean(prepped, builder=make_builder(sweep, knob, y_type))
        err = ""
    except Exception as e:  # keep going; the row is flagged
        res = {k: np.nan for k in ["delta_y", "explained_1", "unexplained_1", "explained_0", "unexplained_0"]}
        res.update(n_1=int((prepped.pop == 1).sum()), n_0=int((prepped.pop == 0).sum()))
        err = repr(e)
    row = {"sweep": sweep, "setting": label, **cell, **res, "error": err,
           "n_features": int(prepped.features.shape[1]), "seconds": round(time.time() - t0, 2)}
    row["explained_flip"] = bool(res["explained_0"] * res["explained_1"] < 0)
    row["unexplained_flip"] = bool(res["unexplained_0"] * res["unexplained_1"] < 0)
    row["any_flip"] = row["explained_flip"] or row["unexplained_flip"]
    return row


def run(sweep: str, settings: list[tuple[str, object]], acs: pd.DataFrame, cells: pd.DataFrame):
    path = OUT_DIR / f"{sweep}_rows.csv"
    cached = pd.read_csv(path, keep_default_na=False) if path.exists() and not RECOMPUTE else None
    have = set(cached["setting"].astype(str)) if cached is not None else set()
    todo = [(lab, k) for lab, k in settings if lab not in have]
    print(f"[{sweep}] {len(have)} cached setting(s), {len(todo)} to fit on {len(cells)} cells", flush=True)
    # Slice each cell once (the parallel workers pickle only their own slice).
    slices = []
    for cell in cells.to_dict("records"):
        sub = acs[acs[cell["subset_name"]].astype(str) == cell["subset_value"]]
        slices.append((cell, sub))
    t0 = time.time()
    for label, knob in todo:
        rows = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(fit_cell)(sweep, label, knob, cell, sub) for cell, sub in slices)
        df = pd.DataFrame(rows)
        for y, g in df.groupby("y_name"):
            print(f"  {sweep} {label:<14} {y:<6} flips: any={int(g.any_flip.sum()):3d} "
                  f"explained={int(g.explained_flip.sum()):3d} unexplained={int(g.unexplained_flip.sum()):3d} "
                  f"errors={int((g.error != '').sum())}   [{time.time() - t0:6.0f}s]", flush=True)
        cached = df if cached is None else pd.concat([cached, df], ignore_index=True)
        cached.to_csv(path, index=False)   # checkpoint after every setting
    return cached


def main(sweeps: list[str]) -> None:
    acs = pd.read_parquet(here() / "census" / "temp" / "acs16_workforce.parquet").drop(columns=["naics_3"])
    cells = table_cells()
    print(f"{len(cells)} cells: {cells.y_name.value_counts().to_dict()}; N_JOBS={N_JOBS}", flush=True)
    grids = {
        "xgb_depth": [(f"depth {d}", d) for d in XGB_DEPTHS],
        "xgb_trees": [(f"trees {n}", n) for n in XGB_TREES_GRID],
        "nn_width": [(f"w={w} s={s}", (w, s)) for w in NN_WIDTHS for s in NN_SEEDS],
        "nn_depth": [(f"d={d} s={s}", (d, s)) for d in NN_DEPTHS for s in NN_SEEDS],
    }
    for sweep in sweeps:
        run(sweep, grids[sweep], acs, cells)


if __name__ == "__main__":
    main(sys.argv[1:] or ["xgb_depth", "xgb_trees", "nn_depth", "nn_width"])
