"""Hypothesis for the draft's lightgbm rows (97/73/34 insurance, 29/23/6 income):
R's wrapper let params$num_iterations = 10000 override nrounds = best_iter, so the final
models were 10,000-tree, lr-0.25 boosters. Reproduce that and count flips."""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import obd_engine.builders as B
from obd_engine import decompose_from_data

class LGB10k(B.LGBMRLike):
    def fit(self, X, y):
        import lightgbm as lgb
        X = np.asarray(X, float); y = np.asarray(y, float)
        if self.binary and np.unique(y).size < 2: self.const_ = float(y[0]); return self
        params = {"objective": "binary" if self.binary else "regression", "learning_rate": 0.25,
                  "num_leaves": 7, "verbose": -1, "num_threads": 1, "seed": 1}
        self.booster_ = lgb.train(params, lgb.Dataset(X, y), num_boost_round=10_000); return self
B._REGISTRY["lgb_10k"] = {"binary": lambda: LGB10k(True), "continuous": lambda: LGB10k(False)}

acs = pd.read_parquet("census/temp/acs16_workforce.parquet").drop(columns=["naics_3"])
subsets = acs[["st", "naics_2"]].astype(str).melt(var_name="subset_name", value_name="subset_value").drop_duplicates()
for y, draft in [("hicov", "97/73/34"), ("pincp", "29/23/6")]:
    t = time.time()
    def task(sn, sv):
        sub = acs[acs[sn].astype(str) == sv]
        return [dict(pop=p, **decompose_from_data(sub, y, p, "lgb_10k")) for p in ("sex_female", "race_bw", "immigrant")]
    out = Parallel(n_jobs=4, backend="threading")(delayed(task)(r.subset_name, r.subset_value) for r in subsets.itertuples())
    d = pd.DataFrame([x for c in out for x in c]); d = d[(d.n_0 > 50) & (d.n_1 > 50) & (d.delta_y.abs() > 0.01)]
    e = d.explained_0 * d.explained_1 < 0; u = d.unexplained_0 * d.unexplained_1 < 0
    print(f"{y}: lightgbm 10,000 trees lr 0.25 7 leaves -> either={int((e|u).sum())} explained={int(e.sum())} unexplained={int(u.sum())}   (draft {draft})  [{time.time()-t:.0f}s]", flush=True)
