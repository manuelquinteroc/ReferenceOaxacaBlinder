"""Why does the draft's lightgbm row differ from XGBoost? Attribution across four variants.

    lgb_r      : the paper's R procedure (lr 0.25, 7 leaves, trees by 3-fold CV + early stopping)
    lgb_lr005  : same CV procedure, learning rate 0.05 (isolates the learning rate)
    lgb_fixed  : lightgbm with XGBoost's fixed settings (250 trees, lr 0.05, 8 leaves ~ depth 3)
    gbt        : XGBoost as in Tables 8-9 (250 trees, depth 3, lr 0.05)

Point-estimate flip counts on every cell, both outcomes. Run from the repo root.
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import obd_engine.builders as B
from obd_engine import decompose_from_data

class LGB_lr005(B.LGBMRLike): LR = 0.05
class LGB_fixed(B.LGBMRLike):
    def fit(self, X, y):
        import lightgbm as lgb
        X = np.asarray(X, float); y = np.asarray(y, float)
        if self.binary and np.unique(y).size < 2: self.const_ = float(y[0]); return self
        params = {"objective": "binary" if self.binary else "regression", "learning_rate": 0.05,
                  "num_leaves": 8, "verbose": -1, "num_threads": 1, "seed": 1}
        self.booster_ = lgb.train(params, lgb.Dataset(X, y), num_boost_round=250); return self
B._REGISTRY["lgb_lr005"] = {"binary": lambda: LGB_lr005(True), "continuous": lambda: LGB_lr005(False)}
B._REGISTRY["lgb_fixed"] = {"binary": lambda: LGB_fixed(True), "continuous": lambda: LGB_fixed(False)}

acs = pd.read_parquet("census/temp/acs16_workforce.parquet").drop(columns=["naics_3"])
subsets = acs[["st", "naics_2"]].astype(str).melt(var_name="subset_name", value_name="subset_value").drop_duplicates()
rows = []
for algo in ["lgb_r", "lgb_lr005", "lgb_fixed", "gbt"]:
    for y in ["hicov", "pincp"]:
        t = time.time()
        def task(sn, sv):
            sub = acs[acs[sn].astype(str) == sv]
            return [dict(pop=p, **decompose_from_data(sub, y, p, algo)) for p in ("sex_female", "race_bw", "immigrant")]
        out = Parallel(n_jobs=6, backend="threading")(delayed(task)(r.subset_name, r.subset_value) for r in subsets.itertuples())
        d = pd.DataFrame([x for c in out for x in c]); d = d[(d.n_0 > 50) & (d.n_1 > 50) & (d.delta_y.abs() > 0.01)]
        e = d.explained_0 * d.explained_1 < 0; u = d.unexplained_0 * d.unexplained_1 < 0
        rows.append(dict(algo=algo, y_name=y, n_fit=len(d), either=int((e | u).sum()), explained=int(e.sum()), unexplained=int(u.sum()), nan_cells=int(d.explained_0.isna().sum()), seconds=round(time.time() - t)))
        print(rows[-1], flush=True)
res = pd.DataFrame(rows); res.to_csv("census/out_py/lgb_vs_xgb_attribution.csv", index=False)
print("\nPaper draft: pincp lightgbm 29/23/6 ; hicov lightgbm 97/23/6 (inconsistent row)")
print(res.to_string(index=False))
