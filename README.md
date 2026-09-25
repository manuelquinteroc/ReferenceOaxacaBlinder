# ReferenceOaxacaBlinder

Replication code for  
**“Do covariates explain why these groups differ? The choice of reference group can reverse conclusions in the Oaxaca–Blinder decomposition.”**

This repository reproduces all empirical results, tables, and figures in the paper.

## Real-data example (ICU application)

The notebooks in `Real-data example/` reproduce the ICU application in Section 3 and Appendix C.1 of the paper (Table 1 and Appendix Tables 3, 4, 5; Figure 1 (Left) and Appendix Figure 2). The pipeline runs in three steps:

1. `construct_ICU_data.ipynb` — builds the clean analysis dataset (`ICU_clean.csv`) from the raw PhysioNet files.
2. `icu_139_final_models.ipynb` — runs linear, logistic, neural net, and XGBoost on 139 clinical subsets; produces the non-TabPFN rows of Table 1 and Appendix Tables 3, 4, 5, plus Figure 1 (Left) and Appendix Figure 2.
3. `icu_139_tabpfn_local.ipynb` — runs TabPFN locally (CPU or GPU); completes the TabPFN row of Table 1 and Appendix Tables 3, 4, 5.

Install Python dependencies with `pip install -r "Real-data example/requirements.txt"`.

**Model complexity (supplementary).** `icu_139_complexity_sweep.ipynb` re-runs the Table 1
decomposition on the same 139 subsets while varying one hyperparameter at a time (XGBoost
depth 1–10 and number of trees 10–10,000; neural-net width at depth 3 and depth at width 32,
three seeds each), everything else at the paper's values. Fits are cached per setting in
`complexity_sweep/`; figures go to `Figures/complexity_{xgb_depth,xgb_trees,nn_width,nn_depth}.pdf`.

### Data

The ICU example uses the PhysioNet Challenge 2012 dataset (Set A), accessed via the Kaggle mirror:

https://www.kaggle.com/datasets/msafi04/predict-mortality-of-icu-patients-physionet

Download and extract the dataset. Then place the files inside: 
Real-data example/archive/

so that the final structure is:
```
Real-data example/
├── construct_ICU_data.ipynb
├── icu_139_final_models.ipynb
├── icu_139_tabpfn_local.ipynb
├── requirements.txt
├── archive/
│   ├── Outcomes-a.txt
│   └── set-a/
│       └── set-a/
│           ├── 132539.txt
│           ├── 132540.txt
│           └── ...
├── Figures/
```

The dataset is not included in this repository.

## Census Data Example

The R scripts in `census` produce the U.S. labor force analyses referenced in Section 3 and Appendix B.2, including Tables 8 and 9.

1. From the project root, run `R --no-save --no-restore`. This will automatically install the `renv` package, which manages the other packages used in this project. From within this R session execute the command `renv::restore()` and type `Y` when prompted to install the remaining required packages.

2. Download the 2016 American Community Survey (ACS) 1-Year Public Use Microdata Sample (PUMS). As of Feb 26, 2025, the data are available at: https://www2.census.gov/programs-surveys/acs/data/pums/2016/1-Year/csv_pus.zip.

3. Unzip the downloaded archive, create the subdirectory `census/raw`, and place `ss16pusa.csv` and `ss16pusb.csv` inside `census/raw/`. The file structure should be:
```
census/
├── raw/
│   ├── ss16pusa.csv
│   ├── ss16pusb.csv
...
```

4. Run the analysis by navigating to `census` from the command line and executing the command `make`. The analysis is configured by `census/Makefile` and will produce:
* `census/out/flip_counts.csv`: statistics for Tables 8 and 9.
* `census/out/aligned_stats.csv`: summary statistics for checking assumption 4.1 in the U.S. labor force example, as quoted in section 4.3.

## Census Data Example — Python pipeline

`census/py/` is a Python port of the R scripts above, sharing `obd_engine/` with the ICU
notebooks so the census tables use the *same* model builders (and hyperparameters) as Table 1.
It writes to `census/temp/` and `census/out_py/` (`flip_counts.csv` feeds Tables 8 and 9).

Download the ACS data as described above, then from the repo root (or `make PY=python3` in
`census/py/`):

```
pip install -r census/py/requirements.txt
python3 census/py/00_load_acs.py            # -> temp/acs16_workforce.parquet  (871,849 rows)
python3 census/py/01_fit_decomposition.py   # point estimates, full design grid
python3 census/py/02_bootstrap_flips.py     # B=1000 bootstrap of the sign-flip cells (resumable)
python3 census/py/03_count_flips.py         # -> out_py/flip_counts.csv
python3 census/py/04_aligned_slope.py       # -> out_py/aligned_stats.csv
```

**TabPFN** (GPU): open `census/py/census_tabpfn_cluster.ipynb` — the census counterpart of the ICU
TabPFN notebook: config cell at the top, three resumable phases (point estimates, B = 1000
bootstrap of the flip cells, flip counts), same engine as the CPU models. `census/py/data/`
ships the analysis table so the node does not need the 3 GB raw download.

Models: `ols`, `glm` (sklearn logistic with its default L2 penalty — the ICU notebook's model),
`glm_r` (unpenalized logistic, the R `glm` the paper's census tables used), `gbt` (XGBoost),
`net` (MLP), and `tabpfn` (registered but not in the default grid — see
`census/py/run_tabpfn_cluster.sh`).
Every script reads optional environment overrides so quick tests and partial runs need no
code edits:

| Variable | Default | Meaning |
|---|---|---|
| `OBD_SUBSET_COLS` | `st,naics_2` | subset dimensions |
| `OBD_POP_NAMES` | `sex_female,race_bw,immigrant` | group definitions |
| `OBD_ALGOS` | `{"pincp": [ols,gbt,net], "hicov": [ols,glm,glm_r,gbt,net]}` (JSON) | algorithms per outcome |
| `OBD_EXTRA_COVARIATES` | 0 | 1 = also use disability / self-employed / government / veteran as covariates (not in the paper's R pipeline) |
| `OBD_OUT_SUFFIX` | *(empty)* | suffix on every temp/out file, e.g. `_test`, `_tabpfn` |
| `OBD_N_JOBS` | 8 | parallel subsets in 01/04 |
| `OBD_B`, `OBD_INNER_JOBS` | 1000, 6 | bootstrap replicates and parallel replicates in 02 |
| `OBD_BOOT_ALGOS` | *(all)* | comma list restricting which algorithms 02 bootstraps |
| `OBD_TABPFN_DEVICE`, `OBD_TABPFN_CLF_PATH`, `OBD_TABPFN_REG_PATH`, `OBD_TABPFN_MAX_ROWS` | auto, auto, auto, 10000 | TabPFN device / classifier and regressor weights / training-row cap (`TABPFN_TOKEN` needed to download weights) |

A quick end-to-end smoke test (about 30 s):

```
OBD_SUBSET_COLS=st OBD_POP_NAMES=sex_female OBD_ALGOS='{"pincp": ["gbt"]}' \
OBD_OUT_SUFFIX=_test OBD_B=20 python3 census/py/01_fit_decomposition.py \
  && OBD_OUT_SUFFIX=_test OBD_B=20 python3 census/py/02_bootstrap_flips.py \
  && OBD_OUT_SUFFIX=_test python3 census/py/03_count_flips.py
```

**Fidelity to the R pipeline.** Two things matter for reproducing Tables 8–9 with this port:

1. *Rank deficiency.* Subgroups are small (n ≈ 50–200) while the design has ~150 columns, so
   exact collinearity is routine. R's `lm.fit`/`glm.fit` alias such columns (drop them);
   a pseudo-inverse spreads the weight across them instead. Both give the same in-sample fit
   but different predictions on the *other* group — which is what the decomposition uses.
   With the pseudo-inverse the OLS sign-flip count for log income came out at 101 against the
   paper's 35; `obd_engine.builders.aliased_free_columns` now reproduces R's column
   selection (33 with the paper's feature set).
2. *Feature set.* The R loader has 11 columns; the Python loader adds four covariates
   (`EXTRA_COVARIATES` in `obd_engine/features.py`). They are excluded by default and
   enabled with `OBD_EXTRA_COVARIATES=1`.

**Model complexity (supplementary).** `census/py/05_complexity_sweep.py` runs the same four
sweeps as the ICU notebook on the 217 + 183 cells of Tables 8–9 (neural-net widths up to 256;
resumable, cached per setting in `census/complexity_sweep/`), and
`census/py/census_complexity_sweep.ipynb` draws the figures into `census/out_py/figures/`.

`flip_counts*.csv` carries an extra `n_with_se` column: the number of flip cells that have
bootstrap standard errors. When it is below `is_flip`, the `reject_*` counts for that row are
lower bounds (the paper reports such rows with "-").

## Sign Flip Probabilities

The R scripts in `prob_of_signflip` produce Figure 1 (Right) and Appendix Figure 4. The file `prob_of_signflip/Makefile` configures these scripts. 

1. If you have not done so for the census data example, run `R --no-save --no-restore` from the project root. This will automatically install the `renv` package, which manages the other packages used in this project. From within this R session execute the command `renv::restore()` and type `Y` when prompted to install the remaining required packages.

2. Run the analysis by navigating to `prob_of_signflip` from the command line and executing the command `make`. The analysis is configured by `prob_of_signflip/Makefile` and will produce:
* `prob_of_signflip/out/standardized.pdf`: Figure 1 (Right)
* `prob_of_signflip/out/raw.pdf`: Appendix Figure 4