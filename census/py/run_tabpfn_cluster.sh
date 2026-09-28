#!/usr/bin/env bash
# TabPFN leg of the census analysis (Tables 8 & 9), meant for a GPU node.
# Batch-job equivalent of census_tabpfn_cluster.ipynb (same phases, same outputs); prefer the
# notebook for an interactive session, this script for SLURM/nohup.
#
# Produces census/temp/nonlinear_fits_tabpfn.parquet (+ boots, + out_py/flip_counts_tabpfn.csv),
# kept separate from the CPU run via OBD_OUT_SUFFIX so the two never clobber each other.
#
# One-time setup on the node:
#   pip install -r census/py/requirements.txt tabpfn==7.1.1
#   # weights (TabPFN >= 7 gates the download behind its license): accept the license at
#   # https://ux.priorlabs.ai, then `export TABPFN_TOKEN=<api key>` so both the classifier and
#   # the regressor checkpoints download on first fit. On an air-gapped node copy the two v2.6
#   # .ckpt files over and set OBD_TABPFN_CLF_PATH / OBD_TABPFN_REG_PATH instead.
#   # Data: census/raw/ss16pusa.csv + ss16pusb.csv (see README), then `python census/py/00_load_acs.py`.
#
# Usage:  bash census/py/run_tabpfn_cluster.sh            # full grid, point estimates only
#         OBD_B=200 bash census/py/run_tabpfn_cluster.sh  # ...plus a B=200 bootstrap of the flip cells
set -euo pipefail
cd "$(dirname "$0")/../.."                          # repo root

mkdir -p census/temp
[ -f census/temp/acs16_workforce.parquet ] || cp census/py/data/acs16_workforce.parquet census/temp/

export OBD_ALGOS='{"pincp": ["tabpfn"], "hicov": ["tabpfn"]}'
export OBD_OUT_SUFFIX=_tabpfn
export OBD_TABPFN_DEVICE="${OBD_TABPFN_DEVICE:-auto}"       # auto = cuda if present, else cpu
export OBD_TABPFN_MAX_ROWS="${OBD_TABPFN_MAX_ROWS:-10000}"  # training rows per group (predicts on all)
export OBD_N_JOBS="${OBD_N_JOBS:-1}"                         # 1 per GPU; raise only on CPU-only nodes
export OBD_INNER_JOBS="${OBD_INNER_JOBS:-1}"

python census/py/01_fit_decomposition.py
if [[ -n "${OBD_B:-}" ]]; then
  python census/py/02_bootstrap_flips.py                     # resumable; rerun to continue
fi
python census/py/03_count_flips.py                           # works with or without bootstraps
