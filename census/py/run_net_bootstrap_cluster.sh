#!/usr/bin/env bash
# Neural-net leg of the census analysis (Tables 8 & 9): point estimates + B = 1000 bootstrap.
# CPU job -- meant for a many-core node (SuperCloud). About 20 CPU-days of MLP fits at B=1000:
# ~10 h on 48 cores, ~5 h on 96. Resumable: re-run the same command to continue.
#
# One-time setup on the node:
#   pip install -r census/py/requirements.txt
#   (data: census/py/data/acs16_workforce.parquet is shipped; the script copies it into place)
#
# Usage:            bash census/py/run_net_bootstrap_cluster.sh
# SLURM (example):  sbatch -c 48 --time=24:00:00 --wrap "bash census/py/run_net_bootstrap_cluster.sh"
#
# Outputs (suffix _net keeps them separate from the CPU run of the other models):
#   census/temp/nonlinear_fits_net.parquet, census/temp/nonlinear_boots_net/  (per-cell chunks)
#   census/out_py/flip_counts_net.csv   <- the Neural net rows, Tables 8/9 format
set -euo pipefail
cd "$(dirname "$0")/../.."                                   # repo root

mkdir -p census/temp
[ -f census/temp/acs16_workforce.parquet ] || cp census/py/data/acs16_workforce.parquet census/temp/

NCPU="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one thread per worker
export OBD_ALGOS='{"pincp": ["net"], "hicov": ["net"]}'
export OBD_OUT_SUFFIX=_net
export OBD_N_JOBS="$NCPU"                                    # parallel subsets (step 01)
export OBD_INNER_JOBS="$NCPU"                                # parallel bootstrap replicates (step 02)
export OBD_B="${OBD_B:-1000}"                                # paper: 1000
export OBD_BOOT_ALGOS=net

python census/py/01_fit_decomposition.py     # ~10 min on 8 cores: point estimates on all 462 cells
python census/py/02_bootstrap_flips.py       # the long part; resumable per cell
python census/py/03_count_flips.py           # -> census/out_py/flip_counts_net.csv
