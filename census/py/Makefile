# Python census pipeline. Mirrors census/Makefile.
# Run from this directory (`cd census/py && make`); scripts resolve the repo root via
# pyprojroot, but recipes invoke from the root for parity with the R Makefile.

PY   := python
ROOT := $(shell cd ../.. && pwd)
TEMP := $(ROOT)/census/temp
OUT  := $(ROOT)/census/out_py

all: $(TEMP)/acs16_workforce.parquet \
     $(TEMP)/nonlinear_fits.parquet \
     $(TEMP)/nonlinear_boots.parquet \
     $(OUT)/flip_counts.csv \
     $(OUT)/aligned_stats.csv

$(TEMP)/acs16_workforce.parquet: 00_load_acs.py $(ROOT)/census/raw/ss16pusa.csv $(ROOT)/census/raw/ss16pusb.csv
	cd $(ROOT) && $(PY) census/py/00_load_acs.py

$(TEMP)/nonlinear_fits.parquet: 01_fit_decomposition.py $(TEMP)/acs16_workforce.parquet
	cd $(ROOT) && $(PY) census/py/01_fit_decomposition.py

$(TEMP)/nonlinear_boots.parquet: 02_bootstrap_flips.py $(TEMP)/nonlinear_fits.parquet
	cd $(ROOT) && $(PY) census/py/02_bootstrap_flips.py

$(OUT)/flip_counts.csv: 03_count_flips.py $(TEMP)/nonlinear_fits.parquet $(TEMP)/nonlinear_boots.parquet
	cd $(ROOT) && $(PY) census/py/03_count_flips.py

$(OUT)/aligned_stats.csv: 04_aligned_slope.py $(TEMP)/nonlinear_fits.parquet $(TEMP)/acs16_workforce.parquet
	cd $(ROOT) && $(PY) census/py/04_aligned_slope.py

.PHONY: all
