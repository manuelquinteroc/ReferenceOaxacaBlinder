all: temp out
all: temp/acs16_workforce.fst
all: temp/nonlinear_fits.fst
all: temp/nonlinear_boots.fst
all: out/flip_counts.csv
all: out/aligned_stats.csv

temp out temp/nonlinear_boots:
	mkdir -p $@

temp/acs16_workforce.fst: 00_load-acs-16.R raw/ss16pusa.csv raw/ss16pusb.csv
	cd .. && Rscript --no-save --no-restore --verbose census/$<
	
temp/nonlinear_fits.fst: 01_fit-nonlinear-decomposition.R	temp/acs16_workforce.fst
	cd .. && Rscript --no-save --no-restore --verbose census/$<
	
temp/nonlinear_boots.fst: 02_bootstrap-nonlinear-decomposition.R temp/nonlinear_fits.fst
	cd .. && Rscript --no-save --no-restore --verbose census/$<	

out/flip_counts.csv: 03_count-flips.R temp/nonlinear_fits.fst temp/nonlinear_boots.fst
	cd .. && Rscript --no-save --no-restore --verbose census/$<

out/aligned_stats.csv: 04_aligned-slope-intercept.R temp/acs16_workforce.fst
	cd .. && Rscript --no-save --no-restore --verbose census/$<