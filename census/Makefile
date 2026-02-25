all: temp out
all: temp/acs16_workforce.fst
all: temp/ob_fits.fst
all: temp/ob_boots.fst
all: out/flip_signif.csv
all: out/aligned_stats.csv

temp out:
	mkdir -p $@

temp/acs16_workforce.fst: 00_load-acs-16.R raw/ss16pusa.csv raw/ss16pusb.csv
	Rscript --vanilla --verbose $<
	
temp/ob_fits.fst: 01_fit-oaxaca-blinder.R temp/acs16_workforce.fst
	Rscript --vanilla --verbose $<
	
temp/ob_boots.fst: 02_bootstrap-flips.R temp/ob_fits.fst
	Rscript --vanilla --verbose $<

out/flip_signif.csv: 03_test-significance.R temp/ob_fits.fst temp/ob_boots.fst
	Rscript --vanilla --verbose $<

out/aligned_stats.csv: 04_aligned-slope-intercept.R temp/ob_fits.fst
	Rscript --vanilla --verbose $<