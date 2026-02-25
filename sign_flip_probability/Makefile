all: out
all: out/raw.pdf
all: out/standardized.pdf

out:
	mkdir -p $@

out/raw.pdf: 00_flips-standardized.R
	Rscript --vanilla --verbose $<

out/standardized.pdf: 01_flips-raw.R
	Rscript --vanilla --verbose $<