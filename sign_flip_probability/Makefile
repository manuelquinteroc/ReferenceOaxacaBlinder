all: out
all: out/raw.pdf
all: out/standardized.pdf

out:
	mkdir -p $@

out/standardized.pdf: 00_flips-standardized.R
	Rscript --vanilla --verbose $<

out/raw.pdf: 01_flips-raw.R
	Rscript --vanilla --verbose $<