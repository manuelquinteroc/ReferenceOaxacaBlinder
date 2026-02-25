all: out
all: out/standardized.pdf
all: out/raw.pdf

out:
	mkdir -p $@

out/standardized.pdf: 00_flips-standardized.R
	cd .. && Rscript --no-save --no-restore --verbose sign_flip_probability/$<

out/raw.pdf: 01_flips-raw.R
	cd .. && Rscript --no-save --no-restore --verbose sign_flip_probability/$<