# ReferenceOaxacaBlinder

Replication code for  
**“Do covariates explain why these groups differ? The choice of reference group can reverse conclusions in the Oaxaca–Blinder decomposition.”**

This repository reproduces all empirical results, tables, and figures in the paper.

## Real-data example (ICU application)

The notebook `Real-data example/icu_analysis.ipynb` reproduces the ICU application in Section 3 and Appendix C.1 of the paper (Tables 1 and 4; Figures 1 and 5).

### Data

The ICU example uses the PhysioNet Challenge 2012 dataset (Set A), accessed via the Kaggle mirror:

https://www.kaggle.com/datasets/msafi04/predict-mortality-of-icu-patients-physionet

Download and extract the dataset. Then place the files inside: 
Real-data example/archive/

so that the final structure is:
```
Real-data example/
├── icu_analysis.ipynb
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

The R scripts in `census` produce the U.S. labor force analyses referenced in Section 5.3, 5.4, and C.2, including Table 8.

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
* `census/out/flip_signif.csv`: summary statistics for section 5.3 and C.2 (Table 8).
* `census/out/aligned_stats.csv`: summary statistics for checking assumption 5.1 in the U.S. labor force example, as quoted in section 5.4.

## Sign Flip Probabilities

The R scripts in `prob_of_signflip` produce Figures 3 and 4. The file `prob_of_signflip/Makefile` configures these scripts. 

1. If you have not done so for the census data example, run `R --no-save --no-restore` from the project root. This will automatically install the `renv` package, which manages the other packages used in this project. From within this R session execute the command `renv::restore()` and type `Y` when prompted to install the remaining required packages.

2. Run the analysis by navigating to `prob_of_signflip` from the command line and executing the command `make`. The analysis is configured by `prob_of_signflip/Makefile` and will produce:
* `prob_of_signflip/out/standardized.pdf`: Figure 3
* `prob_of_signflip/out/raw.pdf`: Figure 4