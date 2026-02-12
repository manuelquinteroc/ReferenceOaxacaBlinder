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


