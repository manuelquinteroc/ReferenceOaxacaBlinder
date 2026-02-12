# ReferenceOaxacaBlinder

Replication code for  
**“Do covariates explain why these groups differ? The choice of reference group can reverse conclusions in the Oaxaca–Blinder decomposition.”**

This repository reproduces all empirical results, tables, and figures in the paper.

The notebook `code/icu_analysis.ipynb` reproduces the ICU application in **Section 3** (and Appendix C.1), including:

- Table 1  
- Table 4  
- Figures 1 and 5  

---

## Data

The ICU application uses the PhysioNet Challenge 2012 dataset (Set A), accessed via the Kaggle mirror:

https://www.kaggle.com/datasets/msafi04/predict-mortality-of-icu-patients-physionet

Download and extract the dataset. After extraction, place the files in the following structure:

```
project-root/
└── archive/
├── Outcomes-a.txt
└── set-a/
└── set-a/
├── 132539.txt
├── 132540.txt
├── ...
```

The notebook assumes this exact folder structure. The dataset is not included in this repository.
