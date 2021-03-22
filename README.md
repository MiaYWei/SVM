# Project Objective

SYSC5405 Project is about protein methylation. (https://en.wikipedia.org/wiki/Protein_methylat ion)This is a type of post-translational modification that can alter a protein’s structure, stability, or functio n. 

The objective of the projetc is applying SVM machine learning algorithm to predict the imbanlance dataset.

Distinguishing between the protein windows that are (‘P’ in last column) or are not (‘N’ in last column) methylated. 

## Evaluation

1. Maximum achievable precision at a recall of at least 50% (Pr@Re50)

2. The correctness of Pr@Re50 prediction over the test dataset

## Repo Structure

        ├── data
        │   ├── csv_result-Descriptors_Calibration.csv
		│   └── csv_result-Descriptors_Training.csv
        ├── README.md
        ├── svm.py
        ├── svm_parameter.py
        ├── svm_kfold.py
        └── svm_roc.py