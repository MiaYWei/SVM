# Import the libraries
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# Import the dataset
data = pd.read_csv('csv_result-Descriptors_Calibration.csv') 

# Convert 'P, N' into '1, 0'
data['class'] = data['class'].map({'P':1,'N':0})

X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

print('Original dataset', X.shape, y.shape)
print(Counter(y))

###################### Dataset Pre-processing ###########################
# Identify and remove outliers
z_scores = stats.zscore(X)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
X = X[filtered_entries]
y = y[filtered_entries]
print('\nRemove outliers', X.shape, y.shape)

# Select features
features_selected = ['ECI_IB_4_N1','Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Gs(U)_NO_ALR_SI71','ISA_NO_NPR_S','IP_NO_PLR_S', 'ECI_NO_PCR_CV']

###################### Split Dataset ###########################
# Split dataset into train and test sets
train, test = train_test_split(data, test_size = 0.2, random_state = 100)# in this our main data is splitted into train and test
X_train = train[features_selected]
y_train=train['class']
X_test= test[features_selected]
y_test =test['class']

###################### Training Dataset ###########################
# Data Standardization
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Resample the imbalance dataset by using SMOTE
model_smote = SVMSMOTE(sampling_strategy='auto', n_jobs = -1, random_state = 42) 
X_train, y_train = model_smote.fit_sample(X_train, y_train) 
print('Training set', X_train.shape, y_train.shape)
print('After oversampling', Counter(y_train))

# The procedure only removes noisy and ambiguous points along the class boundary
from imblearn.under_sampling import EditedNearestNeighbours
undersample = EditedNearestNeighbours(n_neighbors=3)
X_train, y_train = undersample.fit_sample(X_train, y_train)
print('After undersampling', Counter(y_train))

###################### hyperparameters ###########################
# automatic svm hyperparameter tuning using skopt for the ionosphere dataset
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV

# define search space
params = dict()
params['C'] = (1e-2, 100.0, 'log-uniform')
params['gamma'] = (1e-2, 100.0, 'log-uniform')
params['kernel'] = ['linear']
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the search
search = BayesSearchCV(estimator=SVC(shrinking = False, cache_size = 10000, verbose = True, random_state = 42), search_spaces=params, n_jobs=-1, cv=cv, iid=False)
# perform the search
search.fit(X_train, y_train)
# report the best result
print(search.best_score_)
print(search.best_params_)