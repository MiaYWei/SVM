# Import the libraries
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SVMSMOTE
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.feature_selection import f_classif,SelectPercentile

# Import the dataset
data = pd.read_csv('data\csv_result-Descriptors_Calibration.csv') 

# Convert 'P, N' into '1, 0'
data['class'] = data['class'].map({'P':1,'N':0})

X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

print('Original dataset', X.shape, y.shape)
print(Counter(y))

###################### Dataset Pre-processing ###########################
# Identify and remove outliers
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
filtered_entries = ((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR))).any(axis=1) 
X = X[filtered_entries]
y = y[filtered_entries]
print('IQR', X.shape, y.shape)

# Select features
X_new = SelectPercentile(f_classif, percentile=30).fit_transform(X, y)

###################### Split Dataset ###########################
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, random_state=101)

###################### Data Standardization ###########################
# identify outliers in the training dataset
Q1, Q3 = np.percentile(X_train, [25, 75])
IQR = Q3 - Q1
filtered_entries = ((X_train < (Q1 - 1.5 * IQR)) |(X_train > (Q3 + 1.5 * IQR))).any(axis=1) 
X_train = X_train[filtered_entries]
y_train = y_train[filtered_entries]

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Resample the imbalance dataset by using SMOTE
model_smote = SVMSMOTE(sampling_strategy='auto', n_jobs =-1, random_state=42) 
X_train, y_train = model_smote.fit_sample(X_train, y_train) 
print('Training set', X_train.shape, y_train.shape)
print('After oversampling', Counter(y_train))

# The procedure only removes noisy and ambiguous points along the class boundary
from imblearn.under_sampling import EditedNearestNeighbours
undersample = EditedNearestNeighbours(n_neighbors=5)
X_train, y_train = undersample.fit_sample(X_train, y_train)
print('After undersampling', Counter(y_train))

# Train the SVM model on the training set; # C = 50, 70, and 100 has the same result.Max Pr@Re50 0.11159
#classifier = SVC(kernel='linear', gamma=0.665, C=11.73, class_weight='balanced', probability=True, shrinking=False, cache_size=10000, verbose=True, random_state=42)
classifier = SVC(kernel='linear', gamma=2.825, C=19, class_weight='balanced', probability=True, shrinking=False, cache_size=10000, verbose=True, random_state=42)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('SVM Accuracy: %.3f +/- %.3f' % (np.mean(n_scores), np.std(n_scores)))
print('SVM Accuracy - max: %.3f' % np.max(n_scores))