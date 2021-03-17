# Import the libraries
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.metrics import roc_curve, auc

# Import the dataset
dataset = pd.read_csv('csv_result-Descriptors_Calibration.csv') 
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

# # Taking care of missing data
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X) 
# X = imputer.transform(X)

# Convert 'P, N' into '1, 0'
methylated = np.zeros(len(y))
for i in range(len(y)):
    if y[i] == 'P':
        methylated[i] = 1
    if y[i] == 'N':
        methylated[i] = 0

y = methylated
print('Original dataset', X.shape, y.shape)
print(Counter(y))

# Identify outliers in the training dataset
z_scores = stats.zscore(X)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
X = X[filtered_entries]
y = y[filtered_entries]
print('\nRemove outliers', X.shape, y.shape)

# Select features
selector = SelectKBest(k=5)
X = selector.fit_transform(X,y)
print('\nFeature selection', X.shape, y.shape)
#print(selector.scores_)
#print(selector.pvalues_)

###################### Split Dataset ###########################
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print('\nTraining set',X_train.shape, y_train.shape)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

###################### Training Dataset ###########################
# Standarize all features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Resample the imbalance dataset by using SMOTEENN
model_smote = SMOTE(random_state = 42) 
X_train, y_train = model_smote.fit_sample(X_train, y_train) 
print('Training set', X_train.shape, y_train.shape)
counter = Counter(y_train)
print('After oversampling',Counter(y_train))

# Resample the imbalance dataset by undersampling
# from imblearn.under_sampling import TomekLinks
# from imblearn.under_sampling import CondensedNearestNeighbour
# from imblearn.under_sampling import NearMiss
# from imblearn.under_sampling import ClusterCentroids

# counter = Counter(y_train)
# print('Before undersampling', counter)

# # define the undersampling method
# #undersample = TomekLinks()
# #undersample = CondensedNearestNeighbour(n_neighbors=1)
# #undersample = NearMiss(version=3, n_neighbors_ver3=3)
# #undersample = ClusterCentroids(random_state=0)

# # transform the dataset
# #X_train, y_train = undersample.fit_resample(X_train, y_train)

# # summarize the new class distribution
# counter = Counter(y_train)
# print('After undersampling', counter)

# Train the SVM model on the Training set
#classifier = SVC(kernel='linear', class_weight = 'balanced', C=0.1, random_state = 42)
#classifier = SVC(kernel='linear', decision_function_shape = 'ovo', shrinking = False, cache_size = 10000, verbose = True, max_iter = -1, random_state = 0)
classifier = SVC(kernel='rbf', class_weight='balanced', decision_function_shape = 'ovo', shrinking = False, probability=True, cache_size = 10000, verbose = True, random_state = 0)
classifier.fit(X_train, y_train)

###################### Test Dataset ###########################
# Predicte the Test set results
print('\nTest data', X_test.shape)
y_pred = classifier.predict(X_test) 

# predict probabilities
svm_probs = classifier.predict_proba(X_test)
# keep probabilities for the positive outcome only
svm_probs = svm_probs[:, 1]

######################       ROC/AUC   ###########################
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, svm_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('SVM: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
svm_fpr, scm_tpr, _ = roc_curve(y_test, svm_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(svm_fpr, scm_tpr, marker='.', label='SVM')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

# ROC/AUC
FP_rate, TP_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(FP_rate, TP_rate)
print('roc_auc', roc_auc)

###################### Evaluation Dataset ###########################
# Evaluate predictions
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('TN =', tn, 'FP =', fp, 'FN =', fn, 'TP =', tp)
print('ACC =', (tp+tn)/(tp+tn+fn+fp))
print(classification_report(y_test,y_pred))