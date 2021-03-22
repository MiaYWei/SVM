# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_roc_curve, roc_curve, auc, precision_recall_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter

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
z_scores = stats.zscore(X)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
X = X[filtered_entries]
y = y[filtered_entries]
print('\nRemove outliers', X.shape, y.shape)

# Select features
features_selected = ['ECI_IB_4_N1','Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Gs(U)_NO_ALR_SI71','ISA_NO_NPR_S','IP_NO_PLR_S', 'ECI_NO_PCR_CV']
X = X[features_selected]

###################### Training Dataset ###########################
# Data Standardization
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Resample the imbalance dataset by using SMOTE
model_smote = SVMSMOTE(sampling_strategy='auto', n_jobs =-1, random_state=42) 
X, y = model_smote.fit_sample(X, y) 
print('Training set', X.shape, y.shape)
print('After oversampling', Counter(y))

# The procedure only removes noisy and ambiguous points along the class boundary
undersample = EditedNearestNeighbours(n_neighbors=3)
X, y = undersample.fit_sample(X, y)
print('After undersampling', Counter(y))

###################### Dataset Visualization ###########################
# visualize Methylated class
sns.countplot(y,label="Count")
plt.show()

########################## ROC & cross fold ############
# Run classifier with cross-validation and plot ROC curves
from sklearn.ensemble import BaggingClassifier
classifier = SVC(kernel='linear', gamma=0.665, C=11.73, class_weight='balanced', probability=True, shrinking=False, cache_size=10000, verbose=True, random_state=42)
ensemble = BaggingClassifier(base_estimator=classifier, n_estimators=31, random_state=42)

cv = StratifiedKFold(n_splits=10)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    if i==5:
        #classifier.fit(X[train], y[train])
        ensemble.fit(X[train], y[train])
        viz = plot_roc_curve(ensemble, X[test], y[test], name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        X_test = X[test]
        y_test = y[test]

###################### Test Dataset ###########################
# Predict the Test set results
y_pred = ensemble.predict(X_test) 
# Predict probabilities
svm_probs = ensemble.predict_proba(X_test)
# keep probabilities for the positive outcome only
svm_probs = svm_probs[:, 1]

####################### PR Curve #####################
svm_precision, svm_recall, _ = precision_recall_curve(y_test, svm_probs)
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(svm_recall, svm_precision, marker='.', label='SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Max precision a sensitivity of 50% 
precision_recall_50 = []
for i in range(0, len(svm_recall)):
    if svm_recall[i] >= 0.5:
        precision_recall_50.append(svm_precision[i])
        plt.scatter(svm_recall[i], svm_precision[i], linewidths = 0, marker = 'X', color='red')
print('Max Pr@Re50', max(precision_recall_50))

######################       ROC/AUC   ###########################
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# Predict probabilities
svm_probs = classifier.predict_proba(X_test)
# keep probabilities for the positive outcome only
svm_probs = svm_probs[:, 1]

# Calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
svm_auc = roc_auc_score(y_test, svm_probs)

# summarize scores
print('SVM: ROC AUC=%.3f' % (svm_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
svm_fpr, scm_tpr, _ = roc_curve(y_test, svm_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(svm_fpr, scm_tpr, marker='.', label='SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()