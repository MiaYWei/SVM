# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif,SelectPercentile
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.ensemble import BalancedBaggingClassifier
from collections import Counter
import pickle
import time

# Import the dataset
data = pd.read_csv('dataset\csv_result-Descriptors_Training.csv') 
# Convert 'P, N' into '1, 0'
data['class'] = data['class'].map({'P':1,'N':0})
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
print('Original dataset', X.shape, y.shape, Counter(y))

###################### Dataset Pre-processing ###########################
# Select features
sp = SelectPercentile(f_classif, percentile=30)
X = sp.fit_transform(X, y)
print(sp.get_support())

# Identify and remove outliers
Q1, Q3 = np.percentile(X, [25, 75])
IQR = Q3 - Q1
filtered_entries = ((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR))).any(axis=1) 
X = X[filtered_entries]
y = y[filtered_entries]
print('IQR', X.shape, y.shape)

###################### Split Dataset ###########################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
print('Training dataset', X_train.shape, Counter(y_train))
print('Test dataset:', X_test.shape, Counter(y_test))

###################### Training ###########################
# Identify outliers in the training dataset
Q1, Q3 = np.percentile(X_train, [25, 75])
IQR = Q3 - Q1
filtered_entries = ((X_train < (Q1 - 1.5 * IQR)) |(X_train > (Q3 + 1.5 * IQR))).any(axis=1) 
X_train = X_train[filtered_entries]
y_train = y_train[filtered_entries]

# Data Standardization
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Resample the imbalance dataset by using SVMSMOTE
model_smote = SVMSMOTE(sampling_strategy='auto', n_jobs =-1, random_state=42) 
X_train, y_train = model_smote.fit_sample(X_train, y_train) 
print('After oversampling', Counter(y_train))

# The procedure only removes noisy and ambiguous points along the class boundary  
undersample = EditedNearestNeighbours(n_neighbors=3)
X_train, y_train = undersample.fit_sample(X_train, y_train)
print('After undersampling', Counter(y_train))

start_time = time.time()
classifier = SVC(kernel='linear', gamma=2.825, C=19, class_weight='balanced', probability=True, 
                shrinking=False, cache_size=10000, verbose=True, random_state=42)

###################### Bagging ###########################
# ensemble = BalancedBaggingClassifier(base_estimator=classifier, n_estimators=5,
#                                     sampling_strategy='auto',
#                                     replacement=True,
#                                     random_state=42)
# ensemble.fit(X_train, y_train)

from imblearn.ensemble import RUSBoostClassifier
ensemble = RUSBoostClassifier(base_estimator=classifier, n_estimators=2, algorithm='SAMME.R', random_state=0)
ensemble.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))

###################### Test Model ###########################
y_pred = ensemble.predict(X_test) 

######################  ROC/AUC   ###########################
# Generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# Predict probabilities
svm_probs = ensemble.predict_proba(X_test)
# keep probabilities for the positive outcome only
svm_probs = svm_probs[:, 1]

# Calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
svm_auc = roc_auc_score(y_test, svm_probs)

# Summarize scores
print('SVM: ROC AUC=%.3f' % (svm_auc))
# Calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
svm_fpr, scm_tpr, _ = roc_curve(y_test, svm_probs)
# Plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(svm_fpr, scm_tpr, marker='.', label='SVM')
plt.title('Meta - ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

####################### PR Curve #####################
svm_precision, svm_recall, _ = precision_recall_curve(y_test, svm_probs)
# Plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(svm_recall, svm_precision, marker='.', label='SVM')
plt.title('Meta - PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# Max precision at recall at least 50% 
recall_re_greater50 = []
min_recall = 0
precision_recall_50 = []

for i in range(0, len(svm_recall)):
    if (svm_recall[i] >= 0.5) & (svm_precision[i] > 0):
        recall_re_greater50.append(svm_recall[i])
        plt.scatter(svm_recall[i], svm_precision[i], linewidths = 0, marker = 'X', color='green')
for i in range(0, len(recall_re_greater50)):
    min_recall = np.min(recall_re_greater50)
print('min_recall %3f' % min_recall)

for i in range(0, len(svm_recall)):
    if svm_recall[i] == min_recall:
        precision_recall_50.append(svm_precision[i])
print('Pr@Re50: %.3f +/- %.4f' % (np.mean(precision_recall_50), np.std(precision_recall_50)), '\n')

axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,0.35])
plt.axvline(x=0.5, color='green', linestyle='dashdot')
plt.show()

###################### Save Model ###########################
f = open('meta_bagging_e-5_outlier.pickle','wb')
pickle.dump(ensemble,f)
f.close()

###################### Evaluation ###########################
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('Confusion Matrix: TN =', tn, 'FP =', fp, 'FN =', fn, 'TP =', tp)
print('\n',classification_report(y_test,y_pred))

# Evaluate model
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores_accuracy = cross_val_score(ensemble, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1)
scores_precision = cross_val_score(ensemble, X_train, y_train, scoring='precision', cv=10, n_jobs=-1)
scores_recall = cross_val_score(ensemble, X_train, y_train, scoring='recall', cv=10, n_jobs=-1)
print('SVM Accuracy: %.3f +/- %.3f' % (np.mean(scores_accuracy), np.std(scores_accuracy)))
print('SVM Precision: %.3f +/- %.3f' % (np.mean(scores_precision), np.std(scores_precision)))
print('SVM Recall: %.3f +/- %.3f' % (np.mean(scores_recall), np.std(scores_recall)))