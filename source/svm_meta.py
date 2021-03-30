# Import the libraries
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif,SelectPercentile
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Import the dataset
data = pd.read_csv('dataset\csv_result-Descriptors_Calibration.csv') 

# Convert 'P, N' into '1, 0'
data['class'] = data['class'].map({'P':1,'N':0})
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

print('Original dataset', X.shape, y.shape, Counter(y))

###################### Dataset Pre-processing ###########################
#Identify and remove outliers
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

###################### Training Dataset ###########################
# identify outliers in the training dataset
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
print('Training set', X_train.shape, y_train.shape)
print('After oversampling', Counter(y_train))

# For meta leraning, remove ENN can increase accuracy 0.01
# The procedure only removes noisy and ambiguous points along the class boundary  
undersample = EditedNearestNeighbours(n_neighbors=3)
X_train, y_train = undersample.fit_sample(X_train, y_train)
print('After undersampling', Counter(y_train))

# Train the SVM model on the training set; # C = 50, 70, and 100 has the same result.Max Pr@Re50 0.11159
classifier = SVC(kernel='linear', gamma=2.825, C=19, class_weight='balanced', probability=True, shrinking=False, cache_size=10000, verbose=True, random_state=42)
classifier.fit(X_train, y_train)

###################### Bagging ###########################
from imblearn.ensemble import BalancedBaggingClassifier
ensemble = BalancedBaggingClassifier(base_estimator=classifier, n_estimators=1,
                                 sampling_strategy='auto',
                                 replacement=True,
                                 random_state=42)
ensemble.fit(X_train, y_train)

###################### Test Dataset ###########################
# Predict the Test set results
y_pred = ensemble.predict(X_test) 

######################       ROC/AUC   ###########################
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# Predict probabilities
svm_probs = ensemble.predict_proba(X_test)
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
plt.title('Meta - ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

####################### PR Curve #####################
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
svm_precision, svm_recall, _ = precision_recall_curve(y_test, svm_probs)
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(svm_recall, svm_precision, marker='.', label='SVM')
plt.title('Meta - PR Curve')
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

print('Maximum Pr@Re50: %.4f' % np.mean(precision_recall_50), ' +/- %.4f' % np.std(precision_recall_50), '\n')

###################### Evaluation ###########################
#Goal: Maximum achievable precision at a recall of at least 50% (Pr@Re50)
#      The correctness of your Pr@Re50 prediction over the test dataset
print('\nTest data', Counter(y_test))

# Evaluate predictions
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('Confusion Matrix: TN =', tn, 'FP =', fp, 'FN =', fn, 'TP =', tp)
print('\n',classification_report(y_test,y_pred))

###################### Save Model ###########################
import pickle
f = open('svm_meta.pickle','wb')
pickle.dump(ensemble,f)
f.close()

# evaluate model
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(ensemble, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Meta Accuracy: %.3f +/- %.3f' % (np.mean(n_scores), np.std(n_scores)))
print('Meta Accuracy - max: %.3f' % np.max(n_scores))