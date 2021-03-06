# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif,SelectPercentile
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter
import seaborn as sns
import pickle
import time

# Import the dataset
data = pd.read_csv('dataset\csv_result-Descriptors_Training.csv') 
# Convert 'P, N' into '1, 0'
data['class'] = data['class'].map({'P':1,'N':0})
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
print('Original dataset', X.shape, y.shape, Counter(y))

###################### Dataset Visualization ###########################
sns.countplot(data['class'],label="Count")
plt.title('Original Dataset')
plt.show()

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
sp = SelectPercentile(f_classif, percentile=30)
X_new = sp.fit_transform(X, y)
print(sp.get_support())

###################### Split Dataset ###########################
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, random_state=101)
print('Training dataset', X_train.shape, Counter(y_train))
print('Test dataset:', X_test.shape, Counter(y_test))

###################### Training Dataset ###########################
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

print('Training Dataset', Counter(y_train))
sns.countplot(y_train,label="Count")
plt.title('Before Resampling')
plt.show()

# Resample the imbalance dataset by using SVMSMOTE & ENN 
oversample = SVMSMOTE(sampling_strategy='auto', n_jobs =-1, random_state=42) 
X_train, y_train = oversample.fit_sample(X_train, y_train) 

print('After SVMSMOTE', Counter(y_train))
sns.countplot(y_train,label="Count")
plt.title('After SVMSMOTE')
plt.show()

undersample = EditedNearestNeighbours(n_neighbors=5)
X_train, y_train = undersample.fit_sample(X_train, y_train)
print('Training dataset - After resampling', X_train.shape, Counter(y_train))

print('After SVMSMOTE & ENN', Counter(y_train))
sns.countplot(y_train,label="Count")
plt.title('After SVMSMOTE & ENN')
plt.show()

start_time = time.time()
# Train the SVM model on the training set; 
classifier = SVC(kernel='linear', gamma=2.825, C=19, class_weight='balanced', probability=True, 
                 shrinking=False, cache_size=10000, verbose=True, random_state=42)
classifier.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))

###################### Test Model ###########################
y_pred = classifier.predict(X_test) 

######################  ROC/AUC   ###########################
# Generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# Predict probabilities
svm_probs = classifier.predict_proba(X_test)
# Keep probabilities for the positive outcome only
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
plt.title('SVM - ROC Curve')
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
plt.title('SVM - PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# Max precision a sensitivity of 50% 
precision_recall_50 = []
for i in range(0, len(svm_recall)):
    if svm_recall[i] >= 0.5:
        precision_recall_50.append(svm_precision[i])
        plt.scatter(svm_recall[i], svm_precision[i], linewidths = 0, marker = 'X', color='green')
print('Pr@Re50: %4f' % np.mean(precision_recall_50), ' Std: %.4f' % np.std(precision_recall_50), '\n')

axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,0.4])
plt.axvline(x=0.5, color='green', linestyle='dashdot')
plt.show()

###################### Save Model ###########################
f = open('svm.pickle','wb')
pickle.dump(classifier,f)
f.close()

###################### Evaluate  ###########################
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('Confusion Matrix: TN =', tn, 'FP =', fp, 'FN =', fn, 'TP =', tp)
print('\n',classification_report(y_test,y_pred))

# Evaluate model
scores_accuracy = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1)
scores_precision = cross_val_score(classifier, X_train, y_train, scoring='precision', cv=10, n_jobs=-1)
scores_recall = cross_val_score(classifier, X_train, y_train, scoring='recall', cv=10, n_jobs=-1)
print('SVM Accuracy: %.3f +/- %.3f' % (np.mean(scores_accuracy), np.std(scores_accuracy)))
print('SVM Precision: %.3f +/- %.3f' % (np.mean(scores_precision), np.std(scores_precision)))
print('SVM Recall: %.3f +/- %.3f' % (np.mean(scores_recall), np.std(scores_recall)))