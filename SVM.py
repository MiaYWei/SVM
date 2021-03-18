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
dataset = pd.read_csv('csv_result-Descriptors_Training.csv') 
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

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

###################### Dataset Pre-processing ###########################
# Identify outliers in the training dataset
z_scores = stats.zscore(X)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
X = X[filtered_entries]
y = y[filtered_entries]
print('\nRemove outliers', X.shape, y.shape)

# Select features
selector = SelectKBest(k=10)
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

# Resample the imbalance dataset by using SMOTE
model_smote = SMOTE(random_state = 42) 
X_train, y_train = model_smote.fit_sample(X_train, y_train) 
print('Training set', X_train.shape, y_train.shape)
counter = Counter(y_train)
print('After oversampling',Counter(y_train))

# Train the SVM model on the Training set
classifier = SVC(kernel='rbf', class_weight='balanced', decision_function_shape = 'ovo', shrinking = False, probability=True, verbose = False, random_state = 0)
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
svm_auc = roc_auc_score(y_test, svm_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('SVM: ROC AUC=%.3f' % (svm_auc))
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

####################### PR Curve #####################
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
svm_precision, svm_recall, _ = precision_recall_curve(y_test, svm_probs)
svm_f1, svm_auc = f1_score(y_test, y_pred), auc(svm_recall, svm_precision)
# summarize scores
print('SVM: f1=%.3f auc=%.3f' % (svm_f1, svm_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(svm_recall, svm_precision, marker='.', label='SVM')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

###################### Evaluation ###########################
# Evaluate predictions
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('TN =', tn, 'FP =', fp, 'FN =', fn, 'TP =', tp)
print('ACC =', (tp+tn)/(tp+tn+fn+fp))
print(classification_report(y_test,y_pred))


###################### Bagging ###########################
# from sklearn.ensemble import BaggingClassifier
# ensemble = BaggingClassifier(base_estimator=classifier, n_estimators=31, random_state=42)
# ensemble.fit(X, y)