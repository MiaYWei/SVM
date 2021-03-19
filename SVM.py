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

# Display dataset
# pd.set_option('display.max_columns', None)
# print(data.columns)
# print(data.head(3))
# print(data.describe())

###################### Dataset Visualization ###########################
# visualize Methylated class
sns.countplot(data['class'],label="Count")
plt.show()

# Heatmap 
# features_N1= list(data.columns[2:12]) #from B to L
# corr = data[features_N1].corr()
# plt.figure(figsize=(14,14))
# sns.heatmap(corr, annot=True) # annot=True display numbers
# plt.show()

# features_l= list(data.columns[12:28]) 
# corr_l = data[features_l].corr()
# plt.figure(figsize=(14,14))
# sns.heatmap(corr_l, annot=True)
# plt.show()

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

# Train the SVM model on the Training set
classifier = SVC(kernel='linear', gamma = 'auto', decision_function_shape = 'ovo', class_weight='balanced', probability=True, shrinking = False, cache_size = 10000, verbose = True, random_state = 42)
classifier.fit(X_train, y_train)

from sklearn.feature_selection import RFE
selector = RFE(classifier, 8, step = 1)
selector = selector.fit(X_train, y_train)
#print(selector.ranking_)

###################### Test Dataset ###########################
# Predict the Test set results
print('\nTest data', X_test.shape)
y_pred = classifier.predict(X_test) 



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

# Max precision a sensitivity of 50% 
precision_recall_50 = []
for i in range(0, len(svm_recall)):
    if svm_recall[i] >= 0.5:
        precision_recall_50.append(svm_precision[i])
        plt.scatter(svm_recall[i], svm_precision[i], linewidths = 0, marker = 'X', color='red')
print('Max Pr@Re50', max(precision_recall_50))

###################### Evaluation ###########################
#Goal: Maximum achievable precision at a recall of at least 50% (Pr@Re50)
#      The correctness of your Pr@Re50 prediction over the test dataset
print('\nTest data', Counter(y_test))

# Evaluate predictions
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('TN =', tn, 'FP =', fp, 'FN =', fn, 'TP =', tp)
print(classification_report(y_test,y_pred))

# Acture Positive: TP + FN; Acture Negative: FP + TN
print('\nActual Positive', tp+fn)
print('Predict Positive', tp+fp)

print('\nActual Negative', fp+tn)
print('Predict Negative', fn+tn)

print('ACC', (tp+tn)/(tp+tn+fn+fp))
print('TPR (Sensitivity/Recall)', tp/(tp+fn))
print('FPR (1-Specificity)', fp/(tn+fp))

###################### Bagging ###########################
# from sklearn.ensemble import BaggingClassifier
# ensemble = BaggingClassifier(base_estimator=classifier, n_estimators=31, random_state=42)
# ensemble.fit(X, y)