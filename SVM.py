# Import the libraries
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
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
features_N1= list(data.columns[2:12]) #from B to L
corr = data[features_N1].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True) # annot=True display numbers
plt.show()

features_l= list(data.columns[12:28]) 
corr_l = data[features_l].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr_l, annot=True)
plt.show()

###################### Dataset Pre-processing ###########################
# Identify and remove outliers
z_scores = stats.zscore(X)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
X = X[filtered_entries]
y = y[filtered_entries]
print('\nRemove outliers', X.shape, y.shape)

# # Select features
# selector = SelectKBest(k = 10)
# X = selector.fit_transform(X,y)
# print('\nFeature selection', X.shape, y.shape)
# # print(selector.scores_)
# # print(selector.pvalues_)

features_selected = ['ECI_IB_4_N1','Gs(U)_IB_68_N1', 'Gs(U)_NO_ALR_SI71','ISA_NO_NPR_S','IP_NO_PLR_S']

###################### Split Dataset ###########################
# Split dataset into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# print('\nTraining set',X_train.shape, y_train.shape)

train, test = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test
X_train = train[features_selected]
y_train=train['class']
X_test= test[features_selected]
y_test =test['class']

###################### Training Dataset ###########################
# Data Standardization
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Data Normalization -- worse than Standardization
# from sklearn.preprocessing import MinMaxScaler
# norm = MinMaxScaler().fit(X_train)
# X_train = norm.transform(X_train)
# X_test = norm.transform(X_test)

# Resample the imbalance dataset by using SMOTE
model_smote = SMOTE(random_state = 42) 
X_train, y_train = model_smote.fit_sample(X_train, y_train) 
print('Training set', X_train.shape, y_train.shape)
print('After oversampling', Counter(y_train))

# Train the SVM model on the Training set
classifier = SVC(kernel='linear', gamma = 'auto', decision_function_shape = 'ovo', class_weight='balanced', probability=True, cache_size = 10000, verbose = True, random_state = 42)
classifier.fit(X_train, y_train)

from sklearn.feature_selection import RFE
selector = RFE(classifier, 5, step=1)
selector = selector.fit(X_train, y_train)
print(selector.ranking_)
#Result(RFE(classifier, 5, step=1)) -- [22 15 20  9 12  1 18 17  1  5  2 23 24  3  6  7 21 19  1 13  8 10 14  1 16  1 11  4]

###################### Test Dataset ###########################
# Predict the Test set results
print('\nTest data', X_test.shape)
y_pred = classifier.predict(X_test) 

# Predict probabilities
svm_probs = classifier.predict_proba(X_test)
# keep probabilities for the positive outcome only
svm_probs = svm_probs[:, 1]

######################       ROC/AUC   ###########################
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# Calculate scores
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
# print('SVM: f1=%.3f auc=%.3f' % (svm_f1, svm_auc))
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