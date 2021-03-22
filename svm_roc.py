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
from imblearn.under_sampling import EditedNearestNeighbours
undersample = EditedNearestNeighbours(n_neighbors=3)
X, y = undersample.fit_sample(X, y)
print('After undersampling', Counter(y))

###################### Dataset Visualization ###########################
# visualize Methylated class
# sns.countplot(data['class'],label="Count")
# plt.show()

########################## ROC & cross fold ############
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
# Run classifier with cross-validation and plot ROC curves
classifier = SVC(kernel='linear', gamma=0.665, C=11.73, class_weight='balanced', probability=True, shrinking=False, cache_size=10000, verbose=True, random_state=42)
cv = StratifiedKFold(n_splits=6)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = plot_roc_curve(classifier, X[test], y[test], name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC with StratifiedKFold")
ax.legend(loc="lower right")
plt.show()
