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
from sklearn.pipeline import Pipeline
clf = Pipeline([('anova', SelectPercentile(f_classif)),
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel='linear', gamma=0.665, C=11.73, class_weight='balanced', probability=True, shrinking=False, cache_size=10000, verbose=True, random_state=42))])

# #############################################################################
# Plot the cross-validation score as a function of percentile of features
score_means = list()
score_stds = list()
percentiles = (10, 15, 20, 30, 35, 40, 50)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    this_scores = cross_val_score(clf, X, y)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title('Performance of the SVM-Anova varying the percentile of features selected')
plt.xticks(np.linspace(0, 100, 11, endpoint=True))
plt.xlabel('Percentile')
plt.ylabel('Accuracy Score')
plt.axis('tight')
plt.show()