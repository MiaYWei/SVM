# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# Import the dataset
#data = pd.read_csv('test\\svm.csv') 
data = pd.read_csv('test\\Balancedbagging.csv') 
#data = pd.read_csv('test\\RUSboost.csv') 

data['class'] = data['class'].map({'P':1,'N':0})

X_test = data.iloc[:, 1:-3]
y_test_prob = data.iloc[:, -2]
y_test = data.iloc[:, -3]

# Data Standardization
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_test)

######################       ROC/AUC   ###########################
# Generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# Predict probabilities
svm_probs = y_test_prob

# Calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
svm_auc = roc_auc_score(y_test, svm_probs)

# summarize scores
print('AUC=%.3f' % (svm_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
svm_fpr, scm_tpr, _ = roc_curve(y_test, svm_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(svm_fpr, scm_tpr, marker='.', label='SVM')
#plt.title('SVM - ROC Curve')
plt.title('BalancedBagging - ROC Curve')
#plt.title('RUSBoost - ROC Curve')
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
#plt.title('Pure SVM - PR Curve')
plt.title('BalancedBagging - PR Curve')
#plt.title('RUSBoost - PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# Max precision a sensitivity of 50% 
precision_recall_50 = []
for i in range(0, len(svm_recall)):
    if svm_recall[i] >= 0.5:
        precision_recall_50.append(svm_precision[i])
        plt.scatter(svm_recall[i], svm_precision[i], linewidths = 0, marker = 'X', color='green')

axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,0.4])
plt.axvline(x=0.5, color='green', linestyle='dashdot')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

# Pure SVM Model: 0.07595 +/- 0.015
# mu = 0.07595
# sigma = 0.015
# plt.title('Pure SVM model ~ N(0.07595, 0.015) - Prediction Correctness')

#Bagging 0.075629 +/ 0.0142
mu = 0.07563
sigma = 0.0142
plt.title('BalancedBagging ~ N(0.07563, 0.0142) - Prediction Correctness')

#RUSBoost 0.076301 +/- 0.0144
# mu = 0.07630
# sigma = 0.0144
# plt.title('RUSBoost ~ N(0.07630, 0.0144) - Prediction Correctness')

x_actual = np.max(precision_recall_50)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
score = stats.norm.pdf(x_actual, mu, sigma)
plt.scatter(x_actual, score, linewidths = 0, marker = 'X', color='red')
plt.axvline(x=x_actual, color='green', linestyle='dashdot')
plt.axhline(y=score, color='green', linestyle='dashdot')
plt.show()

print('Max Pr@Re50: %4f' % np.max(precision_recall_50))
print('Correctness score: %4f' % score)