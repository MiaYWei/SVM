# Import the libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_selection import f_classif,SelectPercentile
import joblib

# Import the dataset
data = pd.read_csv('csv_result-Descriptors_Training.csv') 
# Convert 'P, N' into '1, 0'
data['class'] = data['class'].map({'P':1,'N':0})

X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
print('Original dataset', X.shape, y.shape)
print(Counter(y))

# Select features
X = SelectPercentile(f_classif, percentile=30).fit_transform(X, y)
print('Select features', X.shape, y.shape)

###################### Training Dataset ###########################
# Data Standardization
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X)

# load the model from disk
import pickle
filename = 'meta_cal_ENN(3).pickle'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y)
print('Prediction Accuracy', result)

# Predict the Labels using the reloaded Model
y_predict = loaded_model.predict(X_test) 
print(len(y_predict))

print('Result', X_test.shape, y.shape)

output=pd.DataFrame(y_predict)
output.to_csv('result_meta_cal_ENN(3).csv', index = False)