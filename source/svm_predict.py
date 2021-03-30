# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectPercentile
from collections import Counter
import pickle

# Import the dataset
data = pd.read_csv('dataset\Blind_Test_features.csv') 
X = data.iloc[:, :]
print(data)
print('Original dataset', X.shape)

# Select features
X = SelectPercentile(f_classif, percentile=30).fit_transform(X, y)
print('Select features', X.shape, )

# Data Standardization
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X)

# Load the model from disk
filename = 'meta_train.pickle'
loaded_model = pickle.load(open(filename, 'rb'))

# Predict the Labels using the reloaded Model
y_predict = loaded_model.predict(X_test) 
output=pd.DataFrame(y_predict)
output.to_csv('result_blind.csv', index = False)