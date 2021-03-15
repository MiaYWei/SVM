# Import the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
from sklearn.feature_selection import SelectKBest

# Import the dataset
dataset = pd.read_csv('csv_result-Descriptors_Training.csv') 
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X) 
X = imputer.transform(X)

# Convert 'P, N' into '1, 0'
methylated = np.zeros(len(y))
for i in range(len(y)):
    if y[i] == 'P':
        methylated[i] = 1
    if y[i] == 'N':
        methylated[i] = 0

y = methylated

# select features using the training set only
selector = SelectKBest(k=10)
X_new = selector.fit_transform(X,y)
print('Feature selection', X_new.shape, y.shape)

# Identify outliers in the training dataset
# solution 1:
# from sklearn.svm import OneClassSVM
# ee = OneClassSVM(nu=0.01)
# yhat = ee.fit_predict(X)

# solution 2:
# from sklearn.ensemble import IsolationForest
# iso = IsolationForest(contamination=0.1)
# yhat = iso.fit_predict(X_train)

# solution 3:
# from sklearn.neighbors import LocalOutlierFactor
# lof = LocalOutlierFactor()
# yhat = lof.fit_predict(X_train)

# solution 4:
# identify outliers in the training dataset
# from sklearn.covariance import EllipticEnvelope
# ee = EllipticEnvelope(contamination=0.01)
# yhat = ee.fit_predict(X_train)

# Select all rows that are not outliers
# mask = yhat != -1
# X, y = X[mask, :], y[mask]
# # Summarize the shape of the updated training dataset
# print(X.shape,y.shape)

###################### Split Dataset ###########################
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, random_state=1)

###################### Training Dataset ###########################
# Standarize all features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Resample the imbalance dataset by using SMOTE
model_smote = SMOTE(random_state = 42) 
X_train, y_train = model_smote.fit_sample(X_train, y_train) 
print(X_train.shape,y_train.shape)

# Train the SVM model on the Training set
classifier = SVC(kernel='rbf', random_state = 42)
classifier.fit(X_train, y_train)

###################### Test Dataset ###########################
# Predicte the Test set results
y_pred = classifier.predict(X_test) 

###################### Evaluation Dataset ###########################
# Evaluate predictions
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('TN =', tn, 'FP =', fp, 'FN =', fn, 'TP =', tp)
print(classification_report(y_test,y_pred))