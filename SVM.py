# Import the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy import stats
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN

# Import the dataset
dataset = pd.read_csv('csv_result-Descriptors_Calibration.csv') #also works for .tsv format header=1 or true header=1, index_col=0
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

# Resample the imbalance dataset by using SMOTE
model_smote = SMOTE(random_state = 42) 
x_smote_resampled, y_smote_resampled = model_smote.fit_sample(X,y) 

# Standarize all features
scaler = StandardScaler()
X_std = scaler.fit_transform(x_smote_resampled)

# Feature selection
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_std, y_smote_resampled)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X_std)
X_std = X_new
print('Feature selection', X_std.shape,y_smote_resampled.shape)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_std, y_smote_resampled, test_size=0.20, random_state=1)
print(X_train.shape)

# Identify outliers in the training dataset
# solution 1:
from sklearn.svm import OneClassSVM
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)

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
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# Summarize the shape of the updated training dataset
print(X_train.shape,y_train.shape)

# Train the SVM model on the Training set
classifier = SVC(kernel='rbf', random_state = 42)
classifier.fit(X_train, y_train)
# Predicte the Test set results
y_pred = classifier.predict(X_test) 


# Evaluate predictions
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('TN =', tn, 'FP =', fp, 'FN =', fn, 'TP =', tp)
print(classification_report(y_test,y_pred))

#Applying k-Fold Cross Validation
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
# print('k-Fold', accuracies)
# print(" k-Fold Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print(" k-Fold Standard Deviation: {:.2f} %".format(accuracies.std()*100))

