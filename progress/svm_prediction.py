# Import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_selection import f_classif,SelectPercentile

# Import the dataset
data = pd.read_csv('dataset\csv_result-Descriptors_Calibration.csv') 
# Convert 'P, N' into '1, 0'
data['class'] = data['class'].map({'P':1,'N':0})

X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
print('Original dataset', X.shape, y.shape)
print(Counter(y))

################### Identify and remove outliers #######################
# Select features
#X = SelectPercentile(f_classif, percentile=30).fit_transform(X, y)
features_selected = ['Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'ECI_IB_5_N1', 'Gs(U)_IB_68_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S']
X= X[features_selected]
print('Select features', X.shape)

# Data Standardization
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X)

# load the model from disk
import pickle
filename = 'pickle\meta_train.pickle'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y)
print(result)

# Predict the Labels using the reloaded Model
y_predict = loaded_model.predict(X_test) 
print('Result', X_test.shape, y.shape)

output=pd.DataFrame(y_predict)
output.to_csv('result_meta_trainXXXXXX.csv', index = False)

# Evaluate predictions
tn, fp, fn, tp = confusion_matrix(y, y_predict).ravel()
print('TN =', tn, 'FP =', fp, 'FN =', fn, 'TP =', tp)
print(classification_report(y,y_predict))

print('Predict Performance:' )
print(' Accuracy = %4f' % ((tp+tn)/(tp+tn+fn+fp)))
print(' Recall  = %4f' % (tp/(tp+fn)))
print(' Presision = %4f' % (tp/(tp+fp)))