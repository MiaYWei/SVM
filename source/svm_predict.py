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
print('Original dataset', X.shape)

# Select features
# mask = [0  1  1  1  1  1  1  0  1 0 0 0 0 0 0 0 0 0 0 0 0 0 0  1 0  1 0 0]
features_selected = ['Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'ECI_IB_5_N1', 'Gs(U)_IB_68_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S']
X_new= X[features_selected]
print('Select features', X_new.shape)

# Data Standardization
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_new)

# Load the model from disk
filename = 'pickle\meta_train.pickle'
loaded_model = pickle.load(open(filename, 'rb'))

# Predict the Labels using the reloaded Model
y_predict = loaded_model.predict(X_test) 
output=pd.DataFrame(y_predict)
output.to_csv('prediction\\result_blind.csv', index = False)
print('Pridection...Done')