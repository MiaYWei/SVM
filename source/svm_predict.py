# Import the libraries
import numpy as np
import pandas as pd
import pickle
import csv
from sklearn.preprocessing import StandardScaler

# Import the blind test dataset
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

# Load the Pre-trained model from disk
filename = 'pickle\meta_train.pickle'
loaded_model = pickle.load(open(filename, 'rb'))

# Predict the labels using the reloaded Model, and save the result in csv file
y_predict = loaded_model.predict_proba(X_test) 
output_value=pd.DataFrame(y_predict)
result_filename = 'prediction\\result_blind_prob.csv'
output_value.to_csv(result_filename, index = False)

y_predict = loaded_model.predict(X_test) 
output_value=pd.DataFrame(y_predict)
result_filename = 'prediction\\result_blind.csv'
output_value.to_csv(result_filename, index = False)

print('Pridection...Done')

# Covert csv file to txt file
prob_csv_file = 'prediction\\result_blind_prob.csv'
prob_txt_file = 'prediction\\result_blind_prob.txt'

csv_file = 'prediction\\result_blind.csv'
txt_file = 'prediction\\result_blind.txt'

def convert_csv_to_txt(sr_csv_file, des_txt_file):
    with open(des_txt_file, "w") as my_output_file:
        with open(sr_csv_file, "r") as my_input_file:
            [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()
    print('Convert to text...Done')

convert_csv_to_txt(prob_csv_file, prob_txt_file)
convert_csv_to_txt(csv_file, txt_file)