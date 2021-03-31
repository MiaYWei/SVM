# Import the libraries
import csv
# Covert csv file to txt file
csv_file = 'submission\\blind_test_prediction_group12.csv'
txt_file = 'submission\\blind_test_prediction_group12.txt'

def convert_csv_to_txt(sr_csv_file, des_txt_file):
    with open(des_txt_file, "w") as my_output_file:
        with open(sr_csv_file, "r") as my_input_file:
            [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()
    print('Convert to text...Done')

convert_csv_to_txt(csv_file, txt_file)