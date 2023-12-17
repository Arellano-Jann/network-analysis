# from helpers import *
# from multiclass_specification import *

from helpers import load_data, clean_data, split_data
from multiclass_specification import direct_multiclass_train, direct_multiclass_test, data_resampling, improved_data_split, get_binary_dataset

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


# def main():
folder_path = 'csv/'
# fname1 = folder_path + 'Monday-WorkingHours.pcap_ISCX.csv'
# fname2 = folder_path + 'Tuesday-WorkingHours.pcap_ISCX.csv'
# fname3 = folder_path + 'Wednesday-workingHours.pcap_ISCX.csv'
fname4 = folder_path + 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
# fname5 = folder_path + 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
# fname6 = folder_path + 'Friday-WorkingHours-Morning.pcap_ISCX.csv'
# fname7 = folder_path + 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
# fname8 = folder_path + 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    
# df1, df2, df3, df4, df5, df6, df7, df8 = load_data(fname1), load_data(fname2), load_data(fname3), load_data(fname4), load_data(fname5), load_data(fname6), load_data(fname7), load_data(fname8)

# df_list = [df1, df2, df3, df4, df5, df6, df7, df8]
# dfs = copy.deepcopy(df_list)
# for df in dfs:
#     clean_data(df) # maybe this won't deep clean it

smallfile = folder_path + 'traffic_data_copy.csv'
df = load_data(smallfile)
print(df.info())
df = clean_data(df)  
print(df.info())

dfs = copy.deepcopy([df])  

# df = pd.concat(dfs[1:])
# df.to_csv(path_or_buf=folder_path+'traffic_data.csv' , index=False) # output to one csv


# DTREE
print('DTREE')
# dtree for each file?
correct = 0
count = 0
for i in range(len(dfs)):
    df_i = dfs[i]
    
    X_train, y_train, X_test, y_test = split_data(df_i) # split train and test data
    
    model = direct_multiclass_train('dt', X_train, y_train) # train model
    acc = direct_multiclass_test(model, X_test, y_test) # get model accuracy
    correct += acc * len(y_test)
    count += len(y_test)
    print('file acc', acc)
print('overall acc', correct/count)
    

# KNN
# whole dataset for knn?
print('KNN')
df = load_data(folder_path+'traffic_data.csv') # load merged dataframes
X_train, y_train, X_test, y_test = split_data(df) # split train and test data

model = direct_multiclass_train('knn', X_train, y_train) # train model
acc = direct_multiclass_test(model, X_test, y_test) # get model accuracy
print('overall acc', acc)


# MLP
# resample whole dataframe
print('MLP')
# resampled_df = data_resampling(df) # undersamples the whole dataframe
resampled_df = df
X_train, y_train, X_test, y_test = split_data(resampled_df) # split train and test data

model = direct_multiclass_train('mlp', X_train, y_train) # train model
acc = direct_multiclass_test(model, X_test, y_test) # get model accuracy
print('overall acc', acc)


# RF
# resample whole dataframe
print('RF')
# resampled_df = data_resampling(df) # undersamples the whole dataframe
resampled_df = df
X_train, y_train, X_test, y_test = split_data(resampled_df) # split train and test data

model = direct_multiclass_train('rf', X_train, y_train) # train model
acc = direct_multiclass_test(model, X_test, y_test) # get model accuracy
print('overall acc', acc)
    
# if __name__ == "__main__":
#     main()

# main()