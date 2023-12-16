# - This function should take a filename and load the data in the file into a Pandas Dataframe.
# The Dataframe should be returned from the function.
import copy
import pandas
import numpy as np

def load_data(fname):
    '''
    load the data in fname into a pandas dataframe and return it
    '''
    df = pandas.read_csv(fname)
    return df

folder_path = './NetworkTraffic/MachineLearningCVE/'
fname1 = folder_path + 'Monday-WorkingHours.pcap_ISCX.csv'
fname2 = folder_path + 'Tuesday-WorkingHours.pcap_ISCX.csv'
fname3 = folder_path + 'Wednesday-workingHours.pcap_ISCX.csv'
fname4 = folder_path + 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
fname5 = folder_path + 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
fname6 = folder_path + 'Friday-WorkingHours-Morning.pcap_ISCX.csv'
fname7 = folder_path + 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
fname8 = folder_path + 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df1, df2, df3, df4, df5, df6, df7, df8 = load_data(fname1), load_data(fname2), load_data(fname3), load_data(fname4), load_data(fname5), load_data(fname6), load_data(fname7), load_data(fname8)


df_list = [df1, df2, df3, df4, df5, df6, df7, df8]
dfs = copy.deepcopy(df_list)

# - This function should take a Pandas Dataframe and either remove or replace all NaN/Inf
# values. If you replace the NaN values, you must choose how to replace them (with mean,
# median, fixed value, etc.). Your choice should be clearly indicated in your documentation.
# This function should also remove any columns of the data that are not numerical features.
# This function will return a cleaned Dataframe.
def clean_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0) # or df = df.dropna()
        
for df in dfs:
    clean_data(df)
# SAME AS ABOVE
# for i in range(len(dfs)):
#     dfs[i].replace([np.inf, -np.inf], np.nan, inplace=True)
#     dfs[i] = dfs[i].fillna(0) # or df = df.dropna()


# - This function should take a Pandas Dataframe and split the Dataframe into training and testing data. This function should split the data into 80% for training and 20% for testing.
# You can do this randomly or use the first 80% for training and the remaining for testing.
# Make your choice clear in the documentation. This function will return four Dataframes: X_train, y_train, X_test, and y_test.
from sklearn.model_selection import train_test_split
def split_data(df):
    return train_test_split(df, test_size=0.2)
    
    # ALTERNATE VERSION
    # mask = np.random.rand(len(df)) < 0.8
    # df_train = df[mask]
    # df_test = df[~mask]

    # X_train = df_train[df_train.columns[:-1]]
    # y_train = df_train[df_train.columns[-1]]

    # X_test = df_test[df_test.columns[:-1]]
    # y_test = df_test[df_test.columns[-1]]
    
    # return X_train, y_train, X_test, and y_test
