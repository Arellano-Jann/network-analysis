# - This function should take a filename and load the data in the file into a Pandas Dataframe.
# The Dataframe should be returned from the function.
import copy
import pandas
import numpy as np

def load_data(fname):
    '''
    load the csv data (fname) into a pandas dataframe and return it
    output: returns the pandas dataframe
    input: filename associated with a csv
    '''
    df = pandas.read_csv(fname)
    print('successfully loaded data')
    return df


# - This function should take a Pandas Dataframe and either remove or replace all NaN/Inf
# values. If you replace the NaN values, you must choose how to replace them (with mean,
# median, fixed value, etc.). Your choice should be clearly indicated in your documentation.
# This function should also remove any columns of the data that are not numerical features.
# This function will return a cleaned Dataframe.
def clean_data(df):
    '''
    cleans the data by replacing inf and NaN values with 0
    output: returns the cleaned dataframe
    input: pandas dataframe to be cleaned
    '''
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0) # or df = df.dropna()
    return df


# - This function should take a Pandas Dataframe and split the Dataframe into training and testing data. This function should split the data into 80% for training and 20% for testing.
# You can do this randomly or use the first 80% for training and the remaining for testing.
# Make your choice clear in the documentation. This function will return four Dataframes: X_train, y_train, X_test, and y_test.
from sklearn.model_selection import train_test_split
def split_data(df, test_size=0.2):
    '''
    splits the input dataframe into training and testing data
    this split is 80% training and 20% testing on default but configurable by setting test_size
    output: returns 4 dataframes which are X_train, y_train, X_test, and y_test
    input: pandas dataframe to be split into train and test
    '''
    return train_test_split(df, test_size=test_size)
    
    # ALTERNATE VERSION
    # mask = np.random.rand(len(df)) < 0.8
    # df_train = df[mask]
    # df_test = df[~mask]

    # X_train = df_train[df_train.columns[:-1]]
    # y_train = df_train[df_train.columns[-1]]

    # X_test = df_test[df_test.columns[:-1]]
    # y_test = df_test[df_test.columns[-1]]
    
    # return X_train, y_train, X_test, and y_test
