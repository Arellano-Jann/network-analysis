# load_data(fname)
# - This function should take a filename and load the data in the file into a Pandas Dataframe.
# The Dataframe should be returned from the function.
# clean_data(df)
# - This function should take a Pandas Dataframe and either remove or replace all NaN/Inf
# values. If you replace the NaN values, you must choose how to replace them (with mean,
# median, fixed value, etc.). Your choice should be clearly indicated in your documentation.
# This function should also remove any columns of the data that are not numerical features.
# This function will return a cleaned Dataframe.
# split_data(df)
# - This function should take a Pandas Dataframe and split the Dataframe into training and
# testing data. This function should split the data into 80% for training and 20% for testing.
# You can do this randomly or use the first 80% for training and the remaining for testing.
# Make your choice clear in the documentation. This function will return four Dataframes:
# X_train, y_train, X_test, and y_test.