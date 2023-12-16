# Direct Multi-Class Classification (30 points)
# Directly use our previous methods for multi-class classification (including Decision Trees and
# KNN) to predict multiple classes.
# Implement the following functions:

# - This function should take the model_name (“dt”, “knn”, “mlp”, “rf’) as input along with
# the training data (two Dataframes) and return a trained model.
def direct_multiclass_train(model_name, X_train, y_train):
    pass

# - This function should take a trained model and evaluate the model on the test data,
# returning an accuracy value.
def direct_multiclass_test(model, X_test, y_test):
    pass


# Direct Multi-Class Classification with Resampling (20 points)
# Perform data resampling to handle the unbalanced data distribution, and then conduct multi-
# class classification using MLP and random forest.
# Implement the following new functions:

# - This function should take the dataframe as input, undersample it using
# sampling_strategy, and return the resampled df.
def data_resampling(df, sampling_strategy):
    pass


# Hierarchical Multi-Class Classification (30 points)
# Perform binary classification first (benign vs. malicious) using MLP. Once a sample has been
# identified as malicious, perform multi-class classification to identify what kind of malicious
# activity is occurring using random forest.
# Implement the following new functions:

# - This function will take the original data df into train and test sets that both contain all the
# categories. Return train and test dataframes: df_train, and df_test.
def improved_data_split(df):
    pass

# - Convert df into a binary dataset and return it.
def get_binary_dataset(df):
    pass
