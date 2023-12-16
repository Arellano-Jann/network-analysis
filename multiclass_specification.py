import numpy as np
import pandas as pd

# Direct Multi-Class Classification (30 points)
# Directly use our previous methods for multi-class classification (including Decision Trees and
# KNN) to predict multiple classes.
# Implement the following functions:

# - This function should take the model_name (“dt”, “knn”, “mlp”, “rf’) as input along with
# the training data (two Dataframes) and return a trained model.
def direct_multiclass_train(model_name, X_train, y_train):
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    correct = 0
    total = 0
    print('-'*30 + ' Training Decision Trees for each file' + '-'*30)
    for i in range(8):
        print('-'*20 + ' file {}'.format(i) + '-'*20)
        df = dfs[i]
        # split the data into training and testing sets
        mask = np.random.rand(len(df)) < 0.8
        df_train = df[mask]
        df_test = df[~mask]

        X_train = df_train[df_train.columns[:-1]]
        y_train = df_train[df_train.columns[-1]]

        X_test = df_test[df_test.columns[:-1]]
        y_test = df_test[df_test.columns[-1]]

        # print(X_train.shape, y_train.shape)
        # print(X_test.shape, y_test.shape)
        print('num of malicious samples in training set {}: {:.0%}'.format(i, sum(y_train != 'BENIGN')/len(y_train)))
        print('num of malicious samples in testing set {}: {:.0%}'.format(i, sum(y_test != 'BENIGN')/len(y_test)))


        # train the model
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)

        cor = 0
        tot = 0
        predictions = model.predict(X_test)
        for j in range(len(predictions)):
            total += 1
            tot += 1
            if predictions[j] == y_test.iloc[j]:
                correct += 1
                cor +=1
        print('accuracy of file {}: {:.4f}'.format(i, cor/tot))

    print('-'*60)
    print('overall accuracy: {:.4f}'.format(correct/total))

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
