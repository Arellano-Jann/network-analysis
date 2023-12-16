import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Direct Multi-Class Classification (30 points)
# Directly use our previous methods for multi-class classification (including Decision Trees and
# KNN) to predict multiple classes.
# Implement the following functions:

# - This function should take the model_name (“dt”, “knn”, “mlp”, “rf’) as input along with the training data (two Dataframes) and return a trained model.
# CHECK FUNCTIONS FOR THEIR OWN X_TRAIN AND Y_TRAIN AND REMOVE AS NEEDED
def direct_multiclass_train(model_name, X_train, y_train):
    if model_name not in ('dt', 'knn', 'mlp', 'rf'): 
        print("Invalid model_name in direct_multiclass_train")
        return
    if model_name == 'dt':
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
        
    if model_name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        correct = 0
        total = 0
        print('-'*30 + ' Training KNN for each file' + '-'*30)
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
            model = KNeighborsClassifier()
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
    
    if model_name == 'mlp':
        from sklearn.neural_network import MLPClassifier
        mask = np.random.rand(len(resampled_df)) < 0.8
        df_train = resampled_df[mask]
        df_test = resampled_df[~mask]

        X_train = df_train[df_train.columns[:-1]]
        y_train = df_train[df_train.columns[-1]]

        X_test = df_test[df_test.columns[:-1]]
        y_test = df_test[df_test.columns[-1]]

        # training
        mlp = MLPClassifier(hidden_layer_sizes=(40,), random_state=1, max_iter=300).fit(X_train, y_train)

        # prediction
        pred = mlp.predict(X_test)
        acc = accuracy_score(pred, y_test)
        print('Test Accuracy : {:.5f}'.format(acc))
        print('Classification_report:')
        print(classification_report(y_test, pred))
    
    if model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        # training
        rfc = RandomForestClassifier().fit(X_train, y_train)

        # prediction
        pred = rfc.predict(X_test)
        acc = accuracy_score(pred, y_test)
        print('Test Accuracy : {:.5f}'.format(acc))
        print('Classification_report:')
        print(classification_report(y_test, pred))
        plt.show()
    

# - This function should take a trained model and evaluate the model on the test data,
# returning an accuracy value.
def direct_multiclass_test(model, X_test, y_test):
    pass


# Direct Multi-Class Classification with Resampling (20 points)
# Perform data resampling to handle the unbalanced data distribution, and then conduct multi-class classification using MLP and random forest.
# Implement the following new functions:

# - This function should take the dataframe as input, undersample it using
# sampling_strategy, and return the resampled df.
def data_resampling(df, sampling_strategy='majority'):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=2)
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    # resample
    X_resampled, y_resampled = rus.fit_resample(X, y)
    resampled_df = pd.DataFrame(columns=df.columns)

    resampled_df[resampled_df.columns[:-1]] = X_resampled
    resampled_df[resampled_df.columns[-1]] = y_resampled

    # resampled_df.to_csv(path_or_buf = folder_path + 'resampled_clean_traffic_data.csv' , index=False)
    return resampled_df


# Hierarchical Multi-Class Classification (30 points)
# Perform binary classification first (benign vs. malicious) using MLP. Once a sample has been
# identified as malicious, perform multi-class classification to identify what kind of malicious
# activity is occurring using random forest.
# Implement the following new functions:

# - This function will take the original data df into train and test sets that both contain all the categories. 
# Return train and test dataframes: df_train, and df_test.
def improved_data_split(df):
    pass

# - Convert df into a binary dataset and return it.
def get_binary_dataset(df):
    pass
