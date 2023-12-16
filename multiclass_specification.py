import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed = 1 # make sure your generate the same random values when using np.random
folder_path = 'csv/' # default folder path

# Direct Multi-Class Classification (30 points)
# Directly use our previous methods for multi-class classification (including Decision Trees and
# KNN) to predict multiple classes.
# Implement the following functions:

# - This function should take the model_name (“dt”, “knn”, “mlp”, “rf’) as input along with the training data (two Dataframes) and return a trained model.
def direct_multiclass_train(model_name, X_train, y_train):
    if model_name not in ('dt', 'knn', 'mlp', 'rf'): 
        print("Invalid model_name in function direct_multiclass_train")
        return
    if model_name == 'dt':
        from sklearn import tree
        from sklearn.tree import DecisionTreeClassifier
        print('-'*30 + ' Training Decision Trees' + '-'*30)
        print('num of malicious samples in training set {}: {:.0%}'.format(i, sum(y_train != 'BENIGN')/len(y_train)))
        print('num of malicious samples in testing set {}: {:.0%}'.format(i, sum(y_test != 'BENIGN')/len(y_test)))

        # train the model
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)
        
    if model_name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        print('-'*30 + ' Training KNN' + '-'*30)
        print('num of malicious samples in training set {}: {:.0%}'.format(i, sum(y_train != 'BENIGN')/len(y_train)))
        print('num of malicious samples in testing set {}: {:.0%}'.format(i, sum(y_test != 'BENIGN')/len(y_test)))

        # train the model
        model = KNeighborsClassifier()
        model = model.fit(X_train, y_train)
    
    if model_name == 'mlp':
        from sklearn.neural_network import MLPClassifier
        # training
        model = MLPClassifier(hidden_layer_sizes=(40,), random_state=1, max_iter=300).fit(X_train, y_train)
        
    if model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        # training
        model = RandomForestClassifier().fit(X_train, y_train)
    return model
    

# - This function should take a trained model and evaluate the model on the test data,
# returning an accuracy value.
def direct_multiclass_test(model, X_test, y_test):

    # prediction
    pred = model.predict(X_test)
    acc = accuracy_score(pred, y_test)
    print('Test Accuracy : {:.5f}'.format(acc))
    print('Classification_report:')
    print(classification_report(y_test, pred))
    plt.show()
    return acc

    # OTHER IMPLEMENT
    correct = 0
    total = 0
    predictions = model.predict(X_test)
    for i in range(len(predictions)):
        total += 1
        if predictions[i] == y_test.iloc[i]:
            correct += 1
    return float(correct/total)


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

    resampled_df.to_csv(path_or_buf=folder_path+'resampled_clean_traffic_data.csv', index=False)
    fig = plt.figure(figsize=(10,6))
    sns.countplot(x=' Label', data=resampled_df)
    plt.show()
    return resampled_df


# Hierarchical Multi-Class Classification (30 points)
# Perform binary classification first (benign vs. malicious) using MLP. Once a sample has been identified as malicious, perform multi-class classification to identify what kind of malicious activity is occurring using random forest.
# Implement the following new functions:

# - This function will take the original data df into train and test sets that both contain all the categories. 
# Return train and test dataframes: df_train, and df_test.
def improved_data_split(df):
    label_set = set(df[' Label'])
    # print(label_set)

    df_train_list=[]
    df_test_list=[]
    print('-'*60)
    for label in label_set:
        mask = np.random.rand(len(df[df[' Label']== label])) < 0.8
        print('num of "{}" data samples: {}'.format(label, len(mask)))
        df_train_list.append(df[df[' Label']== label][mask])
        df_test_list.append(df[df[' Label']== label][~mask])

    df_train = pd.concat(df_train_list)
    df_test = pd.concat(df_test_list)

    df_train.to_csv(folder_path+'train_traffic_data.csv', index=False)
    df_test.to_csv(folder_path+'test_traffic_data.csv', index=False)
    # check if testing set contains all the categories
    print('-'*60)
    print('check if testing set contains all the categories:', set(df_train[' Label']) == set(df_test[' Label']))
    # print(len(df) == (len(df_train) + len(df_test)))
    print('-'*60)
    for label in label_set:
        print('num of "{}" training samples: {}'.format(label, len(df_train[df_train[' Label']== label])))
        # print('-'*30)
        print('num of "{}" testing samples: {}'.format(label, len(df_test[df_test[' Label']== label])))
        
        
    fig = plt.figure(figsize=(25,6))
    
    sns.countplot(x=' Label', data=df_train)
    plt.show()
    
    sns.countplot(x=' Label', data=df_test)
    plt.show()
    
    return df_train, df_test

# - Convert df into a binary dataset and return it.
def get_binary_dataset(df):
    df_binary = df.copy() # this is a deep copy
    # 1. convert all malicious labels to "MALICIOUS"
    df_binary.loc[df_binary[' Label'] != 'BENIGN', ' Label'] = 'MALICIOUS'

    # 2. save the data
    df_binary.to_csv(folder_path+'binary_traffic_data.csv')
    # plot
    fig = plt.figure(figsize=(10,6))
    sns.countplot(x=' Label', data=df_binary)
    plt.show()
    
    return df_binary
