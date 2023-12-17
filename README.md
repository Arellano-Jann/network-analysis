# network-analysis

This assignment shows multiple multi-class classification methods using the network traffic data.

This is done in a few ways:

**Direct Multi-Class Classification [KNN, DT, NN, RF]**

Directly use our previous methods for binary classification (Decision Trees, KNN, Perceptron, Neural Networks) to predict multiple classes.

**Direct Multi-Class Classification with Resampling**

Resample the large, unbalanced dataset to have a smaller and more balanced dataset for classifier

**Today: Hierarchical Multi-Class Classification**

Perform binary classification first (benign vs. malicious). Once a sample has been identified as malicious, perform multi-class classification to identify what kind of malicious activity is occurring.


To run this project, you’ll need to install all
the necessary packages (numpy, pandas, scikit learn, etc.). 

## Helpers.py

`load_data(fname)`
- Input: This function should take a filename 
- Details: Loads the data in the file into a Pandas Dataframe.
- Output: The Dataframe should be returned from the function.

`clean_data(df)`
- Input: This function should take a Pandas Dataframe 
- Details: Replace all NaN/Inf values with 0. This function should also remove any columns of the data that are not numerical features.
- Output: This function will return a cleaned Dataframe.

`split_data(df)`
- Input: This function should take a Pandas Dataframe
- Details: split the Dataframe into training and testing data. This function should split the data into 80% for training and 20% for testing. This function splits it how the train_test_split function does which is randomly.
- Output: This function will return four Dataframes: X_train, y_train, X_test, and y_test.



## multiclass_classification.py
### Direct Multi-Class Classification
Directly use our previous methods for multi-class classification (including Decision Trees and
KNN) to predict multiple classes.

`direct_multiclass_train(model_name, X_train, y_train)`
- Input: This function should take the model_name (“dt”, “knn”, “mlp”, “rf’) as input along with the training data (two Dataframes) 
- Details: Trains a model according to the model_name inputted.
- Output: return a trained model.

`direct_multiclass_test(model, X_test, y_test)`
- Input: This function should take a trained model 
- Details: evaluate the model on the test data,
- Output: returning an accuracy value.

### Direct Multi-Class Classification with Resampling
Perform data resampling to handle the unbalanced data distribution, and then conduct multi-class classification using MLP and random forest

`data_resampling(df, sampling_strategy)`
- Input: This function should take the dataframe as input, undersample it using sampling_strategy
- Details: Perform data resampling to handle the unbalanced data distribution, and then conduct multi-class classification using MLP and random forest.
- Output: return the resampled df.

### Hierarchical Multi-Class Classification Functions
Perform binary classification first (benign vs. malicious) using MLP. Once a sample has been
identified as malicious, perform multi-class classification to identify what kind of malicious
activity is occurring using random forest.

`improved_data_split(df)`
- Input: This function will take the original data df 
- Details: Transforms the df into train and test sets that both contain all the categories. 
- Output: Return train and test dataframes: df_train, and df_test.

`get_binary_dataset(df)`
- Input: This function will take a df 
- Details: Transfom df into a binary dataset
- Output: Returns the binary df