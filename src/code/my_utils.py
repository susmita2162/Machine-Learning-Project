import numpy as np, pandas as pd


#transforming both datasets into neural network input format : tuple(x, y)
#converting target values into OHE-HOT-ENCODING using pandas "get_dummies" function!
def transform_datasets(train_df, test_df):
    columns = list(train_df.columns)
    columns.remove('target')

    train_data = []
    x = train_df[columns].to_numpy()
    y = pd.get_dummies(train_df.target, prefix = 'class').to_numpy()
    for i in range(len(x)):
        train_data.append((list(x[i]), y[i]))

    test_data = []
    x = test_df[columns].to_numpy()
    y = pd.get_dummies(test_df.target, prefix = 'class').to_numpy()
    for i in range(len(x)):
        test_data.append((list(x[i]), y[i]))
    return train_data, test_data


def get_crossValidation_datasets(K, dataset):
    classes = dataset.target.unique()
    k_folds = []
    for cls in classes:
        df_cls = dataset[dataset.target == cls]
        #np.random.seed(42)
        k_folds.append(np.array_split(df_cls.reindex(np.random.permutation(df_cls.index)), K))
    X_train = []
    X_test = []
    for k in range(K):
        train = []
        test = []
        for fold in k_folds:
            b = [x for i,x in enumerate(fold) if i != k]
            c = [x for i,x in enumerate(fold) if i == k]
            train.append(pd.concat(b))
            test.append(pd.concat(c))
        X_train.append(pd.concat(train))
        X_test.append(pd.concat(test))
    return X_train, X_test


''' Performing one hot encoding'''
def one_hotencoding(dataset, col):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(dataset[[col]]).toarray())
    dataset = dataset.join(enc_df)
    for j in range(len(dataset[col].unique())):
        st = col + '_' + str(j)
        dataset.rename(columns={j: st}, inplace=True)
    del dataset[col]
    return dataset



'''
This function return a dictionary with column names as keys and their type as value
  - I set threshold = 5 to comply with votes,cancer,wine datasets in HW3. 
  - In general, we need to set after analyzing the datasets & the performance of trained models.
'''
def get_column_types(df):
    threshold = 5
    types = {}
    for col in df.columns:
        if df[col].nunique() < threshold:
            types[col] = ['categorical', list(df[col].unique())]
        else:
            types[col] = ['numerical']
    return types