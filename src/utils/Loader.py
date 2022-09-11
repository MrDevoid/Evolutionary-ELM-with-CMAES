from keras.datasets import mnist
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
import re 

def get_data(problem_type, dataset, **args):
    if dataset == "mnist":
        return prepare_mnist(how_many = args.get('how_many'), test_size = args.get('test_size'))
    elif dataset == "mnist_full":
        return prepare_mnist_full(how_many = args.get('how_many'))
    elif "_full" in dataset:
        return prepare_dataset_full(problem_type, dataset, header = args.get('header'), string_y = args.get('string_y'))
    else:
        return prepare_dataset(problem_type, dataset, test_size = args.get('test_size'), header = args.get('header'), string_y = args.get('string_y'), how_many = args.get('how_many'))

def get_data_cv(problem_type, dataset, **args):
    return prepare_dataset_cv(problem_type, dataset, header = args.get('header'), string_y = args.get('string_y'), how_many = args.get('how_many'))

def process_mnist(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    y_train = np.repeat(y_train,10).reshape(y_train.shape[0],10)
    y_test = np.repeat(y_test,10).reshape(y_test.shape[0],10)
    for i in range(0,10):
        y_train[:,i] = y_train[:, i]==i
        y_test[:,i] = y_test[:, i]==i
    y_test = np.argmax(y_test, axis=1)
    return {'X_train':X_train.astype(np.int16), 'y_train': y_train, 'X_test': X_test.astype(np.int16), 'y_test': y_test}

def prepare_mnist_full(how_many=60000):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return process_mnist(X_train[:how_many], y_train[:how_many], X_test, y_test)
    
def prepare_mnist(how_many=60000, test_size=0.3):
    (X_train, y_train), _ = mnist.load_data()
    X_train, X_test, y_train, y_test = train_test_split(X_train[:how_many], y_train[:how_many], test_size=test_size)
    return process_mnist(X_train[:how_many], y_train[:how_many], X_test, y_test)

def prepare_dataset(problem_type, filename, test_size=0.3, header=None, string_y=None, how_many=None):
    df = pd.read_csv(filename, header=header)
    if how_many is not None:
        df = df.sample(how_many)
    train, test = train_test_split(df, test_size=test_size)
    X_train = train.drop(len(train.columns)-1, axis=1)
    X_test = test.drop(len(test.columns)-1, axis=1)
    if 'CLASSIFICATION' in problem_type:
        y_train = train[len(train.columns)-1]==string_y
        y_test = test[len(test.columns)-1]==string_y
    else:
        y_train = train[len(train.columns)-1]
        y_test = test[len(test.columns)-1]
    return {'X_train': X_train.to_numpy(), 'y_train': y_train.to_numpy(), 'X_test': X_test.to_numpy(), 'y_test': y_test.to_numpy()}

def prepare_dataset_full(problem_type, filename, header=None, string_y=None, how_many=None):
    filename_train = re.sub("_full", "_train", filename)
    filename_test = re.sub("_full", "_test", filename)
    train = pd.read_csv(filename_train, header=header)
    test = pd.read_csv(filename_test, header=header)
    X_train = train.drop(len(train.columns)-1, axis=1)
    X_test = test.drop(len(test.columns)-1, axis=1)
    if 'CLASSIFICATION' in problem_type:
        y_train = train[len(train.columns)-1]==string_y
        y_test = test[len(test.columns)-1]==string_y
    else:
        y_train = train[len(train.columns)-1]
        y_test = test[len(test.columns)-1]
    return {'X_train': X_train.to_numpy(), 'y_train': y_train.to_numpy(), 'X_test': X_test.to_numpy(), 'y_test': y_test.to_numpy()}

def prepare_dataset_cv(problem_type, filename, header=None, string_y=None, how_many=None):
    filename_train = re.sub("_full", "_train", filename)
    filename_test = re.sub("_full", "_test", filename)
    train = pd.read_csv(filename_train, header=header)
    test = pd.read_csv(filename_test, header=header)
    df = pd.concat([train, test])
    X = df.drop(len(df.columns)-1, axis=1)
    if 'CLASSIFICATION' in problem_type:
        y = df[len(df.columns)-1]==string_y
    else:
        y = train[len(train.columns)-1]
    kf = KFold(n_splits=5)
    # for train_index, test_index in kf.split(X):
    #     print(str(train_index))
    #     print(str(test_index))
    return {'X_train': X.to_numpy(), 'y_train': y.to_numpy(), 'splits':kf}

def prepare_data_cv(data, train_index, test_index, test_size=0.3):
    X = data['X_train']
    y = data['y_train']
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_size)
    return {'X_train': X_train, 'X_validation': X_validation, 'X_test': X_test, 'y_train': y_train, 'y_validation': y_validation, 'y_test': y_test}