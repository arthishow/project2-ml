import numpy as np
import pandas as pd
from scipy import optimize
from surprise import Dataset, Reader

def load_data_surprise(path, threshold):
    """Load the dataset located at path into a Surprise Dataset and a Pandas DataFrame."""
    dataframe = pd.read_csv(path)
    dataframe['User'], dataframe['Item'] = dataframe['Id'].str.split('_', 1).str
    dataframe = dataframe[['User', 'Item', 'Prediction']]
    g = dataframe.groupby('User')
    dataframe = g.filter(lambda x: len(x) > threshold)
    g = dataframe.groupby('Item')
    dataframe = g.filter(lambda x: len(x) > threshold)
    dataset = Dataset.load_from_df(dataframe, Reader())
    return dataset, dataframe


def global_mean_pred(trainset, testset, predset):
    """Compute a global mean prediction for a test set and a prediction set given a training set."""
    mean = trainset.global_mean
    return mean*np.ones([len(testset),1]), mean*np.ones([len(predset),1])


def user_item_func(dic,ele,mean):
    if ele in dic:
        return dic[ele]
    else:
        return mean


def user_mean_pred(trainset, testset, predset):
    """Compute a mean prediction per user for a test set and a prediction set given a training set."""
    mean = trainset.global_mean
    trainset = trainset.build_testset()
    trainset = pd.DataFrame(trainset, columns=['User', 'Item', 'Prediction'])
    g = trainset.groupby('User')
    user_mean = g.mean()
    dict_mean = user_mean.to_dict()['Prediction']
    usermean_test = np.array([user_item_func(dict_mean, p[0], mean) for p in testset])
    usermean_pred = np.array([user_item_func(dict_mean, p[0], mean) for p in predset])
    usermean_test = np.reshape(usermean_test, (len(usermean_test), 1))
    usermean_pred = np.reshape(usermean_pred, (len(usermean_pred), 1))
    return usermean_test, usermean_pred


def item_mean_pred(trainset,testset,predset):
    """Compute a mean prediction per item for a test set and a prediction set given a training set."""
    mean = trainset.global_mean
    trainset = trainset.build_testset()
    trainset = pd.DataFrame(trainset, columns=['User', 'Item', 'Prediction'])
    g = trainset.groupby('Item')
    item_mean  = g.mean()
    dict_mean = item_mean.to_dict()['Prediction']
    itemmean_test = np.array([user_item_func(dict_mean, p[1], mean) for p in testset])
    itemmean_pred = np.array([user_item_func(dict_mean, p[1], mean) for p in predset])
    itemmean_test = np.reshape(itemmean_test, (len(itemmean_test), 1))
    itemmean_pred = np.reshape(itemmean_pred, (len(itemmean_pred), 1))
    return itemmean_test, itemmean_pred


def train_model(algo, trainset, testset, predset):
    """Given an algorithm, return a test set and a prediction set by
    applying the learning algorithm on a training set."""

    def transform(prediction):
        pred = [p[3] for p in prediction]
        return np.reshape(pred, (len(pred), 1))

    algo.fit(trainset)
    pred_test = algo.test(testset)
    pred_sub  = algo.test(predset)
    return transform(pred_test), transform(pred_sub)


def coeff(pred_set, real_val):
    """Compute coefficients that minimize a set of prediction given the real predictions."""
    l = np.shape(pred_set)[1]
    x0 = np.ones([1, l])*1/l
    opt = optimize.minimize(calculate_weighted_mse, x0, args=(real_val, pred_set.T), method='SLSQP')
    return opt['x']


def calculate_weighted_mse(weights, real_label, prediction):
    """Calculate the MSE using weights of the predictions."""
    prediction = weights@prediction
    t = real_label - prediction
    return 1.0 * t.dot(t.T)


def calculate_rmse_round(real_label, prediction):
    """Calculate the RMSE by rounding the predictions."""
    mse = calculate_mse_round(real_label, prediction)
    return np.sqrt(mse/len(real_label))


def calculate_mse_round(real_label, prediction):
    """cCalculate the MSE by rounding the predictions."""
    t = real_label - np.round(prediction)
    return 1.0 * t.dot(t.T)
