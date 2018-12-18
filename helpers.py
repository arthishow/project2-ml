import pandas as pd
import scipy.sparse as sp
from surprise import Dataset, Reader
import numpy as np

def load_data_surprise(path,threshold):
    dataframe = pd.read_csv(path)
    dataframe['User'], dataframe['Item'] = dataframe['Id'].str.split('_', 1).str
    dataframe = dataframe[['User', 'Item', 'Prediction']]
    g = dataframe.groupby('User')
    dataframe = g.filter(lambda x: len(x) > threshold) 
    g = dataframe.groupby('Item')
    dataframe = g.filter(lambda x: len(x) > threshold) 
    dataset = Dataset.load_from_df(dataframe, Reader())
    
    return dataset, dataframe


def load_data_matrix(path):
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    with open(path, "r") as f:
        data = f.read().splitlines()[1:] #skip the header row

    data = [deal_line(line) for line in data]

    min_row, max_row, min_col, max_col = statistics(data)

    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings
def rounding(i):
    return round(i)
def calculate_rmse_pred(prediction):
    t = np.array([p[2]- rounding(p[3]) for p in prediction])
    return np.sqrt((1.0 * t.dot(t.T))/len(prediction))
def n_error(prediction):
    n=0
    g=0
    for p in prediction:
        if rounding(p[3])>3 and p[2]>3:
            if p[2]!=rounding(p[3]):
                n+=1
            g+=1
    return n/g
def calculate_r(prediction):
    n = np.zeros([5,5])
    err = np.zeros([5,5])
    for p in prediction:
        n[int(p[2])-1,int(rounding(p[3]))-1]+=1
    return n
def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - np.round(prediction)
    return 1.0 * t.dot(t.T)
