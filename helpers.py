import pandas as pd
import scipy.sparse as sp
from surprise import Dataset, Reader


def load_data_surprise(path):
    dataframe = pd.read_csv(path)
    dataframe['User'], dataframe['Item'] = dataframe['Id'].str.split('_', 1).str
    dataframe = dataframe[['User', 'Item', 'Prediction']]
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
