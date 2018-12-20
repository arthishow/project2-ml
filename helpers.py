import pandas as pd
from scipy import optimize
from surprise import Dataset, Reader,accuracy
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

def transform(prediction):
    pred=[p[3] for p in prediction]
    return np.reshape(pred,(len(pred),1))

def global_mean_pred(trainset,testset,predset):
    mean=trainset.global_mean
    return mean*np.ones([len(testset),1]),mean*np.ones([len(predset),1]),

def user_item_func(dic,ele,mean):
    if ele in dic:
        return dic[ele]
    else:
        return mean

def user_mean_pred(trainset,testset,predset):
    mean=trainset.global_mean
    trainset=trainset.build_testset()
    trainset=pd.DataFrame(trainset, columns=['User', 'Item', 'Prediction'])
    g = trainset.groupby('User')
    user_mean  = g.mean()
    dict_mean = user_mean.to_dict()['Prediction']
    usermean_test=np.array([user_item_func(dict_mean,p[0],mean) for p in testset])
    usermean_pred=np.array([user_item_func(dict_mean,p[0],mean) for p in predset])
    usermean_test=np.reshape(usermean_test,(len(usermean_test),1))
    usermean_pred=np.reshape(usermean_pred,(len(usermean_pred),1))
    return usermean_test,usermean_pred

def item_mean_pred(trainset,testset,predset):
    mean = trainset.global_mean
    trainset = trainset.build_testset()
    trainset = pd.DataFrame(trainset, columns=['User', 'Item', 'Prediction'])
    g = trainset.groupby('Item')
    item_mean  = g.mean()
    dict_mean = item_mean.to_dict()['Prediction']
    itemmean_test = np.array([user_item_func(dict_mean,p[1],mean) for p in testset])
    itemmean_pred = np.array([user_item_func(dict_mean,p[1],mean) for p in predset])
    itemmean_test = np.reshape(itemmean_test,(len(itemmean_test),1))
    itemmean_pred = np.reshape(itemmean_pred,(len(itemmean_pred),1))
    return itemmean_test,itemmean_pred

def train_model(algo,trainset,testset,predset):
    print('Fit model...')
    algo.fit(trainset)
    print('Make predictions...')
    pred_test = algo.test(testset)
    pred_sub  = algo.test(predset)
    return transform(pred_test),transform(pred_sub)
    
def coeff(pred_set,real_val):
    l=np.shape(pred_set)[1]
    x0=np.ones([1,l])*1/l
    opt=optimize.minimize(calculate_poud_mse,x0,args=(real_val,pred_set.T),method='SLSQP')
    return opt['x']
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
    t = real_label - prediction
    return 1.0 * t.dot(t.T)
def calculate_mse_round(real_label, prediction):
    """calculate MSE."""
    t = real_label - np.round(prediction)
    return 1.0 * t.dot(t.T)
def calculate_poud_mse(poud,real_label,prediction):
    prediction= poud @ prediction
    t = real_label - prediction
    return 1.0 * t.dot(t.T)    
def calculate_poud_rmse_round(poud,real_label,prediction):
    prediction= poud @ prediction
    t = real_label - np.round(prediction)
    return np.sqrt((1/len(real_label))* t.dot(t.T)) 
def calculate_rmse(real_label, prediction):
    """calculate MSE."""
    mse=calculate_mse(real_label, prediction)
    return np.sqrt(mse/len(real_label))
def calculate_rmse_round(real_label, prediction):
    """calculate MSE."""
    mse=calculate_mse_round(real_label, prediction)
    return np.sqrt(mse/len(real_label))
