import gc
import pickle
import torch
import numpy as np
from uncertain.datasets import amazon, goodbooks, movielens
from uncertain.cross_validation import random_train_test_split as split
from uncertain.interactions import ExplicitInteractions
from uncertain.metrics import rmse_score, recommendation_score, correlation, rpi_score, classification


def dataset_loader(name, seed):
    if name == 'goodbooks':
        data = goodbooks.get_goodbooks_dataset()
    elif name == 'amazon':
        data = amazon.get_amazon_dataset()
    elif name == 'netflix':
        with open('/home/vcoscrato/Documents/Data/netflix.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        data = movielens.get_movielens_dataset(name)
    train, test = split(data, test_percentage=0.2, random_state=seed)
    return train, test


def generate_netflix():
    files = [
        '/home/vcoscrato/Documents/Data/combined_data_1.txt',
        '/home/vcoscrato/Documents/Data/combined_data_2.txt',
        '/home/vcoscrato/Documents/Data/combined_data_3.txt',
        '/home/vcoscrato/Documents/Data/combined_data_4.txt',
    ]

    coo_row = []
    coo_col = []
    coo_val = []

    for file_name in files:
        print('processing {0}'.format(file_name))
        with open(file_name, "r") as f:
            movie = -1
            for line in f:
                if line.endswith(':\n'):
                    movie = int(line[:-2]) - 1
                    continue
                assert movie >= 0
                splitted = line.split(',')
                user = int(splitted[0])
                rating = float(splitted[1])
                coo_row.append(user)
                coo_col.append(movie)
                coo_val.append(rating)
        gc.collect()

    print('transformation...')

    coo_val = np.array(coo_val, dtype=np.float32)
    coo_col = np.array(coo_col, dtype=np.int32)
    coo_row = np.array(coo_row)
    user, indices = np.unique(coo_row, return_inverse=True)

    gc.collect()

    data = ExplicitInteractions(indices, coo_col, coo_val)
    with open('/home/vcoscrato/Documents/Data/netflix.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return 'Success!'


def evaluate(model, test, train, uncertainty):

    out = {}
    est = model.predict(test.user_ids, test.item_ids)
    if type(est) is tuple:
        est, unc = est
    p, r, a, s = recommendation_score(model, test, train, max_k=10)

    out['RMSE'] = rmse_score(est, test.ratings)
    out['Precision'] = p.mean(axis=0)
    out['Recall'] = r.mean(axis=0)

    if uncertainty:
        error = torch.abs(test.ratings - est)
        quantiles = torch.quantile(unc, torch.linspace(0, 1, 21, device=unc.device, dtype=unc.dtype))
        out['Quantile RMSE'] = torch.zeros(20)
        for idx in range(20):
            ind = torch.bitwise_and(quantiles[idx] <= unc, unc < quantiles[idx + 1])
            out['Quantile RMSE'][idx] = torch.sqrt(torch.square(error[ind]).mean())
        quantiles = torch.quantile(a, torch.linspace(0, 1, 21, device=a.device, dtype=a.dtype))
        out['Quantile MAP'] = torch.zeros(20)
        for idx in range(20):
            ind = torch.bitwise_and(quantiles[idx] <= a, a < quantiles[idx + 1])
            out['Quantile MAP'][idx] = p[ind, -1].mean()
        out['RRI'] = s.nansum(0) / (~s.isnan()).float().sum(0)
        out['Correlation'] = correlation(error, unc)
        out['RPI'] = rpi_score(error, unc)
        out['Classification'] = classification(error, unc)

    return out
