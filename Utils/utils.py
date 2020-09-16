import numpy as np
from spotlight.datasets import amazon, goodbooks, movielens
from spotlight.cross_validation import random_train_test_split as split
from spotlight.evaluation import rmse_score
from spotlight.interactions import Interactions
import torch.nn as nn
from spotlight.layers import ZeroEmbedding, ScaledEmbedding
import torch
import gc
import pickle


def dataset_loader(name):
    if name == 'goodbooks':
        data = goodbooks.get_goodbooks_dataset()
    elif name == 'amazon':
        data = amazon.get_amazon_dataset()
    elif name == 'netflix':
        with open('/home/vcoscrato/Documents/Data/netflix.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        data = movielens.get_movielens_dataset(name)
    train, test = split(data, random_state=np.random.RandomState(0), test_percentage=0.1)
    return train, test


class BiasNet(nn.Module):

    def __init__(self, num_users, num_items, sparse=False):

        super(BiasNet, self).__init__()

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return user_bias + item_bias


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

    data = Interactions(indices, coo_col, coo_val)
    with open('/home/vcoscrato/Documents/Data/netflix.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return 'Success!'


def load_models(path):
    models = {}
    with open(path + 'fitted/R.pkl', 'rb') as f:
        models['R'] = pickle.load(f)
    with open(path + 'fitted/empirical.pkl', 'rb') as f:
        models['Empirical'] = pickle.load(f)
    with open(path + 'fitted/ensemble.pkl', 'rb') as f:
        models['Ensemble'] = pickle.load(f)
    with open(path + 'fitted/resample.pkl', 'rb') as f:
        models['Resample'] = pickle.load(f)
    with open(path + 'fitted/CPMF.pkl', 'rb') as f:
        models['CPMF'] = pickle.load(f)
    with open(path + 'fitted/KorenSill.pkl', 'rb') as f:
        models['OrdRec'] = pickle.load(f)
    with open(path + 'fitted/double.pkl', 'rb') as f:
        models['Double'] = pickle.load(f)
    with open(path + 'fitted/linear.pkl', 'rb') as f:
        models['Linear'] = pickle.load(f)
    return models