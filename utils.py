import numpy as np
from uncertain.datasets import amazon, goodbooks, movielens
from uncertain.cross_validation import random_train_test_split as split
from uncertain.interactions import Interactions
import gc
import pickle


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
    train, test = split(data, test_percentage=0.1, random_state=seed)
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

    data = Interactions(indices, coo_col, coo_val)
    with open('/home/vcoscrato/Documents/Data/netflix.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return 'Success!'
