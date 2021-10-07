import os
import h5py
import requests
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from scipy.sparse import csc_matrix
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


def cosine_similarities(mat):
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    return col_normed_mat.T * col_normed_mat


def split(interactions, test_items):
    train_idx = []
    val_idx = []
    test_idx = []
    for u in range(interactions.user.nunique()):
        idx = interactions.index[interactions.user == u]
        if len(idx) > test_items * 3:
            idx = idx[interactions.timestamp[idx].argsort()]
            test_idx += idx[-test_items:].tolist()
            val_idx += idx[-2 * test_items:-test_items].tolist()
            train_idx += idx[:-2 * test_items].tolist()
    interactions.drop('timestamp', 1, inplace=True)
    return interactions.loc[train_idx].to_numpy(), \
           interactions.loc[val_idx].to_numpy(), \
           interactions.loc[test_idx].to_numpy()


class MovieLens(LightningDataModule):

    def __init__(self, implicit=False, batch_size=512):
        super().__init__()
        self.implicit = implicit
        self.batch_size = batch_size

        path = 'data/movielens.hdf5'
        # Download
        if not os.path.isfile(path):

            url = 'https://github.com/maciejkula/recommender_datasets/releases/download/v0.2.0/movielens_20M.hdf5'
            req = requests.get(url, stream=True)
            with open(path, 'wb') as f:
                for chunk in req.iter_content(chunk_size=2 ** 20):
                    f.write(chunk)

        # Load
        with h5py.File(path, 'r') as data:
            data = pd.DataFrame({'user': data['/user_id'][:],
                                 'item': data['/item_id'][:],
                                 'score': data['/rating'][:],
                                 'timestamp': data['/timestamp'][:]})

        if self.implicit:
            data = data[data.score >= 4].drop('score', 1)

            # Remove users with profile < 6 (minimum 4 train, 4 val and 4 test instances)
            length = data.user.value_counts()
            data.drop(data.index[data.user.isin(length.index[length <= 12])], 0, inplace=True)

        # Make sure user and item ids are consecutive integers
        data.user = data.user.factorize()[0]
        data.item = data.item.factorize()[0]

        # Shapes
        self.n_user = data.user.nunique()
        self.n_item = data.item.nunique()

        # Split
        self.train, self.val, self.test = split(data, test_items=4)

        # Item similarities
        csc = csc_matrix((self.train[:, 2], (self.train[:, 0], self.train[:, 1])), shape=[self.n_user, self.n_item])
        self.item_similarity = cosine_similarities(csc).toarray()

        # Finish
        print(f'MovieLens data prepared: {self.n_user} users, {self.n_item} items.')
        print(f'{len(self.train)} Train interactions, {len(self.val)} validation and test interactions.')

    def to_ordinal(self):
        self.train[:, 2], self.score_labels = pd.factorize(self.train[:, 2], sort=True)
        self.val[:, 2] = pd.factorize(self.val[:, 2], sort=True)[0]

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, drop_last=True, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, drop_last=False, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, drop_last=False, shuffle=False, num_workers=8)
