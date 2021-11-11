import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class Data(LightningDataModule):

    def __init__(self, data, implicit=False, batch_size=int(1e5)):
        super().__init__()
        self.implicit = implicit
        self.batch_size = batch_size

        data = data.sort_values('timestamps').drop(columns='timestamps')

        if self.implicit:
            data = data[data.score >= 4].drop('score', 1)

        # Remove users with profile < 12 (minimum 4 train, 4 val and 4 test instances)
        length = data.user.value_counts().drop(columns='timestamps')
        data.drop(data.index[data.user.isin(length.index[length <= 12])], 0, inplace=True)

        # Make sure user and item ids are consecutive integers
        data.user = data.user.factorize()[0]
        data.item = data.item.factorize()[0]

        # Shapes
        self.n_user = data.user.nunique()
        self.n_item = data.item.nunique()

        # Split
        test = data.groupby('user').tail(4)
        self.train_val = data.drop(index=test.index)
        if self.implicit:
            self.csr = csr_matrix((np.ones_like(self.train_val.user), (self.train_val.item, self.train_val.user)),
                                  shape=(self.n_item, self.n_user))
        else:
            self.csr = csr_matrix((self.train_val.score, (self.train_val.item, self.train_val.user)),
                                  shape=(self.n_item, self.n_user))
        val = self.train_val.groupby('user').tail(4)
        self.train = self.train_val.drop(index=val.index).to_numpy()
        self.test = test.to_numpy()
        self.val = val.to_numpy()

        # Finish
        print(f'MovieLens data prepared: {self.n_user} users, {self.n_item} items.')
        print(f'{len(self.train)} Train interactions, {len(self.val)} validation and test interactions.')

    def to_ordinal(self):
        self.train[:, 2], self.score_labels = pd.factorize(self.train[:, 2], sort=True)
        self.val[:, 2] = pd.factorize(self.val[:, 2], sort=True)[0]

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, drop_last=False, shuffle=False, num_workers=4)
