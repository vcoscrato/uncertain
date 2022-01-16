import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.metrics.pairwise import cosine_distances


class Data(LightningDataModule):

    def __init__(self, data, users_on_test, test_ratio=0.2, val_ratio=0.1, implicit=False, batch_size=int(1e5)):
        super().__init__()
        self.implicit = implicit
        self.batch_size = batch_size

        data = data.sort_values('timestamps').drop(columns='timestamps')

        if self.implicit:
            data = data[data.score >= 4].drop('score', 1)

        # Drop user with too small profile
        length = data.user.value_counts().drop(columns='timestamps')
        data.drop(data.index[data.user.isin(length.index[length <= 5])], 0, inplace=True)

        # Drop items with < 5 ratings
        length = data.item.value_counts()
        data.drop(data.index[data.item.isin(length.index[length < 5])], 0, inplace=True)

        # Make sure user and item ids are consecutive integers
        data.user = data.user.factorize()[0]
        data.item = data.item.factorize()[0]

        # Shapes
        self.n_user = data.user.nunique()
        self.n_item = data.item.nunique()

        # Split
        rng = np.random.default_rng(0)
        self.test_users = np.sort(rng.choice(range(self.n_user), size=users_on_test, replace=False))
        test = data[data.user.isin(self.test_users)].groupby('user').apply(lambda x: x.tail(int(test_ratio * len(x)))
                                                                           ).reset_index(level=0, drop=True)
        self.train_val = data.drop(index=test.index)
        if self.implicit:
            csr = csr_matrix((np.ones_like(self.train_val.user), (self.train_val.item, self.train_val.user)),
                             shape=(self.n_item, self.n_user))
        else:
            csr = csr_matrix((self.train_val.score, (self.train_val.item, self.train_val.user)),
                             shape=(self.n_item, self.n_user))
        self.distances = cosine_distances(csr)
        val = self.train_val.groupby('user').apply(lambda x: x.tail(int(val_ratio * len(x)))).reset_index(level=0, drop=True)
        self.train = self.train_val.drop(index=val.index).to_numpy()
        self.test = test.to_numpy()
        self.val = val.to_numpy()

        # Heuristic measures
        self.user = self.train_val.groupby('user').agg({'item': 'size', 'score': 'var'})
        self.user.columns = ['support', 'variance']
        self.user['diversity'] = np.empty(self.n_user)
        for user in tqdm(range(self.n_user)):
            rated = self.train_val.item[self.train_val.user == user].to_numpy()
            self.user['diversity'][user] = self.distances[rated][:, rated].sum() / 2 / sum(range(len(rated)))

        self.item = self.train_val.groupby('item').agg({'user': 'size', 'score': 'var'})
        self.item.columns = ['support', 'variance']
        empty = np.where(~pd.Series(np.arange(self.n_item)).isin(self.item.index))[0]
        empty = pd.DataFrame(np.full((len(empty), 2), float('NaN')), index=empty, columns=['support', 'variance'])
        self.item = self.item.append(empty).sort_index().fillna(0)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        self.rand = {'users': torch.randint(0, self.n_user, (1000000,), generator=g_cpu),
                     'items': torch.randint(0, self.n_item, (1000000,), generator=g_cpu)}

        # Finish
        print(f'MovieLens data prepared: {self.n_user} users, {self.n_item} items.')
        print(f'{len(self.train)} train, {len(self.val)} validation and {len(self.test)} test interactions.')

    def to_ordinal(self):
        self.train[:, 2], self.score_labels = pd.factorize(self.train[:, 2], sort=True)
        self.val[:, 2] = pd.factorize(self.val[:, 2], sort=True)[0]

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, drop_last=False, shuffle=False, num_workers=4)
