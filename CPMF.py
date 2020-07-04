"""
Factorization models for explicit feedback problems.
"""

import numpy as np
import pandas as pd
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids

from spotlight.torch_utils import gpu
from Utils.metrics import rmse_score, epi_score, eri_score, graphs_score, _get_precision_recall_eri

from scipy.stats import spearmanr, pearsonr


class CPMFPar(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.user_embeddings.weight.data.uniform_(-.01, .01)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_embeddings.weight.data.uniform_(-.01, .01)

        self.user_gammas = torch.nn.Embedding(num_users, 5)
        self.user_gammas.weight.data.uniform_(-0.01, 0.01)
        self.item_gammas = torch.nn.Embedding(num_items, 5)
        self.item_gammas.weight.data.uniform_(-0.01, 0.01)

        self.var_activation = torch.nn.Softplus()

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids).squeeze()
        item_embedding = self.item_embeddings(item_ids).squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        user_gamma = self.user_gammas(user_ids).squeeze()
        item_gamma = self.item_gammas(item_ids).squeeze()

        var = self.var_activation((user_gamma * item_gamma).sum(1))

        return dot, var


class CPMF(object):

    def __init__(self,
                 embedding_dim,
                 n_iter,
                 batch_size,
                 sigma_u,
                 sigma_v,
                 learning_rate,
                 use_cuda):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._sigma_u = sigma_u
        self._sigma_v = sigma_v
        self._use_cuda = use_cuda

        self._num_users = None
        self._num_items = None
        self._net = None
        self._optimizer = None

        self.train_loss = None
        self.test_loss = None

    def __repr__(self):

        return _repr_model(self)

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items,
         self._num_ratings) = (interactions.num_users,
                               interactions.num_items,
                               len(interactions.ratings))

        self._net = gpu(CPMFPar(self._num_users,
                                self._num_items,
                                self._embedding_dim),
                        self._use_cuda)

        self._optimizer = optim.Adam(
            self._net.parameters(),
            lr=self._learning_rate,
            weight_decay=0.1
            )

        self.train_loss = []

    def _mse(self, observed_ratings, output):

        predicted_ratings, variance = output

        return (((observed_ratings - predicted_ratings) ** 2) / variance).sum() + torch.log(variance).sum()

    def _add_penalty(self, size):

        penalty_u = (self._net.user_embeddings.weight ** 2).sum() / self._sigma_u * (size/self._num_ratings)
        penalty_v = (self._net.item_embeddings.weight ** 2).sum() / self._sigma_v * (size/self._num_ratings)

        return penalty_u + penalty_v

    def fit(self, interactions, test=None, verbose=False):

        if not self._initialized:
            self._initialize(interactions)

        if test:
            self.test_loss = []

        user_ids_tensor = gpu(torch.from_numpy(interactions.user_ids.astype(np.int64)), self._use_cuda)
        item_ids_tensor = gpu(torch.from_numpy(interactions.item_ids.astype(np.int64)), self._use_cuda)
        ratings_tensor = gpu(torch.from_numpy(interactions.ratings), self._use_cuda)

        for epoch_num in range(self._n_iter):

            self.train_loss.append(0)

            for i in range(0, self._num_ratings, self._batch_size):

                self._optimizer.zero_grad()

                predictions = self._net(user_ids_tensor[i:i + self._batch_size],
                                        item_ids_tensor[i:i + self._batch_size])

                loss = self._mse(ratings_tensor[i:i + self._batch_size], predictions)
                #loss += self._add_penalty(len(user_ids_tensor[i:i + self._batch_size]))
                loss.backward()
                self._optimizer.step()

                self.train_loss[-1] += loss.item()

            if test:
                predictions = model.predict(test.user_ids, test.item_ids)
                loss = self._mse(torch.tensor(test.ratings), torch.tensor(predictions))
                #loss += self._add_penalty(len(test.ratings))
                self.test_loss.append(loss.item())

            if verbose:
                print('{}-th epoch loss: '.format(i), (self.train_loss[-1]/self._num_ratings, self.test_loss[-1]/len(test.ratings)))

    def predict(self, user_ids, item_ids=None):

        self._net.train(False)

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self._num_items,
                                                  self._use_cuda)

        out = self._net(user_ids, item_ids)

        return out[0].cpu().detach().numpy().flatten(), out[1].cpu().detach().numpy().flatten()

    def evaluate(self, test, train):

        preds = self.predict(test.user_ids, test.item_ids)
        self.rmse = rmse_score(preds[0], test.ratings)
        self.epi = epi_score(preds, test.ratings)
        self.quantiles, self.intervals = graphs_score(preds, test.ratings)

        #df = pd.DataFrame(data={'user': test.user_ids, 'item': test.item_ids, 'error': np.square(preds[0] - test.ratings)})
        #user_rmse = df.groupby('user').mean()['error']
        #user_var = self._net.user_gammas.weight.data.cpu().detach()[torch.LongTensor(user_rmse.index)].numpy().flatten()
        #user_rmse = np.array(user_rmse)
        #self.user_rmse_var_corr = (pearsonr(user_rmse, user_var)[0], spearmanr(user_rmse, user_var)[0])

        #item_rmse = df.groupby('item').mean()['error']
        #item_var = self._net.item_gammas.weight.data.cpu().detach()[torch.LongTensor(item_rmse.index)].numpy().flatten()
        #item_rmse = np.array(item_rmse)
        #self.item_rmse_var_corr = (pearsonr(item_rmse, item_var)[0], spearmanr(item_rmse, item_var)[0])

        idx = test.ratings >= 4
        test_ = deepcopy(test)
        test_.user_ids = test_.user_ids[idx]
        test_.item_ids = test_.item_ids[idx]
        test_.ratings = test_.ratings[idx]
        test_ = test_.tocsr()
        train_ = train.tocsr()

        avg_reliability, std_reliability = preds[1].mean(), preds[1].std()
        k = np.arange(1, 11)

        uid = []
        precision = []
        recall = []
        eri = []

        for user_id, row in enumerate(test_):

            if not len(row.indices):
                continue

            predictions, reliabilities = model.predict(user_id)
            predictions *= -1

            rated = train_[user_id].indices
            predictions[rated] = np.infty

            predictions = predictions.argsort()

            targets = row.indices

            user_precision, user_recall, user_eri = zip(*[
                _get_precision_recall_eri(predictions, reliabilities, avg_reliability, std_reliability, targets, x)
                for x in k
            ])

            uid.append(user_id)
            precision.append(user_precision)
            recall.append(user_recall)
            eri.append(user_eri)

        precision = np.array(precision).squeeze()
        recall = np.array(recall).squeeze()
        self.eri = np.nanmean(np.array(eri).squeeze(), 0)

        #map = precision[:, 9].flatten()
        #user_var = self._net.user_gammas.weight.data.cpu().detach()[torch.LongTensor(uid)].numpy().flatten()
        #self.user_map_var_corr = (pearsonr(map, user_var)[0], spearmanr(map, user_var)[0])

        self.precision = precision.mean(0)
        self.recall = recall.mean(0)


from Utils.utils import dataset_loader
train, test = dataset_loader('1M')

model = CPMF(embedding_dim=50, n_iter=100, sigma_u=0.4, sigma_v=0.4, learning_rate=.02, batch_size=2048, use_cuda=True)
model.fit(train, test, verbose=True)
model.evaluate(test, train)
print('RMSE = {}; Precision = {}; Recall = {}'.format(model.rmse, model.precision[-1], model.recall[-1]))
print('EPI = {}; ERI = {}'.format(model.epi, model.eri))

from matplotlib import pyplot as plt
preds = model.predict(test.user_ids, test.item_ids)
plt.hist(preds[1])

'''
train_rmse = []
test_rmse = []

    if (i+1) % 25 == 0:
        train_rmse.append(rmse_score(model.predict(train.user_ids, train.item_ids)[0], train.ratings))
        test_rmse.append(rmse_score(model.predict(test.user_ids, test.item_ids)[0], test.ratings))
model.evaluate(test, train)
print(train_rmse)
print(test_rmse)
print(model.epi, model.eri)



f, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].hist(model._net.user_gammas.weight.data.cpu().detach().numpy().flatten())
ax[1].hist(model._net.item_gammas.weight.data.cpu().detach().numpy().flatten())

preds = model.predict(test.user_ids, test.item_ids)
np.corrcoef(preds[0], preds[1])

test_csr = test.tocsr()




spearmanr(mse, var)

'''