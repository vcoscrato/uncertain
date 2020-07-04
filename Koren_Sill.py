"""
Factorization models for explicit feedback problems.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids

from spotlight.torch_utils import gpu
from Utils.metrics import rmse_score, epi_score, eri_score, graphs_score, precision_recall_eri_score


class KorenSillNet(nn.Module):

    def __init__(self, num_users, num_items, num_labels, embedding_dim=32):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.user_embeddings.weight.data.uniform_(-.01, .01)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_embeddings.weight.data.uniform_(-.01, .01)

        self.item_biases = nn.Embedding(num_items, 1)
        self.item_biases.weight.data.zero_()

        self.user_betas = nn.Embedding(num_users, num_labels-1)
        self.user_betas.weight.data.uniform_()

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids).squeeze()
        item_embedding = self.item_embeddings(item_ids).squeeze()

        item_bias = self.item_biases(item_ids).squeeze()

        y = ((user_embedding * item_embedding).sum(1) + item_bias).reshape(-1, 1)

        user_beta = self.user_betas(user_ids)
        user_beta[:, 1:] = torch.exp(user_beta[:, 1:])
        user_distribution = 1 / (1 + torch.exp(y - user_beta.cumsum(1)))

        ones = torch.ones((len(user_distribution), 1), device=user_beta.device)
        user_distribution = torch.cat((user_distribution, ones), 1)

        user_distribution[:, 1:] -= user_distribution[:, :-1]

        return user_distribution


class KorenSill(object):

    def __init__(self,
                 embedding_dim,
                 n_iter,
                 batch_size,
                 learning_rate,
                 use_cuda):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._use_cuda = use_cuda

        self._num_users = None
        self._num_items = None
        self._rating_labels = None
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
         self._num_ratings,
         self._rating_labels) = (interactions.num_users,
                                 interactions.num_items,
                                 len(interactions.ratings),
                                 np.unique(interactions.ratings))

        self._net = gpu(KorenSillNet(self._num_users,
                                     self._num_items,
                                     len(self._rating_labels),
                                     self._embedding_dim),
                        self._use_cuda)

        self._optimizer = optim.Adam(
            self._net.parameters(),
            lr=self._learning_rate,
            weight_decay=1.5e-6
            )

        self.train_loss = []

    def _loss(self, observed_ratings, output):

        return -output[range(len(output)), observed_ratings].mean()

    def fit(self, interactions, test=None, verbose=False):

        if not self._initialized:
            self._initialize(interactions)

        if test:
            self.test_loss = []
            test_ratings = torch.from_numpy(test.ratings - 1).long()

        user_ids_tensor = gpu(torch.from_numpy(interactions.user_ids.astype(np.int64)), self._use_cuda)
        item_ids_tensor = gpu(torch.from_numpy(interactions.item_ids.astype(np.int64)), self._use_cuda)
        ratings_tensor = gpu((torch.from_numpy(interactions.ratings) - 1).long(), self._use_cuda)

        for epoch_num in range(self._n_iter):

            self.train_loss.append(0)

            for i in range(0, self._num_ratings, self._batch_size):

                self._optimizer.zero_grad()

                predictions = self._net(user_ids_tensor[i:i + self._batch_size],
                                        item_ids_tensor[i:i + self._batch_size])

                loss = self._loss(ratings_tensor[i:i + self._batch_size], predictions)
                loss.backward()
                self._optimizer.step()

                self.train_loss[-1] += loss.item()

            if test:
                predictions = model.predict(test.user_ids, test.item_ids)
                self.test_loss.append(self._loss(test_ratings, predictions))

            if verbose:
                print('Epoch loss: ', (self.train_loss[-1], self.test_loss[-1]))


    def predict(self, user_ids, item_ids=None):

        self._net.train(False)

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self._num_items,
                                                  self._use_cuda)

        out = self._net(user_ids, item_ids)

        return out.cpu().detach().numpy()

    def evaluate(self, test, train):

        preds = self.predict(test.user_ids, test.item_ids)
        #point_pred = (np.argmax(preds, axis=1) + 1).astype(float)
        point_pred = (preds * self._rating_labels).sum(1)

        self.rmse = rmse_score(point_pred, test.ratings)
        self.epi = epi_score(preds, test.ratings)
        self.quantiles, self.intervals = graphs_score(preds, test.ratings)

        p, r, e = precision_recall_eri_score(self, test, train, np.arange(1, 11))
        self.precision = p.mean(axis=0)
        self.recall = r.mean(axis=0)
        self.eri = np.nanmean(e, axis=0)


from Utils.utils import dataset_loader
train, test = dataset_loader('1M')

model = KorenSill(embedding_dim=200, n_iter=30, learning_rate=.05, batch_size=100000, use_cuda=True)
model.fit(train, test, verbose=True)

