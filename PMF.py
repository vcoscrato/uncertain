"""
Factorization models for explicit feedback problems.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids
from spotlight.evaluation import rmse_score, precision_recall_score
from spotlight.torch_utils import shuffle, minibatch, gpu, assert_no_grad


class BilinearNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.user_embeddings.weight.data.uniform_(-.01, .01)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_embeddings.weight.data.uniform_(-.01, .01)

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        return dot


class PMF(object):

    def __init__(self,
                 embedding_dim,
                 n_iter,
                 lambda_u,
                 lambda_v,
                 learning_rate,
                 batch_size,
                 use_cuda):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._lambda_u = lambda_u
        self._lambda_v = lambda_v
        self._batch_size = batch_size
        self._use_cuda = use_cuda

        self._num_users = None
        self._num_items = None
        self._num_ratings = None
        self._net = None
        self._optimizer = None

    def __repr__(self):

        return _repr_model(self)

    @property
    def _initialized(self):
        return self._net is not None

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._num_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items,
         self._num_ratings) = (interactions.num_users,
                               interactions.num_items,
                               len(interactions.ratings))

        self._net = gpu(BilinearNet(self._num_users,
                                    self._num_items,
                                    self._embedding_dim),
                        self._use_cuda)

        self._optimizer = optim.Adam(
            self._net.parameters(),
            lr=self._learning_rate
            )

    def _mse(self, observed_ratings, predicted_ratings):

        assert_no_grad(observed_ratings)

        return ((observed_ratings - predicted_ratings) ** 2).sum()

    def _add_penalty(self, size):

        penalty_u = (self._net.user_embeddings.weight ** 2).sum() * (self._lambda_u*size/self._num_ratings)
        penalty_v = (self._net.item_embeddings.weight ** 2).sum() * (self._lambda_v*size/self._num_ratings)

        return penalty_u + penalty_v

    def fit(self, interactions, verbose=False):

        if not self._initialized:
            self._initialize(interactions)

        user_ids_tensor = gpu(torch.from_numpy(interactions.user_ids.astype(np.int64)), self._use_cuda)
        item_ids_tensor = gpu(torch.from_numpy(interactions.item_ids.astype(np.int64)), self._use_cuda)
        ratings_tensor = gpu(torch.from_numpy(interactions.ratings), self._use_cuda)

        for epoch_num in range(self._n_iter):

            for i in range(0, self._num_ratings, self._batch_size):

                self._optimizer.zero_grad()

                predictions = self._net(user_ids_tensor[i:i + self._batch_size],
                                        item_ids_tensor[i:i + self._batch_size])

                loss = self._mse(ratings_tensor[i:i + self._batch_size], predictions)
                loss += self._add_penalty(len(user_ids_tensor[i:i + self._batch_size]))
                loss.backward()
                self._optimizer.step()

    def predict(self, user_ids, item_ids=None):

        self._check_input(user_ids, item_ids, allow_items_none=True)

        self._net.train(False)

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self._num_items,
                                                  self._use_cuda)
        print(user_ids, user_ids)

        out = self._net(user_ids, item_ids)

        return out.cpu().detach().numpy().flatten()

    def evaluate(self, test, train):

        self.rmse = rmse_score(self, test)

        p, r = precision_recall_score(self, test, train, np.arange(1, 11))
        self.precision = p.mean(axis=0)
        self.recall = r.mean(axis=0)



from Utils.utils import dataset_loader
train, test = dataset_loader('1M')

model = PMF(embedding_dim=50, n_iter=1, lambda_u=6, lambda_v=6, learning_rate=.1, batch_size=1000000, use_cuda=True)
train_rmse = []
test_rmse = []
for i in range(100):
    model.fit(train, verbose=True)
    if (i+1) % 10 == 0:
        train_rmse.append(rmse_score(model, train))
        test_rmse.append(rmse_score(model, test))
model.evaluate(test, train)
print(train_rmse)
print(test_rmse)
