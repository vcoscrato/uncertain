"""
Factorization models for explicit feedback problems.
"""

import numpy as np
from pandas import factorize

import torch
import torch.nn as nn
import torch.optim as optim

from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids

from spotlight.torch_utils import gpu
from Utils.metrics import rmse_score, rpi_score, rri_score, graphs_score, precision_recall_rri_score


class KorenSillNet(nn.Module):

    def __init__(self, num_users, num_items, num_labels, embedding_dim):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.user_embeddings.weight.data.uniform_(-.01, .01)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_embeddings.weight.data.uniform_(-.01, .01)

        self.item_biases = nn.Embedding(num_items, 1)
        self.item_biases.weight.data.zero_()

        self.user_betas = nn.Embedding(num_users, num_labels-1)
        self.user_betas.weight.data.zero_()

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids).squeeze()
        item_embedding = self.item_embeddings(item_ids).squeeze()

        item_bias = self.item_biases(item_ids).squeeze()
        y = ((user_embedding * item_embedding).sum(1) + item_bias).reshape(-1, 1)

        user_beta = self.user_betas(user_ids)
        user_beta[:, 1:] = torch.exp(user_beta[:, 1:])
        user_distribution = torch.div(1, 1 + torch.exp(y - user_beta.cumsum(1)))

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
                 l2,
                 use_cuda):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
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
                                 gpu(torch.from_numpy(np.unique(interactions.ratings)), self._use_cuda))

        self._net = gpu(KorenSillNet(self._num_users,
                                     self._num_items,
                                     len(self._rating_labels),
                                     self._embedding_dim),
                        self._use_cuda)
        '''
        self._optimizer = optim.Adam(
            self._net.parameters(),
            lr=self._learning_rate,
            weight_decay=self._l2
            )
        '''
        self._optimizer = torch.optim.Adam(
            [{'params': self._net.user_embeddings.parameters(), 'weight_decay': self._l2},
             {'params': self._net.item_embeddings.parameters(), 'weight_decay': self._l2},
             {'params': self._net.item_biases.parameters(), 'weight_decay': self._l2},
             {'params': self._net.user_betas.parameters(), 'weight_decay': self._l2}],
            lr=self._learning_rate
            )

        self.train_loss = []

    def _loss(self, observed_ratings, output):

        return -output[range(len(output)), observed_ratings].mean()

    def fit(self, interactions, test=None, verbose=False):

        if not self._initialized:
            self._initialize(interactions)

        if test:
            self.test_loss = []
            test_users = gpu(torch.from_numpy(test.user_ids.astype(np.int64)), self._use_cuda)
            test_items = gpu(torch.from_numpy(test.item_ids.astype(np.int64)), self._use_cuda)
            test_ratings = gpu(torch.from_numpy(factorize(test.ratings, sort=True)[0]), self._use_cuda)

        user_ids_tensor = gpu(torch.from_numpy(interactions.user_ids.astype(np.int64)), self._use_cuda)
        item_ids_tensor = gpu(torch.from_numpy(interactions.item_ids.astype(np.int64)), self._use_cuda)
        ratings_tensor = gpu(torch.from_numpy(factorize(interactions.ratings, sort=True)[0]), self._use_cuda)

        for epoch_num in range(self._n_iter):

            self.train_loss.append(0)
            idx = torch.randperm(self._num_ratings)
            user_ids_tensor = user_ids_tensor[idx]
            item_ids_tensor = item_ids_tensor[idx]
            ratings_tensor = ratings_tensor[idx]

            for i in range(0, self._num_ratings, self._batch_size):

                self._optimizer.zero_grad()

                predictions = self._net(user_ids_tensor[i:i + self._batch_size],
                                        item_ids_tensor[i:i + self._batch_size])

                loss = self._loss(ratings_tensor[i:i + self._batch_size], predictions)
                loss.backward()
                self._optimizer.step()

                self.train_loss[-1] += loss.item()

            self.train_loss[-1] /= len(range(0, self._num_ratings, self._batch_size))

            if test:
                self._net.train(False)
                predictions = self._net(test_users, test_items)
                self.test_loss.append(self._loss(test_ratings, predictions).item())

            if verbose:
                if test:
                    print('Epoch {} loss: '.format(epoch_num+1), (self.train_loss[-1], self.test_loss[-1]))
                else:
                    print('Epoch {} loss: '.format(epoch_num + 1), (self.train_loss[-1]))

    def predict(self, user_ids, item_ids=None, dist=False):

        self._net.train(False)

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self._num_items,
                                                  self._use_cuda)

        out = self._net(user_ids, item_ids)

        if dist:
            return out.cpu().detach().numpy()

        # Most probable rating
        #mean = self._rating_labels[(out.argmax(1))]
        #confidence = out.max(1)[0]

        # Average ranking
        mean = (out * self._rating_labels).sum(1)
        var = ((out * self._rating_labels**2).sum(1) - mean**2).abs()
        #confidence = var.max() - var

        return mean.cpu().detach().numpy(), var.cpu().detach().numpy()

    def evaluate(self, test, train):

        preds = self.predict(test.user_ids, test.item_ids)

        self.rmse = rmse_score(preds[0], test.ratings)
        self.rpi = rpi_score(preds, test.ratings)
        self.quantiles, self.intervals = graphs_score(preds, test.ratings)

        p, r, e = precision_recall_rri_score(self, test, train, np.arange(1, 11))
        self.precision = p.mean(axis=0)
        self.recall = r.mean(axis=0)
        self.rri = np.nanmean(e, axis=0)


from Utils.utils import dataset_loader
dataset = '10M'
train, test = dataset_loader(dataset)
train.ratings = train.ratings*2
test.ratings = test.ratings*2
if dataset == '1M':
    wd = 2e-6
    n_inter = 200
    lr = 0.02
elif dataset == '10M':
    wd = 1e-6
    n_inter = 20
    lr = 0.02
else:
    wd = 1e-7
    n_inter = 20

model = KorenSill(embedding_dim=50, n_iter=n_inter, learning_rate=lr, batch_size=int(1e6), l2=wd, use_cuda=True)
model.fit(train, test, verbose=True)
model.evaluate(test, train)
print(model.rmse, model.rpi)
print(model.precision)
print(model.rri)

'''
self = model
preds = model.predict(test.user_ids, test.item_ids)
user_ids, item_ids = _predict_process_ids(test.user_ids, test.item_ids,
                                          self._num_items,
                                          self._use_cuda)
out = model._net(user_ids, item_ids).detach()
mean = (out * self._rating_labels).sum(1)
var = (out * self._rating_labels**2).sum(1) - mean**2
confidence = var.max() - var
'''

user_embedding = model._net.user_embeddings.weight[test.user_ids]
item_embedding = model._net.item_embeddings.weight[test.item_ids]
item_bias = model._net.item_biases.weight[test.item_ids].squeeze()
user_beta = model._net.user_betas.weight[test.user_ids]

y = ((user_embedding * item_embedding).sum(1) + item_bias).reshape(-1, 1)
user_beta[:, 1:] = torch.exp(user_beta[:, 1:])
user_distribution = torch.div(1, 1 + torch.exp(y - user_beta.cumsum(1)))

ones = torch.ones((len(user_distribution), 1), device=user_beta.device)
user_distribution = torch.cat((user_distribution, ones), 1)

user_distribution[:, 1:] -= user_distribution[:, :-1]
print(user_distribution)
print(test.ratings[10:20])

mean = (user_distribution * model._rating_labels).sum(1)
var = ((user_distribution * model._rating_labels ** 2).sum(1) - mean ** 2)
from matplotlib import pyplot as plt
f, ax = plt.subplots(figsize=(10, 5))
ax.plot(var.cpu().detach().numpy(), mean.cpu().detach().numpy(), 'o', markersize=2)
ax.set_ylabel('Predicted')
ax.set_xlabel('Variance')