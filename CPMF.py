"""
Factorization models for explicit feedback problems.
"""

import numpy as np
import torch
from tqdm import tqdm

from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids

from spotlight.torch_utils import gpu
from uncertain.metrics import rmse_score, rpi_score, graphs_score, precision_recall_rri_score, classification

from scipy.stats import spearmanr, pearsonr


class CPMFPar(torch.nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
        self.user_embeddings.weight.data.uniform_(-.01, .01)
        self.item_embeddings = torch.nn.Embedding(num_items, embedding_dim)
        self.item_embeddings.weight.data.uniform_(-.01, .01)

        self.user_gammas = torch.nn.Embedding(num_users, 1)
        self.user_gammas.weight.data.uniform_(-0.01, 0.01)
        self.item_gammas = torch.nn.Embedding(num_items, 1)
        self.item_gammas.weight.data.uniform_(-0.01, 0.01)

        self.var_activation = torch.nn.Softplus()

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids).squeeze()
        item_embedding = self.item_embeddings(item_ids).squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        user_gamma = self.user_gammas(user_ids).squeeze()
        item_gamma = self.item_gammas(item_ids).squeeze()

        var = self.var_activation(user_gamma + item_gamma)

        return dot, var


class CPMF(object):

    def __init__(self,
                 embedding_dim,
                 n_iter,
                 batch_size,
                 sigma,
                 learning_rate,
                 use_cuda):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._sigma = sigma
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

        self._optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self._learning_rate,
            weight_decay=self._batch_size/self._num_ratings/self._sigma
            )

        self.train_loss = []

    def _loss_func(self, observed_ratings, output):

        predicted_ratings, variance = output

        return (((observed_ratings - predicted_ratings) ** 2) / variance).sum() + torch.log(variance).sum()

    def fit(self, interactions, test=None, verbose=False):

        if not self._initialized:
            self._initialize(interactions)

        if test:
            self.test_loss = []

        user_ids_tensor = gpu(torch.from_numpy(interactions.user_ids.astype(np.int64)), self._use_cuda)
        item_ids_tensor = gpu(torch.from_numpy(interactions.item_ids.astype(np.int64)), self._use_cuda)
        ratings_tensor = gpu(torch.from_numpy(interactions.ratings), self._use_cuda)

        for epoch_num in tqdm(range(self._n_iter), desc='CPMF'):

            self.train_loss.append(0)

            for i in range(0, self._num_ratings, self._batch_size):

                self._optimizer.zero_grad()

                predictions = self._net(user_ids_tensor[i:i + self._batch_size],
                                        item_ids_tensor[i:i + self._batch_size])

                loss = self._loss_func(ratings_tensor[i:i + self._batch_size], predictions)
                loss.backward()
                self._optimizer.step()

                self.train_loss[-1] += loss.item()

            self.train_loss[-1] /= len(range(0, self._num_ratings, self._batch_size))

            self.train_loss[-1] /= self._num_ratings
            if test:
                predictions = self.predict(test.user_ids, test.item_ids)
                loss = self._loss_func(torch.tensor(test.ratings), torch.tensor(predictions))
                self.test_loss.append(loss.item()/len(test.ratings))

            if verbose:
                print('{}-th epoch loss: '.format(epoch_num), (self.train_loss[-1], self.test_loss[-1]))

    def predict(self, user_ids, item_ids=None):

        self._net.train(False)

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self._num_items,
                                                  self._use_cuda)

        out = self._net(user_ids, item_ids)

        return out[0].cpu().detach().numpy().flatten(), np.sqrt(out[1].cpu().detach().numpy().flatten())

    def evaluate(self, test, train):

        predictions = self.predict(test.user_ids, test.item_ids)
        error = np.abs(test.ratings - predictions[0])
        self.correlation = pearsonr(error, predictions[1]), spearmanr(error, predictions[1])

        self.rmse = rmse_score(predictions[0], test.ratings)
        self.rpi = -rpi_score(predictions, test.ratings)
        self.quantiles, self.intervals = graphs_score(predictions, test.ratings)

        p, r, e = precision_recall_rri_score(self, test, train, np.arange(1, 11))
        self.precision = p.mean(axis=0)
        self.recall = r.mean(axis=0)
        self.rri = -np.nanmean(e, axis=0)
        self.classification = classification(predictions, error, test)

from Utils.utils import dataset_loader
train, test = dataset_loader('10M', seed=0)

model = CPMF(embedding_dim=50, n_iter=50, sigma=0.02, learning_rate=.02, batch_size=int(1e6), use_cuda=False)
model.fit(train, test, verbose=True)
model.evaluate(test, train)
print('RMSE = {}; Precision = {}; Recall = {}'.format(model.rmse, model.precision[-1], model.recall[-1]))
print('RPI = {}; RRI = {}'.format(model.rpi, model.rri))

from matplotlib import pyplot as plt
preds = model.predict(test.user_ids, test.item_ids)
plt.hist(preds[1])
