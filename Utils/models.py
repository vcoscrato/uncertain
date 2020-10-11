import numpy as np
import torch
import pandas as pd
from copy import deepcopy
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from spotlight.cross_validation import random_train_test_split as split
from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids
from spotlight.torch_utils import gpu
from spotlight.layers import ZeroEmbedding

from .metrics import rmse_score, rpi_score, graphs_score, precision_recall_rri_score, rri_score


class Empirical(object):
    def __init__(self, base_model, type):
        self.base_model = base_model
        self.type = type

    def fit(self, train):
        train = train.tocsr()
        if self.type == 'user_support':
            self.uncertainty = -train.getnnz(1).astype(np.float32)
        elif self.type == 'item_support':
            self.uncertainty = -train.getnnz(0).astype(np.float32)
        elif self.type == 'item_variance':
            square = train.copy();
            square.data **= 2
            self.uncertainty = np.array(square.mean(0)).flatten() - np.array(train.mean(0)).flatten() ** 2
        else:
            Exception('type must be one of ("user_support", "item_support", "item_variance").')
        
    def predict(self, user_ids, item_ids=None):
        user_ids_, item_ids_ = _predict_process_ids(user_ids, item_ids,
                                                  self.base_model._num_items, False)
        if 'user' in self.type:
            return self.base_model.predict(user_ids, item_ids), self.uncertainty[user_ids_]
        else:
            return self.base_model.predict(user_ids, item_ids), self.uncertainty[item_ids_]

    def evaluate(self, test, train, k):

        predictions = self.predict(test.user_ids, test.item_ids)
        self.rpi = -rpi_score(predictions, test.ratings)
        error = np.abs(test.ratings - predictions[0])
        self.correlation = pearsonr(error, predictions[1]), spearmanr(error, predictions[1])
        p, r, rri = precision_recall_rri_score(self, test, train, k)
        self.precision = p.mean(axis=0)
        self.recall = r.mean(axis=0)
        self.rri = -np.nanmean(rri, axis=0)
        self.quantiles, self.intervals = graphs_score(predictions, test.ratings)
        self.quantiles, self.intervals = self.quantiles[::-1], self.intervals[::-1]


class Ensemble(object):
    def __init__(self, base_model, n_models):
        base_model.verbose = False
        self.models = [base_model]
        self.n_models = n_models

    def fit(self, train, test):
        for i in tqdm(range(1, self.n_models), desc='Ensemble'):
            self.models.append(deepcopy(self.models[0]))
            self.models[i]._initialize(train)
            self.models[i].fit(train)

    def predict(self, user_ids, item_ids=None):
        self.models[0]._check_input(user_ids, item_ids, allow_items_none=True)
        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self.models[0]._num_items, self.models[0]._use_cuda)
        if np.isscalar(user_ids):
            user_ids = [user_ids]
        predictions = np.empty((len(user_ids), len(self.models)))
        for idx, model in enumerate(self.models):
            model._net.train(False)
            predictions[:, idx] = model._net(user_ids, item_ids).detach().cpu().numpy()
        estimates = predictions.mean(axis=1)
        errors = predictions.std(axis=1)
        return estimates, errors
    
    def evaluate(self, test, train, k):
        predictions = self.predict(test.user_ids, test.item_ids)
        self.rmse = rmse_score(predictions[0], test.ratings)
        self.rpi = -rpi_score(predictions, test.ratings)
        error = np.abs(test.ratings - predictions[0])
        self.correlation = pearsonr(error, predictions[1]), spearmanr(error, predictions[1])
        p, r, rri = precision_recall_rri_score(self, test, train, k)
        self.precision = p.mean(axis=0)
        self.recall = r.mean(axis=0)
        self.rri = -np.nanmean(rri, axis=0)
        self.quantiles, self.intervals = graphs_score(predictions, test.ratings)


class Resample(object):
    def __init__(self, base_model, n_models):
        base_model.verbose = False
        self.base_model = base_model
        self.models = []
        self.n_models = n_models

    def fit(self, train, test):
        for i in tqdm(range(self.n_models), desc='Resample'):
            train_, _ = split(train, random_state=np.random.RandomState(i), test_percentage=0.1)
            self.models.append(deepcopy(self.base_model))
            self.models[i]._initialize(train_)
            self.models[i].fit(train_)

    def predict(self, user_ids, item_ids=None):
        self.base_model._check_input(user_ids, item_ids, allow_items_none=True)
        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self.base_model._num_items, self.base_model._use_cuda)
        if np.isscalar(user_ids):
            user_ids = [user_ids]
        predictions = np.empty((len(user_ids), len(self.models)))
        for idx, model in enumerate(self.models):
            model._net.train(False)
            predictions[:, idx] = model._net(user_ids, item_ids).detach().cpu().numpy()
        estimates = self.base_model._net(user_ids, item_ids).detach().cpu().numpy()
        errors = predictions.std(axis=1)
        return estimates, errors
    
    def evaluate(self, test, train, k):
        predictions = self.predict(test.user_ids, test.item_ids)
        self.rmse = rmse_score(predictions[0], test.ratings)
        self.rpi = -rpi_score(predictions, test.ratings)
        error = np.abs(test.ratings - predictions[0])
        self.correlation = pearsonr(error, predictions[1]), spearmanr(error, predictions[1])
        self.rri = -np.nanmean(rri_score(self, test, train, k), axis=0)
        self.quantiles, self.intervals = graphs_score(predictions, test.ratings)


class BiasNet(torch.nn.Module):

    def __init__(self, num_users, num_items, sparse=False):

        super(BiasNet, self).__init__()

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return user_bias + item_bias


class ModelWrapper(object):
    def __init__(self, rating_estimator, error_estimator):
        self.R = rating_estimator
        self.E = error_estimator

    def predict(self, user_ids, item_ids=None):
        estimates = self.R.predict(user_ids, item_ids)
        errors = np.maximum(self.E.predict(user_ids, item_ids), 0)
        return estimates, errors
    
    def evaluate(self, test, train, k):
        predictions = self.predict(test.user_ids, test.item_ids)
        error = np.abs(test.ratings - predictions[0])
        self.correlation = pearsonr(error, predictions[1]), spearmanr(error, predictions[1])
        self.rpi = -rpi_score(predictions, test.ratings)
        self.rri = -np.nanmean(rri_score(self, test, train, k), axis=0)
        self.quantiles, self.intervals = graphs_score(predictions, test.ratings)


class OrdRecNet(torch.nn.Module):

    def __init__(self, num_users, num_items, num_labels, embedding_dim=32):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
        self.user_embeddings.weight.data.uniform_(-.01, .01)
        self.item_embeddings = torch.nn.Embedding(num_items, embedding_dim)
        self.item_embeddings.weight.data.uniform_(-.01, .01)

        self.item_biases = torch.nn.Embedding(num_items, 1)
        self.item_biases.weight.data.zero_()

        self.user_betas = torch.nn.Embedding(num_users, num_labels-1)
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

        user_distribution[:, 1:] -= user_distribution[:, :-1].clone()

        return user_distribution


class OrdRec(object):

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

        self._net = gpu(OrdRecNet(self._num_users,
                                  self._num_items,
                                  len(self._rating_labels),
                                  self._embedding_dim),
                        self._use_cuda)

        self._optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self._learning_rate,
            weight_decay=self._l2
            )

        '''
        self._optimizer = torch.optim.Adam(
            [{'params': self._net.user_embeddings.parameters(), 'weight_decay': self._l2},
             {'params': self._net.item_embeddings.parameters(), 'weight_decay': self._l2},
             {'params': self._net.item_biases.parameters(), 'weight_decay': self._l2},
             {'params': self._net.user_betas.parameters(), 'weight_decay': 0.1}],
            lr=self._learning_rate
            )
        '''

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
            test_ratings = gpu(torch.from_numpy(pd.factorize(test.ratings, sort=True)[0]), self._use_cuda)

        user_ids_tensor = gpu(torch.from_numpy(interactions.user_ids.astype(np.int64)), self._use_cuda)
        item_ids_tensor = gpu(torch.from_numpy(interactions.item_ids.astype(np.int64)), self._use_cuda)
        ratings_tensor = gpu(torch.from_numpy(pd.factorize(interactions.ratings, sort=True)[0]), self._use_cuda)

        for epoch_num in tqdm(range(self._n_iter), desc='OrdRec'):

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

        return mean.cpu().detach().numpy(), var.cpu().detach().numpy()

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