import numpy as np
import torch
import pandas as pd
from copy import deepcopy
from scipy.stats import spearmanr, pearsonr

from spotlight.cross_validation import random_train_test_split as split
from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids
from spotlight.torch_utils import gpu

from .metrics import rmse_score, rpi_score, graphs_score, precision_recall_rri_score, _get_precision_recall_rri


class EnsembleRecommender(object):
    def __init__(self, base_model, n_models):
        base_model.verbose = False
        self.models = [base_model]
        self.n_models = n_models
        self.rmse = [base_model.rmse]
        self.rpi = [0]

    def fit(self, train, test):
        for i in range(1, self.n_models):
            self.models.append(deepcopy(self.models[0]))
            self.models[i]._initialize(train)
            self.models[i].fit(train)
            preds = self.predict(test.user_ids, test.item_ids)
            self.rmse.append(rmse_score(preds[0], test.ratings))
            self.rpi.append(rpi_score(preds, test.ratings))

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


class ResampleRecommender(object):
    def __init__(self, base_model, n_models):
        base_model.verbose = False
        self.base_model = base_model
        self.models = []
        self.n_models = n_models
        self.rpi = [0]

    def fit(self, train, test):
        for i in range(self.n_models):
            train_, _ = split(train, random_state=np.random.RandomState(i), test_percentage=0.1)
            self.models.append(deepcopy(self.base_model))
            self.models[i]._initialize(train_)
            self.models[i].fit(train_)
            if len(self.models) > 1:
                self.rpi.append(rpi_score(self.predict(test.user_ids, test.item_ids), test.ratings))
        return self

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


class ModelWrapper(object):
    def __init__(self, rating_estimator, error_estimator):
        self.R = rating_estimator
        self.rmse = rating_estimator.rmse
        self.E = error_estimator
        self.ermse = error_estimator.rmse

    def predict(self, user_ids, item_ids=None):
        estimates = self.R.predict(user_ids, item_ids)
        errors = np.maximum(self.E.predict(user_ids, item_ids), 0)
        return estimates, errors


class KorenSillNet(torch.nn.Module):

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

        user_distribution[:, 1:] -= user_distribution[:, :-1]

        return user_distribution


class KorenSill(object):

    def __init__(self,
                 embedding_dim,
                 n_iter,
                 batch_size,
                 learning_rate,
                 weight_decay,
                 use_cuda):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._weight_decay = weight_decay
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

        self._optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay
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
            test_ratings = gpu(torch.from_numpy(pd.factorize(test.ratings, sort=True)[0]), self._use_cuda)

        user_ids_tensor = gpu(torch.from_numpy(interactions.user_ids.astype(np.int64)), self._use_cuda)
        item_ids_tensor = gpu(torch.from_numpy(interactions.item_ids.astype(np.int64)), self._use_cuda)
        ratings_tensor = gpu(torch.from_numpy(pd.factorize(interactions.ratings, sort=True)[0]), self._use_cuda)

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
        confidence = var.max() - var

        return mean.cpu().detach().numpy(), confidence.cpu().detach().numpy()

    def evaluate(self, test, train):

        preds = self.predict(test.user_ids, test.item_ids)

        self.rmse = rmse_score(preds[0], test.ratings)
        self.rpi = rpi_score(preds, test.ratings)
        self.quantiles, self.intervals = graphs_score(preds, test.ratings)

        p, r, e = precision_recall_rri_score(self, test, train, np.arange(1, 11))
        self.precision = p.mean(axis=0)
        self.recall = r.mean(axis=0)
        self.rri = np.nanmean(e, axis=0)


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

        for epoch_num in range(self._n_iter):

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

        return out[0].cpu().detach().numpy().flatten(), 1 / np.sqrt(out[1].cpu().detach().numpy().flatten())

    def evaluate(self, test, train):

        preds = self.predict(test.user_ids, test.item_ids)
        self.rmse = rmse_score(preds[0], test.ratings)
        self.rpi = rpi_score(preds, test.ratings)
        self.quantiles, self.intervals = graphs_score(preds, test.ratings)

        df = pd.DataFrame(data={'user': test.user_ids, 'item': test.item_ids, 'error': np.square(preds[0] - test.ratings)})
        user_rmse = df.groupby('user').mean()['error']
        user_var = self._net.user_gammas.weight.data.cpu().detach()[torch.LongTensor(user_rmse.index)].numpy().flatten()
        user_rmse = np.array(user_rmse)
        self.user_rmse_var_corr = (pearsonr(user_rmse, user_var)[0], spearmanr(user_rmse, user_var)[0])

        item_rmse = df.groupby('item').mean()['error']
        item_var = self._net.item_gammas.weight.data.cpu().detach()[torch.LongTensor(item_rmse.index)].numpy().flatten()
        item_rmse = np.array(item_rmse)
        self.item_rmse_var_corr = (pearsonr(item_rmse, item_var)[0], spearmanr(item_rmse, item_var)[0])

        idx = test.ratings >= 4
        test_ = deepcopy(test)
        test_.user_ids = test_.user_ids[idx]
        test_.item_ids = test_.item_ids[idx]
        test_.ratings = test_.ratings[idx]
        #preds = self.predict(test_.user_ids, test_.item_ids)
        test_ = test_.tocsr()
        train_ = train.tocsr()

        avg_reliability, std_reliability = preds[1].mean(), preds[1].std()
        print(avg_reliability, std_reliability)
        k = np.arange(1, 11)

        uid = []
        precision = []
        recall = []
        rri = []

        for user_id, row in enumerate(test_):

            if not len(row.indices):
                continue

            predictions, reliabilities = self.predict(user_id)
            predictions *= -1

            rated = train_[user_id].indices
            predictions[rated] = np.infty

            predictions = predictions.argsort()

            targets = row.indices

            user_precision, user_recall, user_rri = zip(*[
                _get_precision_recall_rri(predictions, reliabilities, avg_reliability, std_reliability, targets, x)
                for x in k
            ])

            uid.append(user_id)
            precision.append(user_precision)
            recall.append(user_recall)
            rri.append(user_rri)

        precision = np.array(precision).squeeze()
        recall = np.array(recall).squeeze()
        self.rri = np.nanmean(np.array(rri).squeeze(), 0)

        map = precision[:, 9].flatten()
        user_var = self._net.user_gammas.weight.data.cpu().detach()[torch.LongTensor(uid)].numpy().flatten()
        self.user_map_var_corr = (pearsonr(map, user_var)[0], spearmanr(map, user_var)[0])

        self.precision = precision.mean(0)
        self.recall = recall.mean(0)