from uncertain.cross_validation import random_train_test_split as split
from uncertain.utils import gpu, minibatch
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from uncertain.metrics import rmse_score, recommendation_score, correlation, rpi_score, classification


class EnsembleRecommender(object):

    def __init__(self,
                 base_model,
                 n_models):

        self.models = [base_model]
        self.models[0]._verbose = False
        self.n_models = n_models

    @property
    def is_uncertain(self):
        return True

    def fit(self, train, validation):

        for _ in tqdm(range(self.n_models-1), desc='Ensemble'):
            self.models.append(deepcopy(self.models[0]))
            self.models[-1]._path += '_temp'
            self.models[-1]._verbose = False
            self.models[-1].initialize(train)
            self.models[-1].fit(train, validation)
            
    def _predict_process_ids(self, user_ids, item_ids):

        if item_ids is None:
            item_ids = torch.arange(self.models[0]._num_items)

        if np.isscalar(user_ids):
            user_ids = torch.tensor(user_ids)

        if item_ids.size() != user_ids.size():
            user_ids = user_ids.expand(item_ids.size())

        user_var = gpu(user_ids, self.models[0]._use_cuda)
        item_var = gpu(item_ids, self.models[0]._use_cuda)

        return user_var.squeeze(), item_var.squeeze()

    def predict(self, user_ids, item_ids=None):

        user_ids, item_ids = self._predict_process_ids(user_ids, item_ids)

        predictions = torch.empty((len(user_ids), len(self.models)), device=user_ids.device)
        for idx, model in enumerate(self.models):
            model._net.train(False)
            predictions[:, idx] = model._net(user_ids, item_ids).detach()

        estimates = predictions.mean(1)
        errors = predictions.std(1)

        return estimates, errors

    def recommend(self, user_id, train=None, top=10):

        predictions = self.predict(user_id)

        if not self.is_uncertain:
            predictions = -predictions
            uncertainties = None
        else:
            uncertainties = predictions[1]
            predictions = -predictions[0]

        if train is not None:
            rated = train.item_ids[train.user_ids == user_id]
            predictions[rated] = float('inf')

        idx = predictions.argsort()
        predictions = idx[:top]
        if self.is_uncertain:
            uncertainties = uncertainties[idx][:top]

        return predictions, uncertainties

    def evaluate(self, test, train):

        out = {}
        loader = minibatch(test, batch_size=int(1e5))
        est = []
        if self.is_uncertain:
            unc = []
            for u, i, _ in loader:
                predictions = self.predict(u, i)
                est.append(predictions[0])
                unc.append(predictions[1])
            unc = torch.hstack(unc)
        else:
            for u, i, _ in loader:
                est.append(self.predict(u, i))
        est = torch.hstack(est)

        p, r, a, s = recommendation_score(self, test, train, max_k=10)

        out['RMSE'] = rmse_score(est, test.ratings)
        out['Precision'] = p.mean(axis=0)
        out['Recall'] = r.mean(axis=0)

        if self.is_uncertain:
            error = torch.abs(test.ratings - est)
            idx = torch.randperm(len(unc))[:int(1e5)]
            quantiles = torch.quantile(unc[idx], torch.linspace(0, 1, 21, device=unc.device, dtype=unc.dtype))
            out['Quantile RMSE'] = torch.zeros(20)
            for idx in range(20):
                ind = torch.bitwise_and(quantiles[idx] <= unc, unc < quantiles[idx + 1])
                out['Quantile RMSE'][idx] = torch.sqrt(torch.square(error[ind]).mean())
            quantiles = torch.quantile(a, torch.linspace(0, 1, 21, device=a.device, dtype=a.dtype))
            out['Quantile MAP'] = torch.zeros(20)
            for idx in range(20):
                ind = torch.bitwise_and(quantiles[idx] <= a, a < quantiles[idx + 1])
                out['Quantile MAP'][idx] = p[ind, -1].mean()
            out['RRI'] = s.nansum(0) / (~s.isnan()).float().sum(0)
            out['Correlation'] = correlation(error, unc)
            out['RPI'] = rpi_score(error, unc)
            out['Classification'] = classification(error, unc)

        return out
    
    
class ResampleRecommender(object):
    
    def __init__(self,
                 base_model,
                 n_models):

        self.base_model = base_model
        self.base_model._verbose = False
        self.models = []
        self.n_models = n_models

    @property
    def is_uncertain(self):
        return True

    def fit(self, train, validation):
        for i in tqdm(range(self.n_models), desc='Resample'):
            train_, _ = split(train, random_state=i, test_percentage=0.1)
            self.models.append(deepcopy(self.base_model))
            self.models[-1]._path += '_temp'
            self.models[-1]._verbose = False
            self.models[i].initialize(train_)
            self.models[i].fit(train_, validation)

    def _predict_process_ids(self, user_ids, item_ids):

        if item_ids is None:
            item_ids = torch.arange(self.base_model._num_items)

        if np.isscalar(user_ids):
            user_ids = torch.tensor(user_ids)

        if item_ids.size() != user_ids.size():
            user_ids = user_ids.expand(item_ids.size())

        user_var = gpu(user_ids, self.base_model._use_cuda)
        item_var = gpu(item_ids, self.base_model._use_cuda)

        return user_var.squeeze(), item_var.squeeze()

    def predict(self, user_ids, item_ids=None):

        user_ids, item_ids = self._predict_process_ids(user_ids, item_ids)

        predictions = torch.empty((len(user_ids), len(self.models)), device=user_ids.device)
        for idx, model in enumerate(self.models):
            model._net.train(False)
            predictions[:, idx] = model._net(user_ids, item_ids).detach()
        estimates = self.base_model._net(user_ids, item_ids).detach()
        errors = predictions.std(1)

        return estimates, errors

    def recommend(self, user_id, train=None, top=10):

        predictions = self.predict(user_id)

        if not self.is_uncertain:
            predictions = -predictions
            uncertainties = None
        else:
            uncertainties = predictions[1]
            predictions = -predictions[0]

        if train is not None:
            rated = train.item_ids[train.user_ids == user_id]
            predictions[rated] = float('inf')

        idx = predictions.argsort()
        predictions = idx[:top]
        if self.is_uncertain:
            uncertainties = uncertainties[idx][:top]

        return predictions, uncertainties

    def evaluate(self, test, train):

        out = {}
        loader = minibatch(test, batch_size=int(1e5))
        est = []
        if self.is_uncertain:
            unc = []
            for u, i, _ in loader:
                predictions = self.predict(u, i)
                est.append(predictions[0])
                unc.append(predictions[1])
            unc = torch.hstack(unc)
        else:
            for u, i, _ in loader:
                est.append(self.predict(u, i))
        est = torch.hstack(est)

        p, r, a, s = recommendation_score(self, test, train, max_k=10)

        out['RMSE'] = rmse_score(est, test.ratings)
        out['Precision'] = p.mean(axis=0)
        out['Recall'] = r.mean(axis=0)

        if self.is_uncertain:
            error = torch.abs(test.ratings - est)
            idx = torch.randperm(len(unc))[:int(1e5)]
            quantiles = torch.quantile(unc[idx], torch.linspace(0, 1, 21, device=unc.device, dtype=unc.dtype))
            out['Quantile RMSE'] = torch.zeros(20)
            for idx in range(20):
                ind = torch.bitwise_and(quantiles[idx] <= unc, unc < quantiles[idx + 1])
                out['Quantile RMSE'][idx] = torch.sqrt(torch.square(error[ind]).mean())
            quantiles = torch.quantile(a, torch.linspace(0, 1, 21, device=a.device, dtype=a.dtype))
            out['Quantile MAP'] = torch.zeros(20)
            for idx in range(20):
                ind = torch.bitwise_and(quantiles[idx] <= a, a < quantiles[idx + 1])
                out['Quantile MAP'][idx] = p[ind, -1].mean()
            out['RRI'] = s.nansum(0) / (~s.isnan()).float().sum(0)
            out['Correlation'] = correlation(error, unc)
            out['RPI'] = rpi_score(error, unc)
            out['Classification'] = classification(error, unc)

        return out
