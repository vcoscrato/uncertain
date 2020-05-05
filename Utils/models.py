import numpy as np
from copy import deepcopy
from spotlight.cross_validation import random_train_test_split as split
from .metrics import rpi, rmse_rpi_wrapper
from spotlight.factorization._components import _predict_process_ids


class EnsembleRecommender(object):
    def __init__(self, base_model, n_models):
        base_model.verbose = False
        self.models = [base_model]
        self.n_models = n_models
        self.rmse = [base_model.rmse]
        self.rpi = [0]

    def fit(self, train, val, test):
        for i in range(1, self.n_models):
            self.models.append(deepcopy(self.models[0]).fit(train, val))
            metrics = rmse_rpi_wrapper(self, test)
            self.rmse.append(metrics[0])
            self.rpi.append(metrics[1])
        return self

    def predict(self, user_ids, item_ids=None):
        self.models[0].model._check_input(user_ids, item_ids, allow_items_none=True)
        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self.models[0].model._num_items, self.models[0].model._use_cuda)
        if np.isscalar(user_ids):
            user_ids = [user_ids]
        predictions = np.empty((len(user_ids), len(self.models)))
        for idx, model in enumerate(self.models):
            model.model._net.train(False)
            predictions[:, idx] = model.model._net(user_ids, item_ids).detach().cpu().numpy()
        estimates = predictions.mean(axis=1)
        reliabilities = 1 / predictions.std(axis=1)
        return estimates, reliabilities


class ResampleRecommender(object):
    def __init__(self, base_model, n_models):
        base_model.verbose = False
        self.base_model = base_model
        self.models = []
        self.n_models = n_models
        self.rpi = [0]
        
    def fit(self, train, val, test):
        for i in range(self.n_models):
            train_, _ = split(train, random_state=np.random.RandomState(i), test_percentage=0.1)
            self.models.append(deepcopy(self.base_model.fit(train_, val)))
            if len(self.models) > 1:
                self.rpi.append(rpi(self, test))
        return self

    def predict(self, user_ids, item_ids=None):
        self.base_model.model._check_input(user_ids, item_ids, allow_items_none=True)
        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self.base_model.model._num_items, self.base_model.model._use_cuda)
        if np.isscalar(user_ids):
            user_ids = [user_ids]
        predictions = np.empty((len(user_ids), len(self.models)))
        for idx, model in enumerate(self.models):
            model.model._net.train(False)
            predictions[:, idx] = model.model._net(user_ids, item_ids).detach().cpu().numpy()
        estimates = self.base_model.model._net(user_ids, item_ids).detach().cpu().numpy()
        reliabilities = 1 / predictions.std(axis=1)
        return estimates, reliabilities


class ModelWrapper(object):
    def __init__(self, rating_estimator, error_estimator):
        self.R = rating_estimator
        self.E = error_estimator

    def predict(self, user_ids, item_ids=None):
        estimates = self.R.predict(user_ids, item_ids)
        reliabilities = 1 / self.E.predict(user_ids, item_ids)
        return estimates, reliabilities
