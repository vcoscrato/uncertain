from uncertain.cross_validation import random_train_test_split as split
from uncertain.utils import gpu
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm


class EnsembleRecommender(object):

    def __init__(self,
                 base_model,
                 n_models):

        self.models = [base_model]
        self.models[0]._verbose = False
        self.n_models = n_models

    def fit(self, interactions):

        for _ in tqdm(range(self.n_models-1), desc='Ensemble'):
            self.models.append(deepcopy(self.models[0]))
            self.models[-1]._initialize(interactions)
            self.models[-1].fit(interactions)
            
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
    
    
class ResampleRecommender(object):
    
    def __init__(self,
                 base_model,
                 n_models):

        self.base_model = base_model
        self.base_model._verbose = False
        self.models = []
        self.n_models = n_models

    def fit(self, interactions):
        for i in tqdm(range(self.n_models), desc='Resample'):
            train, _ = split(interactions, random_state=i, test_percentage=0.1)
            self.models.append(deepcopy(self.base_model))
            self.models[i]._initialize(train)
            self.models[i].fit(train)

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