import os
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from uncertain.utils import gpu
from uncertain.models.base import BaseRecommender
from uncertain.cross_validation import random_train_test_split


class MultiModelling(BaseRecommender):

    def __init__(self,
                 base_model,
                 n_models,
                 type):

        self.n_models = n_models
        if type == 'Ensemble':
            self.type = type
            self.models = [deepcopy(base_model)]
            self.models[0]._verbose = False
            self.models[0]._path += '_temp'
        elif type == 'Resample':
            self.type = type
            self.models = []
            self.base_model = deepcopy(base_model)
            self.base_model._verbose = False
            self.base_model._path += '_temp'
        else:
            raise Exception("Type has to be 'Ensemble' or ' Resample'.")
        
        super().__init__(base_model.num_users, base_model.num_items, base_model.num_ratings)

    @property
    def is_uncertain(self):
        return True

    def fit(self, train, validation):

        if self.type == 'Ensemble':
            for _ in tqdm(range(self.n_models-1), desc='Ensemble'):
                self.models.append(deepcopy(self.models[0]))
                self.models[-1].initialize(train)
                self.models[-1].fit(train, validation)
            os.remove(self.models[0]._path)

        else:
            for i in tqdm(range(self.n_models), desc='Resample'):
                train_, _ = random_train_test_split(train, random_state=i, test_percentage=0.1)
                self.models.append(deepcopy(self.base_model))
                self.models[i].initialize(train_)
                self.models[i].fit(train_, validation)
            os.remove(self.base_model._path)
            
    def _predict_process_ids(self, user_ids, item_ids):

        if item_ids is None:
            item_ids = torch.arange(self.models[0].num_items)

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
