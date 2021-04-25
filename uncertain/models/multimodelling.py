import os
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from uncertain.models.base import Recommender
from uncertain.cross_validation import random_train_test_split


class Ensemble(Recommender):

    def __init__(self,
                 base_model,
                 n_models):

        self.n_models = n_models
        self.models = [deepcopy(base_model)]
        self.models[0]._verbose = False
        self.models[0]._path = os.getcwd() + 'tmp'

        super().__init__(base_model.user_labels, base_model.item_labels, base_model.device)

    @property
    def is_uncertain(self):
        return True

    def fit(self, train, validation):
        for _ in tqdm(range(self.n_models - 1), desc='Ensemble'):
            self.models.append(deepcopy(self.models[0]))
            self.models[-1].initialize(train)
            self.models[-1].fit(train, validation)

    def predict(self, user_ids, item_ids):

        predictions = torch.empty((len(user_ids), len(self.models)), device=user_ids.device)
        for idx, model in enumerate(self.models):
            model._net.train(False)
            predictions[:, idx] = model._net(user_ids, item_ids).detach()

        estimates = predictions.mean(1)
        errors = predictions.std(1)

        return estimates, errors


class Resample(Recommender):

    def __init__(self,
                 base_model,
                 n_models):

        self.n_models = n_models
        self.models = []
        self.base_model = deepcopy(base_model)
        self.base_model._verbose = False
        self.base_model._path = os.getcwd()+'tmp'

        super().__init__(base_model.user_labels, base_model.item_labels, base_model.device)

    @property
    def is_uncertain(self):
        return True

    def fit(self, train, validation):

        for i in tqdm(range(self.n_models), desc='Resample'):
            train_, _ = random_train_test_split(train, random_state=i, test_percentage=0.1)
            self.models.append(deepcopy(self.base_model))
            self.models[i].initialize(train_)
            self.models[i].fit(train_, validation)

    def predict(self, user_ids, item_ids):

        predictions = torch.empty((len(user_ids), len(self.models)), device=user_ids.device)
        for idx, model in enumerate(self.models):
            model._net.train(False)
            predictions[:, idx] = model._net(user_ids, item_ids).detach()

        estimates = predictions.mean(1)
        errors = predictions.std(1)

        return estimates, errors
