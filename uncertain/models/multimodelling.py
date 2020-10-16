from uncertain.models.BaseRecommender import _predict_process_ids
from uncertain.cross_validation import random_train_test_split as split
from torch import empty
from copy import deepcopy


class EnsembleRecommender(object):

    def __init__(self,
                 base_model,
                 n_models):

        self.models = [base_model]
        self.n_models = n_models

    def fit(self, interactions):

        for _ in tqdm(range(self.n_models-1), desc='Ensemble'):
            self.models.append(deepcopy(self.models[0]))
            self.models[-1]._initialize(interactions)
            self.models[-1].fit(interactions)

    def predict(self, user_ids, item_ids=None):

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self.models[0]._num_items,
                                                  self.models[0]._use_cuda)

        predictions = empty((len(user_ids), len(self.models)))
        for idx, model in enumerate(self.models):
            model._net.train(False)
            predictions[:, idx] = model._net(user_ids, item_ids)
        estimates = predictions.mean(1)
        errors = predictions.std(1)

        return estimates.cpu().detach().numpy(), errors.cpu().detach().numpy()
    
    
class ResampleRecommender(object):
    
    def __init__(self,
                 base_model,
                 n_models):

        self.base_model = base_model
        self.n_models = n_models

    def fit(self, interactions):
        for i in tqdm(range(self.n_models), desc='Resample'):
            train, _ = split(interactions, random_state=i, test_percentage=0.1)
            self.models.append(deepcopy(self.base_model))
            self.models[i]._initialize(train)
            self.models[i].fit(train)

    def predict(self, user_ids, item_ids=None):

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self.models[0]._num_items,
                                                  self.models[0]._use_cuda)

        predictions = empty((len(user_ids), len(self.models)))
        for idx, model in enumerate(self.models):
            model._net.train(False)
            predictions[:, idx] = model._net(user_ids, item_ids)
        estimates = self.base_model._net(user_ids, item_ids)
        errors = predictions.std(1)

        return estimates.cpu().detach().numpy(), errors.cpu().detach().numpy()