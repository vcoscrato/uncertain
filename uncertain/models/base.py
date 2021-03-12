import torch
import numpy as np
from uncertain.utils import gpu, minibatch
from uncertain.metrics import rmse_score, recommendation_score, correlation, rpi_score, classification, determination


class Recommender(object):

    def __init__(self, num_users=None, num_items=None, num_ratings=None, use_cuda=False):

        self.num_users = num_users
        self.num_items = num_items
        self.num_ratings = num_ratings
        self._use_cuda = use_cuda

    def _predict_process_ids(self, interactions=None, user_ids=None):

        if interactions is not None:
            if torch.is_tensor(interactions):
                return interactions[:, 0], interactions[:, 1]
            else:
                return interactions.users(), interactions.items()

        else:
            item_ids = torch.arange(self.num_items)

        if np.isscalar(user_ids):
            user_ids = torch.tensor(user_ids)

        if item_ids.size() != user_ids.size():
            user_ids = user_ids.expand(item_ids.size())

        user_var = gpu(user_ids, self._use_cuda)
        item_var = gpu(item_ids, self._use_cuda)

        return user_var.squeeze(), item_var.squeeze()

    def recommend(self, user_id, train=None, top=10, remove_uncertain=0):

        predictions = self.predict(user_id)

        if not self.is_uncertain:
            predictions = predictions
            uncertainties = None
        else:
            uncertainties = predictions[1]
            predictions = predictions[0]

        if train is not None:
            rated = train.interactions[:, 1][train.interactions[:, 0] == user_id]
            predictions[rated] = float('inf')

        ranking = predictions.argsort(desc=True)

        if self.is_uncertain:
            uncertainties = uncertainties[ranking][:top]

            if remove_uncertain > 0 & remove_uncertain < 1:
                threshold = uncertainties.quantile(1 - remove_uncertain)
                idx = uncertainties < threshold
                ranking, uncertainties = ranking[uncertainties < idx], uncertainties[uncertainties < idx]

            elif remove_uncertain != 0:
                raise ValueError('remove_uncertainty should be within [0, 1)')

        return predictions[:top], uncertainties[:top]
