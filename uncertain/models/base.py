import torch
import numpy as np
from uncertain.utils import gpu, minibatch


class Recommender(object):

    def __init__(self, num_users=None, num_items=None, num_ratings=None, use_cuda=False):

        self.num_users = num_users
        self.num_items = num_items
        self.num_ratings = num_ratings
        self._use_cuda = use_cuda
        self.remove_uncertain = None

    def _predict_process_ids(self, interactions=None, user_id=None):

        if interactions is not None:
            if torch.is_tensor(interactions):
                return interactions[:, 0], interactions[:, 1]
            else:
                return interactions.users(), interactions.items()

        else:
            item_ids = gpu(torch.arange(self.num_items), self._use_cuda)
            user_ids = gpu(torch.full_like(item_ids, user_id))

        return user_ids, item_ids

    def recommend(self, user_id, train=None, top=10):

        predictions = self.predict(user_id=user_id)

        if not self.is_uncertain:
            predictions = predictions
            uncertainties = None
        else:
            uncertainties = predictions[1]
            predictions = predictions[0]

        if train is not None:
            rated = train.items()[train.users() == user_id]
            predictions[rated] = -float('inf')
            ranking = predictions.argsort(descending=True)[:-len(rated)]
        else:
            ranking = predictions.argsort(descending=True)

        if self.is_uncertain:
            uncertainties = uncertainties[ranking]

            if self.remove_uncertain is not None:

                threshold = uncertainties.quantile(1 - self.remove_uncertain)
                idx = uncertainties < threshold
                ranking, uncertainties = ranking[idx], uncertainties[idx]

            return ranking[:top], uncertainties[:top]

        return ranking[:top], None
