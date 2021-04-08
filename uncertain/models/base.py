import torch
import numpy as np
from uncertain.utils import gpu, minibatch
from uncertain.data_structures import Recommendations


class Recommender(object):

    def __init__(self, num_users=None, num_items=None, num_ratings=None, use_cuda=False):

        self.num_users = num_users
        self.num_items = num_items
        self.num_ratings = num_ratings
        self._use_cuda = use_cuda

    def predict_interactions(self, interactions):

        return self.predict(interactions.users(), interactions.items())

    def predict_user(self, user_id):

        item_ids = gpu(torch.arange(self.num_items), self._use_cuda)
        user_ids = gpu(torch.full_like(item_ids, user_id))

        return self.predict(user_ids, item_ids)

    def recommend(self, user_id, train=None, top=10):

        predictions = self.predict_user(user_id=user_id)

        if not self.is_uncertain:
            predictions = predictions
            uncertainties = None
        else:
            uncertainties = predictions[1]
            predictions = predictions[0]

        if train is not None:
            rated = train.get_rated_items(user_id)
            predictions[rated] = -float('inf')
            ranking = predictions.argsort(descending=True)[:-len(rated)][:top]
        else:
            ranking = predictions.argsort(descending=True)[:top]

        if self.is_uncertain:
            uncertainties = uncertainties[ranking]

        return Recommendations(user_id, ranking, uncertainties)
