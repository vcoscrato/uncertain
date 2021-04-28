import torch
import numpy as np
from uncertain.data_structures import Recommendations


class Recommender(object):

    def __init__(self, user_labels=None, item_labels=None, device='cpu'):

        self.user_labels = user_labels
        self.item_labels = item_labels
        self.device = device

    @property
    def num_users(self):
        return len(self.user_labels)

    @property
    def num_items(self):
        return len(self.item_labels)

    def predict_interactions(self, interactions, batch_size=1e100):

        if batch_size > len(interactions):
            return self.predict(interactions.users, interactions.items)
        else:
            batch_size = int(batch_size)
            est = torch.empty(len(interactions), device=self.device)
            if self.is_uncertain:
                unc = torch.empty(len(interactions), device=self.device)

            loader = interactions.minibatch(batch_size)
            for minibatch_num, (users, items, _) in enumerate(loader):
                preds = self.predict(users, items)
                if self.is_uncertain:
                    est[(minibatch_num * batch_size):((minibatch_num + 1) * batch_size)] = preds[0]
                    unc[(minibatch_num * batch_size):((minibatch_num + 1) * batch_size)] = preds[1]
                else:
                    est[(minibatch_num * batch_size):((minibatch_num + 1) * batch_size)] = preds

            if self.is_uncertain:
                return est, unc
            else:
                return est

    def predict_user(self, user_id):

        item_ids = torch.arange(self.num_items, device=self.device)
        user_ids = torch.full_like(item_ids, user_id)

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

        return Recommendations(user_id, ranking, self.item_labels[ranking], uncertainties)

    def sample_items(self, shape):

        return torch.randint(0, self.num_items, (shape,), device=self.device)


class UncertainWrapper(Recommender):
    """
    Wraps a rating estimator with an uncertainty estimator.

    Parameters
    ----------
    ratings: :class:`uncertain.models.base`
        A recommendation model.
    uncertainty: :class:`uncertain.models.base`
        An uncertainty estimator: A class containing a predict
        function that returns an uncertainty estimate for the
        given user, item pairs.
    """

    def __init__(self,
                 ratings,
                 uncertainty):

        self.ratings = ratings
        self.uncertainty = uncertainty

        super().__init__(ratings.user_labels, ratings.item_labels, ratings.device)

    @property
    def is_uncertain(self):
        return True

    def predict(self, interactions=None, user_id=None):

        ratings = self.ratings.predict(interactions, user_id)
        uncertainty = self.uncertainty.predict(interactions, user_id)

        return ratings, uncertainty


class ItemPopularity(Recommender):

    def __init__(self, interactions):

        self.item_popularity = interactions.get_item_popularity()
        super().__init__(interactions.user_labels, interactions.item_labels, device=interactions.device)

    @property
    def is_uncertain(self):
        return False

    def predict(self, user_ids, item_ids):

        return self.item_popularity[item_ids]


class UserProfileLength(Recommender):
    """
    Linear uncertainty estimator that uses the
    sum of static user and/or item coefficients.


    Parameters
    ----------
    user_factor: tensor or array
        A tensor or array containing the user coefficients.
    item_factor: tensor or array
        A tensor or array containing the item coefficients.
    """

    def __init__(self, interactions):

        self.user_profile = interactions.get_user_profile_length()
        super().__init__(interactions.user_labels, interactions.item_labels, device=interactions.device)

    @property
    def is_uncertain(self):
        return False

    def predict(self, user_ids, item_ids):

        return self.user_profile[user_ids]


class ItemVariance(Recommender):

    def __init__(self, interactions):

        self.item_variance = interactions.get_item_variance()
        super().__init__(interactions.user_labels, interactions.item_labels, device=interactions.device)

    @property
    def is_uncertain(self):
        return False

    def predict(self, user_ids, item_ids):

        return self.item_variance[item_ids]