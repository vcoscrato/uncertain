import torch
import numpy as np
from uncertain.data_structures import Recommendations


class Recommender(object):
    """
    Base class for recommendation systems.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def predict_user(self, user_id):

        item_ids = torch.arange(self.num_items, device=self.device)
        user_ids = torch.full_like(item_ids, user_id)

        return self.predict(user_ids, item_ids)

    def recommend(self, user, remove_items=None, top=10):

        predictions = self.predict_user(user_id=user)

        if not self.is_uncertain:
            predictions = predictions
        else:
            uncertainties = predictions[1]
            predictions = predictions[0]

        if remove_items is not None:
            predictions[remove_items] = -float('inf')
            ranking = predictions.argsort(descending=True)[:-len(remove_items)][:top]
        else:
            ranking = predictions.argsort(descending=True)[:top]

        kwargs = {'user': user, 'items': ranking}
        if self.is_uncertain:
            kwargs['uncertainties'] = uncertainties[ranking]

        if hasattr(self, 'user_labels'):
            kwargs['user_label'] = self.user_labels[user_id]
        if hasattr(self, 'item_labels'):
            kwargs['item_labels'] = self.item_labels[ranking.cpu()]

        return Recommendations(**kwargs)

    def sample_items(self, shape):

        return torch.randint(0, self.num_items, (shape,), device=self.device)


class UncertainWrapper(Recommender):
    """
    Wraps a score estimator with an uncertainty estimator.

    Parameters
    ----------
    scores: :class:`uncertain.models.base`
        A recommendation model.
    uncertainty: :class:`uncertain.models.base`
        An uncertainty estimator: A class containing a predict
        function that returns an uncertainty estimate for the
        given user, item pairs.
    """

    def __init__(self, interactions, scores, uncertainty):

        self.scores = scores
        self.uncertainty = uncertainty

        super().__init__(**interactions.pass_args())

    @property
    def is_uncertain(self):
        return True

    def predict(self, interactions=None, user_id=None):

        scores = self.scores.predict(interactions, user_id)
        uncertainty = self.uncertainty.predict(interactions, user_id)

        return scores, uncertainty


class ItemPopularity(Recommender):

    def __init__(self, interactions):

        self.item_popularity = interactions.get_item_popularity()
        super().__init__(**interactions.pass_args())

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
        super().__init__(**interactions.pass_args())

    @property
    def is_uncertain(self):
        return False

    def predict(self, user_ids, item_ids):

        return self.user_profile[user_ids]


class ItemVariance(Recommender):

    def __init__(self, interactions):

        self.item_variance = interactions.get_item_variance()
        super().__init__(**interactions.pass_args())

    @property
    def is_uncertain(self):
        return False

    def predict(self, user_ids, item_ids):

        return self.item_variance[item_ids]