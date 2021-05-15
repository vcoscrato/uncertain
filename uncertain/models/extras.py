import torch
from uncertain.models.base import Recommender


class Ensemble(Recommender):

    def __init__(self, interactions, models):
        super().__init__()
        self.pass_args(interactions)
        self.models = models

    @property
    def is_uncertain(self):
        return True

    def predict(self, user_id):
        predictions = torch.empty((self.num_items, len(self.models)))
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict(user_id)
        estimates = predictions.mean(1)
        errors = predictions.std(1)
        return estimates, errors


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
        given user.
    """
    def __init__(self, interactions, scores, uncertainty):
        super().__init__()
        self.pass_args(interactions)
        self.scores = scores
        self.uncertainty = uncertainty

    @property
    def is_uncertain(self):
        return True

    def predict(self, user_id):
        scores = self.scores.predict(user_id)
        uncertainty = self.uncertainty.predict(user_id)
        return scores, uncertainty


class ItemPopularity(Recommender):

    def __init__(self, interactions):
        super().__init__()
        self.pass_args(interactions)
        self.item_popularity = interactions.get_item_popularity()

    @property
    def is_uncertain(self):
        return False

    def predict(self, user_id):
        return self.item_popularity


class UserProfileLength(Recommender):

    def __init__(self, interactions):
        super().__init__()
        self.pass_args(interactions)
        self.user_profile = interactions.get_user_profile_length()

    @property
    def is_uncertain(self):
        return False

    def predict(self, user_id):
        return self.user_profile[torch.full(self.num_items, user_id)]


class ItemVariance(Recommender):

    def __init__(self, interactions):
        super().__init__()
        self.pass_args(interactions)
        self.item_variance = interactions.get_item_variance()

    @property
    def is_uncertain(self):
        return False

    def predict(self, user_id):
        return self.item_variance
