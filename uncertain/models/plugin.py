import os
import torch
from copy import deepcopy
from uncertain import Interactions
from uncertain.models.base import Recommender
from uncertain.cross_validation import random_train_test_split
from uncertain.models import FunkSVD
from uncertain.utils import minibatch


class LinearUncertainty(object):
    """
    Basic uncertainty estimator that uses the
    sum of static user and/or item coefficients.


    Parameters
    ----------
    user_uncertainty: tensor
        A tensor containing the uncertainty coefficient for each user.
    item_uncertainty: tensor
        A tensor containing the uncertainty coefficient for each user.
    """

    def __init__(self,
                 user_uncertainty,
                 item_uncertainty):

        self.user = user_uncertainty
        self.item = item_uncertainty

    def predict(self, user_ids, item_ids):

        user_uncertainty = self.user[user_ids] if self.user is not None else 0
        item_uncertainty = self.item[item_ids] if self.item is not None else 0

        return user_uncertainty + item_uncertainty


class CVUncertainty(object):

    def __init__(self, recommender, uncertainty):

        self.recommender = deepcopy(recommender)
        self.recommender._path += '_temp'
        self.recommender._verbose = False
        self.uncertainty = uncertainty

    def fit(self, train, val):

        fold1, fold2 = random_train_test_split(train, test_percentage=0.5, random_state=0)

        model_cv = deepcopy(self.recommender)
        model_cv.initialize(fold1)
        model_cv.fit(fold1, val)
        errors = torch.empty_like(fold2.ratings)
        loader = minibatch(fold2, batch_size=int(1e6))
        for minibatch_num, (interactions, ratings) in enumerate(loader):
            preds = model_cv.predict(interactions)
            errors[(minibatch_num * int(1e6)):((minibatch_num + 1) * int(1e6))] = torch.abs(ratings - preds)

        model_cv.initialize(fold2)
        model_cv.fit(fold2, val)
        errors_ = torch.empty_like(fold1.ratings)
        loader = minibatch(fold1, batch_size=int(1e6))
        for minibatch_num, (interactions, ratings) in enumerate(loader):
            preds = model_cv.predict(interactions)
            errors_[(minibatch_num * int(1e6)):((minibatch_num + 1) * int(1e6))] = torch.abs(ratings - preds)

        os.remove(self.recommender._path)
        train_errors = Interactions(torch.vstack((fold2.interactions, fold1.interactions)),
                                    torch.cat((errors, errors_)), num_users=train.num_users, num_items=train.num_items)

        train_set, val_set = random_train_test_split(train_errors, test_percentage=0.2, random_state=0)
        self.uncertainty.fit(train_set, val_set)


class PlugIn(Recommender):
    """
    Wraps a rating estimator with an uncertainty estimator.

    Parameters
    ----------
    ratings: :class:`uncertain.models.base`
        A rating estimator.
    uncertainty: :class:`uncertain.UncertaintyWrapper.LinearUncertaintyEstimator
        An uncertainty estimator: A class containing a predict
        function that returns an uncertainty estimate for the
        given user, item pairs.
    """

    def __init__(self,
                 ratings,
                 uncertainty):

        self.ratings = ratings
        self.uncertainty = uncertainty

        super().__init__(ratings.num_users, ratings.num_items, ratings.num_ratings, ratings._use_cuda)

    @property
    def is_uncertain(self):
        return True

    def predict(self, interactions=None, user_ids=None):

        ratings = self.ratings.predict(interactions, user_ids)
        uncertainty = self.uncertainty.predict(interactions, user_ids)

        return ratings, uncertainty
