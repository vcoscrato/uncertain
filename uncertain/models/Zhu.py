import os
import torch
from copy import deepcopy
from uncertain import Interactions
from uncertain.models.base import Recommender
from uncertain.cross_validation import random_train_test_split
from uncertain.models import FunkSVD


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
        errors = model_cv.predict_interactions(fold2, batch_size=1e3)

        model_cv.initialize(fold2)
        model_cv.fit(fold2, val)
        errors_ = model_cv.predict_interactions(fold1, batch_size=1e3)

        os.remove(self.recommender._path)
        train_errors = Interactions(torch.vstack((fold2.interactions, fold1.interactions)),
                                    torch.cat((errors, errors_)),
                                    user_labels=train.user_labels, item_labels=train.item_labels)

        train_set, val_set = random_train_test_split(train_errors, test_percentage=0.2, random_state=0)
        self.uncertainty.fit(train_set, val_set)
