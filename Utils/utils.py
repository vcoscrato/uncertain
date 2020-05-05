import numpy as np
from spotlight.datasets import amazon, goodbooks, movielens
from spotlight.cross_validation import random_train_test_split as split
from spotlight.evaluation import rmse_score
import torch.nn as nn
from spotlight.layers import ZeroEmbedding


def dataset_loader(name):
    if name == 'goodbooks':
        data = goodbooks.get_goodbooks_dataset()
    elif name == 'amazon':
        data = amazon.get_amazon_dataset()
    else:
        data = movielens.get_movielens_dataset(name)
    train, test = split(data, random_state=np.random.RandomState(0), test_percentage=0.1)
    return train, test


class BiasNet(nn.Module):

    def __init__(self, num_users, num_items, sparse=False):

        super(BiasNet, self).__init__()

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return user_bias + item_bias


class IterativeLearner(object):
    def __init__(self, model, max_iterations, tolerance=0, random_state=0):
        self.model = model
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.val_rmse = []
        self.best_val_rmse = np.infty
        self.random_state=np.random.RandomState(random_state)
        self.predict = model.predict

    def _initialize(self, train, val):
        self.iterations = 0
        self.model._initialize(train)
        self.val_rmse.append(rmse_score(self.model, val))
        return self

    def fit(self, train, val):
        self._initialize(train, val)
        improvement = np.infty
        while (improvement > self.tolerance) and (self.iterations < self.max_iterations):
            self.model.fit(train)
            self.iterations += 1
            self.val_rmse.append(rmse_score(self.model, val))
            improvement = self.val_rmse[-2] - self.val_rmse[-1]
        return self
