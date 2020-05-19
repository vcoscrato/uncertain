import numpy as np
from spotlight.datasets import amazon, goodbooks, movielens
from spotlight.cross_validation import random_train_test_split as split
from spotlight.evaluation import rmse_score
import torch.nn as nn
from spotlight.layers import ZeroEmbedding, ScaledEmbedding
import torch


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


class KorenSill(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32, beta_dim=5,
                 user_embedding_layer=None, item_embedding_layer=None, sparse=False):

        super(KorenSill, self).__init__()

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)
        self.user_ts = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.user_betas = ScaledEmbedding(num_users, beta_dim, sparse=sparse)

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        y = dot + user_bias + item_bias

        user_t = self.user_ts(user_ids)
        user_betas = self.user_betas(item_ids)

        user_ts = user_t + torch.exp(user_betas)
        user_distribution = 1 / (1 + torch.exp(y - user_t))

        return user_distribution




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


def KorenSillLoss(a):
    return 0