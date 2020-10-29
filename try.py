import torch
import numpy as np
from pandas import factorize, DataFrame
from copy import deepcopy
from matplotlib import pyplot as plt

from uncertain.models.ExplicitFactorizationModel import ExplicitFactorizationModel
from uncertain.models.UncertainWrapper import UncertainWrapper, LinearUncertaintyEstimator
from uncertain.models.multimodelling import EnsembleRecommender, ResampleRecommender
from uncertain.cross_validation import random_train_test_split as split
from uncertain.interactions import Interactions
from uncertain.models.CPMF import CPMF
from uncertain.models.OrdRec import OrdRec

from utils import dataset_loader
from evaluation import evaluate

train, test = dataset_loader('goodbooks', 0)
train.gpu()
test.gpu()

MF_params = {'embedding_dim': 10, 'n_iter': 100, 'l2': 9e-6, 'learning_rate': 0.05,
             'batch_size': int(1e6), 'use_cuda': True}
CPMF_params = {'embedding_dim': 10, 'n_iter': 100, 'sigma': 5e4, 'learning_rate': 0.05,
               'batch_size': int(1e6), 'use_cuda': True}
OrdRec_params = {'embedding_dim': 10, 'n_iter': 50, 'l2': 6e-8, 'learning_rate': 0.05,
                 'batch_size': int(1e6), 'use_cuda': True}

model = CPMF(**CPMF_params)
model.fit(train, test)
print(evaluate(model, test, train, uncertainty=True))

'''
factor = factorize(train.ratings.cpu(), sort=True)
rating_labels = torch.from_numpy(factor[1].astype(np.float64)).cuda()
train.ratings = torch.from_numpy(factor[0]).cuda()
og = test.ratings
test.ratings = torch.from_numpy(factorize(test.ratings.cpu(), sort=True)[0]).cuda()
model = OrdRec(rating_labels, **OrdRec_params)
model.fit(train, test)
test.ratings = og
print(evaluate(model, test, train, uncertainty=True))
'''