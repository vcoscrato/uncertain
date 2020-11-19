import torch
import numpy as np
from pandas import factorize, DataFrame
from copy import deepcopy
from matplotlib import pyplot as plt

from uncertain.models.ExplicitFactorizationModel import ExplicitFactorizationModel
from uncertain.models.UncertainWrapper import UncertainWrapper, LinearUncertaintyEstimator
from uncertain.models.multimodelling import EnsembleRecommender, ResampleRecommender
from uncertain.cross_validation import user_based_split, random_train_test_split
from uncertain.interactions import ExplicitInteractions
from uncertain.models.CPMF import CPMF
from uncertain.models.OrdRec import OrdRec
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.datasets.goodbooks import get_goodbooks_dataset


from utils import dataset_loader, evaluate


train = get_movielens_dataset('100K')
train, test = random_train_test_split(train, test_percentage=0.2, random_state=0)
train.gpu()
test.gpu()

MF_params = {'embedding_dim': 10, 'n_iter': 50, 'l2_penalty': 1e-5,
             'learning_rate': 0.001, 'batch_size': 2048, 'use_cuda': True}
CPMF_params = {'embedding_dim': 10, 'n_iter': 50, 'l2_base': 1e-5, 'l2_var': 4e-5,
               'learning_rate': 0.001, 'batch_size': 2048, 'use_cuda': True}
OrdRec_params = {'embedding_dim': 10, 'n_iter': 50, 'l2_base': 5e-7, 'l2_step': 3e-7,
                 'learning_rate': 0.001, 'batch_size': 2048, 'use_cuda': True}


baseline = ExplicitFactorizationModel(**MF_params)
baseline.fit(train, test)
print(evaluate(baseline, test, train, uncertainty=False))


cpmf = CPMF(**CPMF_params)
cpmf.fit(train, test)
print(evaluate(cpmf, test, train, uncertainty=True))


factor = factorize(train.ratings.cpu(), sort=True)
rating_labels = torch.from_numpy(factor[1].astype(np.float64)).cuda()
train.ratings = torch.from_numpy(factor[0]).cuda()
og_ratings = test.ratings
test.ratings = torch.from_numpy(factorize(test.ratings.cpu(), sort=True)[0]).cuda()
ordrec = OrdRec(rating_labels, **OrdRec_params)
ordrec.fit(train, test)
test.ratings = og_ratings
print(evaluate(ordrec, test, train, uncertainty=True))
