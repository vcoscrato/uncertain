import torch
import numpy as np
from pandas import factorize
from copy import deepcopy

from uncertain.models.explicit.ExplicitFactorizationModel import ExplicitFactorizationModel
from uncertain.models.explicit.CPMF import CPMF
from uncertain.models.explicit.OrdRec import OrdRec
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split

data = get_movielens_dataset(variant='100K')
train, test = random_train_test_split(data, test_percentage=0.2)
train = train.gpu()
test = test.gpu()

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'Empirical study/baseline.pth', 'use_cuda': True}
baseline = ExplicitFactorizationModel(**MF_params)
baseline.fit(train, test)
baseline.evaluate(test, train)

user_accuracy = torch.empty((train.num_users, 2))
fold1, fold2 = random_train_test_split(train, random_state=0, test_percentage=0.5)
model_cv = ExplicitFactorizationModel(**MF_params)
model_cv._path = 'Empirical study/cv_trash.pth'
model_cv.initialize(fold1)
model_cv.fit(fold1, test)
for user in range(train.num_users):
    targets = fold2.item_ids[fold2.user_ids == user]
    if len(targets) > 0:
        recommendations = model_cv.recommend(user, fold1)[0]
        user_accuracy[user, 0] = np.mean([item in targets for item in recommendations])
    else:
        user_accuracy[user, 0] = np.nan
model_cv = ExplicitFactorizationModel(**MF_params)
model_cv._path = 'Empirical study/cv_trash2.pth'
model_cv.initialize(fold2)
model_cv.fit(fold2, test)
for user in range(train.num_users):
    targets = fold1.item_ids[fold1.user_ids == user]
    if len(targets) > 0:
        recommendations = model_cv.recommend(user, fold2)[0]
        user_accuracy[user, 1] = np.mean([item in targets for item in recommendations])
    else:
        user_accuracy[user, 0] = np.nan

user_accuracy = np.nanmean(user_accuracy, 1)
real = np.empty(train.num_users)
for user in range(train.num_users):
    targets = test.item_ids[test.user_ids == user]
    if len(targets) > 0:
        recommendations = model_cv.recommend(user, train)[0]
        real[user] = np.nanmean([item in targets for item in recommendations])
    else:
        real[user] = np.nan

print(np.corrcoef(real[np.logical_not(np.isnan(real))], user_accuracy[np.logical_not(np.isnan(real))]))
