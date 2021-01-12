import torch
import numpy as np
from pandas import factorize

from uncertain.models.explicit.ExplicitFactorizationModel import ExplicitFactorizationModel
from uncertain.models.explicit.CPMF import CPMF
from uncertain.models.explicit.OrdRec import OrdRec
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split, user_based_split
from uncertain.models import multimodelling

data = get_movielens_dataset(variant='100K')
train, val, test = user_based_split(interactions=data, n_validation=4, n_test=6)
train = train.gpu()
val = val.gpu()
test = test.gpu()

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'Empirical study/baseline.pth', 'use_cuda': True}
baseline = ExplicitFactorizationModel(**MF_params)
baseline.fit(train, val)
print(baseline.evaluate(test, train))

CPMF_params = {'embedding_dim': 50, 'l2_base': 0, 'l2_var': 0, 'learning_rate': 5,
               'batch_size': 512, 'path': 'Empirical study/cpmf.pth', 'use_cuda': True}
cpmf = CPMF(**CPMF_params)
cpmf.fit(train, val)
print(cpmf.evaluate(test, train))

OrdRec_params = {'embedding_dim': 50, 'l2_base': 0, 'l2_step': 0, 'learning_rate': 10,
                 'batch_size': 512, 'path': 'Empirical study/ordrec.pth', 'use_cuda': True}
factor = factorize(train.ratings.cpu(), sort=True)
rating_labels = torch.from_numpy(factor[1].astype(np.float64)).cuda()
train.ratings = torch.from_numpy(factor[0]).cuda()
val.ratings = torch.from_numpy(factorize(val.ratings.cpu(), sort=True)[0]).cuda()
ordrec = OrdRec(rating_labels, **OrdRec_params)
ordrec.fit(train, val)
print(ordrec.evaluate(test, train))

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'Empirical study/baseline.pth', 'use_cuda': True}
baseline = ExplicitFactorizationModel(**MF_params)
baseline.initialize(train)
baseline._net.load_state_dict(torch.load(baseline._path))

ensemble = multimodelling.EnsembleRecommender(base_model=baseline, n_models=5)
ensemble.fit(train, val)
print(ensemble.evaluate(test, train))

resample = multimodelling.ResampleRecommender(base_model=baseline, n_models=5)
resample.fit(train, val)
print(resample.evaluate(test, train))