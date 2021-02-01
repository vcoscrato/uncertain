import torch
import numpy as np
from pandas import factorize

from uncertain.models import FunkSVD, CPMF, OrdRec, MultiModelling
from uncertain.models import LinearUncertainty, CVUncertainty, PlugIn
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split

ML = get_movielens_dataset(variant='1M').cuda()
train, test = random_train_test_split(ML, test_percentage=0.2, random_state=0)
test, val = random_train_test_split(test, test_percentage=0.5, random_state=0)

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'Empirical study/baseline.pth', 'use_cuda': True}
baseline = FunkSVD(**MF_params)
baseline.fit(train, val)
print(baseline.evaluate(test, train))

CPMF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 5,
               'batch_size': 512, 'path': 'Empirical study/cpmf.pth', 'use_cuda': True}
cpmf = CPMF(**CPMF_params)
cpmf.fit(train, val)
print(cpmf.evaluate(test, train))

OrdRec_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 10,
                 'batch_size': 512, 'path': 'Empirical study/ordrec.pth', 'use_cuda': True}
factor = factorize(train.ratings.cpu(), sort=True)
rating_labels = torch.from_numpy(factor[1].astype(np.float64)).cuda()
train.ratings = torch.from_numpy(factor[0]).cuda()
val.ratings = torch.from_numpy(factorize(val.ratings.cpu(), sort=True)[0]).cuda()
ordrec = OrdRec(rating_labels, **OrdRec_params)
ordrec.fit(train, val)
print(ordrec.evaluate(test, train))

ensemble = MultiModelling(base_model=baseline, n_models=5, type='Ensemble')
ensemble.fit(train, val)
print(ensemble.evaluate(test, train))

resample = MultiModelling(base_model=baseline, n_models=5, type='Resample')
resample.fit(train, val)
print(resample.evaluate(test, train))

item_support = PlugIn(baseline, LinearUncertainty(None, -train.get_item_support()))
print(item_support.evaluate(test, train))

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'Empirical study/funksvdcv.pth', 'use_cuda': True}
funksvdcv = CVUncertainty(recommender=baseline, params=MF_params)
funksvdcv.fit(train, val)
funksvdcv = PlugIn(baseline, funksvdcv.uncertainty)
print(funksvdcv.evaluate(test, train))

MF_params = {'embedding_dim': 0, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'Empirical study/biascv.pth', 'use_cuda': True}
biascv = CVUncertainty(recommender=baseline, params=MF_params)
biascv.fit(train, val)
biascv = PlugIn(baseline, biascv.uncertainty)
print(biascv.evaluate(test, train))