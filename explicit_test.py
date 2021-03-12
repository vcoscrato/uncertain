import torch
import numpy as np
from pandas import factorize

from uncertain.models import Linear, FunkSVD, CPMF, OrdRec, Ensemble, Resample
from uncertain.models import LinearUncertainty, CVUncertainty, PlugIn
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split
from uncertain.metrics import recommendation_score, pairwise

ML = get_movielens_dataset(variant='100K').cuda()
train, test = random_train_test_split(ML, test_percentage=0.2, random_state=0)
test, val = random_train_test_split(test, test_percentage=0.5, random_state=0)

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'Legacy/Empirical study/baseline.pth', 'use_cuda': True}
baseline = FunkSVD(**MF_params)
baseline.initialize(train)
baseline.fit(train, val)
#print(baseline.evaluate_zhu(test, relevance_threshold=4, max_k=5))

CPMF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 4,
               'batch_size': 512, 'path': 'Legacy/Empirical study/cpmf.pth', 'use_cuda': True}
cpmf = CPMF(**CPMF_params)
cpmf.initialize(train)
cpmf.fit(train, val)
#print(cpmf.evaluate_zhu(test, relevance_threshold=4, max_k=5))

ensemble = Ensemble(base_model=baseline, n_models=5)
ensemble.fit(train, val)
#print(ensemble.evaluate_zhu(test, relevance_threshold=4, max_k=5))

resample = Resample(base_model=baseline, n_models=5)
resample.fit(train, val)
#print(resample.evaluate_zhu(test, relevance_threshold=4, max_k=5))

user_support = PlugIn(baseline, LinearUncertainty(None, -train.get_user_support()))
#print(user_support.evaluate_zhu(test, relevance_threshold=4, max_k=5))

item_support = PlugIn(baseline, LinearUncertainty(None, -train.get_item_support()))
#print(item_support.evaluate_zhu(test, relevance_threshold=4, max_k=5))

item_variance = PlugIn(baseline, LinearUncertainty(None, train.get_item_variance()))
#print(item_variance.evaluate_zhu(test, relevance_threshold=4, max_k=5))

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'Legacy/Empirical study/funksvdcv.pth', 'use_cuda': True}
uncertainty = FunkSVD(**MF_params)
funksvdcv = CVUncertainty(recommender=baseline, uncertainty=uncertainty)
funksvdcv.fit(train, val)
funksvdcv = PlugIn(baseline, funksvdcv.uncertainty)
#print(funksvdcv.evaluate_zhu(test, relevance_threshold=4, max_k=5))

MF_params = {'embedding_dim': 0, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'Legacy/Empirical study/biascv.pth', 'use_cuda': True}
uncertainty = Linear(**MF_params)
biascv = CVUncertainty(recommender=baseline, params=uncertainty)
biascv.fit(train, val)
biascv = PlugIn(baseline, biascv.uncertainty)
#print(biascv.evaluate_zhu(test, relevance_threshold=4, max_k=5))

OrdRec_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 10,
                 'batch_size': 512, 'path': 'Legacy/Empirical study/ordrec.pth', 'use_cuda': True}
factor = factorize(train.ratings.cpu(), sort=True)
rating_labels = torch.from_numpy(factor[1].astype(np.float64)).cuda()
train.ratings = torch.from_numpy(factor[0]).cuda()
val.ratings = torch.from_numpy(factorize(val.ratings.cpu(), sort=True)[0]).cuda()
ordrec = OrdRec(rating_labels, **OrdRec_params)
ordrec.initialize(train)
ordrec.fit(train, val)
#print(ordrec.evaluate_zhu(test, relevance_threshold=4, max_k=5))