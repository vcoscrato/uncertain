import torch
import numpy as np
from pandas import factorize

from uncertain.models.explicit.ExplicitFactorizationModel import ExplicitFactorizationModel
from uncertain.models.explicit.CPMF import CPMF
from uncertain.models.explicit.OrdRec import OrdRec
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split

from utils import evaluate

data = get_movielens_dataset(variant='10M')
train, test = random_train_test_split(data, test_percentage=0.2)
train = train.gpu()
test = test.gpu()

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'Empirical study/1M.pth', 'use_cuda': True}
baseline = ExplicitFactorizationModel(**MF_params)
baseline.fit(train, test)
baseline.evaluation = evaluate(baseline, test, train, uncertainty=False)

CPMF_params = {'embedding_dim': 50, 'l2_base': 0, 'l2_var': 0, 'learning_rate': 5,
               'batch_size': 512, 'path': 'Empirical study/1M.pth', 'use_cuda': True}
cpmf = CPMF(**CPMF_params)
cpmf.fit(train, test)
cpmf.evaluation = evaluate(cpmf, test, train, uncertainty=True)

OrdRec_params = {'embedding_dim': 50, 'l2_base': 0, 'l2_step': 0, 'learning_rate': 10,
                 'batch_size': 512, 'path': 'Empirical study/1M.pth', 'use_cuda': True}
factor = factorize(train.ratings.cpu(), sort=True)
rating_labels = torch.from_numpy(factor[1].astype(np.float64)).cuda()
train.ratings = torch.from_numpy(factor[0]).cuda()
og_ratings = test.ratings
test.ratings = torch.from_numpy(factorize(test.ratings.cpu(), sort=True)[0]).cuda()
ordrec = OrdRec(rating_labels, **OrdRec_params)
ordrec.fit(train, test)
test.ratings = og_ratings
ordrec.evaluation = evaluate(ordrec, test, train, uncertainty=True)
