import torch
import numpy as np
from pandas import factorize

from uncertain.models import FunkSVD, CPMF, OrdRec, MultiModelling
from uncertain.models import LinearUncertainty, CVUncertainty, PlugIn
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split
from uncertain.metrics import recommendation_score, pairwise

ML = get_movielens_dataset(variant='1M').cuda()
ML.ratings = None
train, test = random_train_test_split(ML, test_percentage=0.2, random_state=0)
test, val = random_train_test_split(test, test_percentage=0.5, random_state=0)

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 10,
             'batch_size': 512, 'path': 'Legacy/Empirical study/implicit.pth', 'use_cuda': True}
baseline = FunkSVD(**MF_params)
baseline.fit(train, val)
p, r, a, s = recommendation_score(baseline, test, train, None, max_k=10)
print(p.mean(axis=0))

CPMF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 5,
               'batch_size': 512, 'path': 'Legacy/Empirical study/imp_cpmf.pth', 'use_cuda': True}
cpmf = CPMF(**CPMF_params)
cpmf.fit(train, val)
p, r, a, s = recommendation_score(cpmf, test, train, None, max_k=10)
print(p.mean(axis=0))
print(pairwise(cpmf, test))

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
from scipy.stats import norm
grid = np.arange(0, 1.5, 0.001)
dist = norm.pdf(grid, 0.7608, 0.3387)
f, ax = plt.subplots()
ax.plot(grid, dist, color='k')
ax.fill_between(grid, dist, where=np.logical_and(grid > 0.083, grid < 1.438), alpha=0.3, color='b')
ax.set_xlabel('Relevance')
ax.set_ylabel('Density')






















import torch
import numpy as np
from pandas import factorize

from uncertain import Interactions
from uncertain.models import FunkSVD, CPMF, OrdRec, MultiModelling
from uncertain.models import LinearUncertainty, CVUncertainty, PlugIn
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split
from uncertain.metrics import recommendation_score, pairwise

coo_row = []
coo_col = []
coo_val = []

with open('/home/vcoscrato/Documents/Data/lastfm.dat', "r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        splitted = line.split('\t')
        user = int(splitted[0])
        artist = int(splitted[1])
        plays = int(splitted[2].strip())
        coo_row.append(user)
        coo_col.append(artist)
        coo_val.append(plays)

users = np.unique(coo_row, return_inverse=True)[1]
items = np.unique(coo_col, return_inverse=True)[1]
plays = np.array(coo_val)

data = Interactions((users, items)).cuda()
train, test = random_train_test_split(data, test_percentage=0.2, random_state=0)
test, val = random_train_test_split(test, test_percentage=0.5, random_state=0)

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 20,
             'batch_size': 512, 'path': 'Legacy/Empirical study/implicit.pth', 'use_cuda': True}
baseline = FunkSVD(**MF_params)
baseline.fit(train, val)
p, r, a, s = recommendation_score(baseline, test, train, None, max_k=10)
print(p.mean(axis=0))
print(pairwise(baseline, test))

CPMF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 10,
               'batch_size': 512, 'path': 'Legacy/Empirical study/imp_cpmf.pth', 'use_cuda': True}
cpmf = CPMF(**CPMF_params)
cpmf.fit(train, val)
p, r, a, s = recommendation_score(cpmf, test, train, None, max_k=10)
print(p.mean(axis=0))
print(pairwise(cpmf, test))