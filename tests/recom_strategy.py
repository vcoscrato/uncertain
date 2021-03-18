import torch
import numpy as np
from pandas import factorize

from uncertain.models import Linear, FunkSVD, CPMF, OrdRec, Ensemble, Resample
from uncertain.models import LinearUncertainty, CVUncertainty, PlugIn
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split
from uncertain.metrics import recommendation_score, rmse_score, correlation, rpi_score, classification, diversity
from uncertain.utils import minibatch

ML = get_movielens_dataset(variant='100K').cuda()
train, test = random_train_test_split(ML, test_percentage=0.2, random_state=0)
test, val = random_train_test_split(test, test_percentage=0.5, random_state=0)

CPMF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 4,
               'batch_size': 512, 'path': 'tests/test_models/cpmf.pth', 'use_cuda': True}
cpmf = CPMF(**CPMF_params)
cpmf.initialize(train)
cpmf.load()

precision, rri, div = torch.empty((4, 10)), torch.empty((4, 9)), []
grid = torch.tensor((0, 0.1, 0.3, 0.5)).cuda()
for i, remove_uncertain in enumerate(grid):
    cpmf.remove_uncertain = remove_uncertain
    precision[i], _, rri[i] = recommendation_score(cpmf, test, train, max_k=10)
    div.append(diversity(cpmf, test, train, 10))


hit_rate_certain, hit_rate_uncertain = [], []
exp_surprise_certain, exp_surprise_uncertain = [], []
for u in range(test.num_users):

    targets = test.items()[torch.logical_and(test.users() == u,
                                             test.ratings >= 4)]
    if not len(targets):
        continue

    rated = train.items()[train.users() == u]
    with torch.no_grad():
        rated_var = cpmf._net.item_embeddings(rated)

    recommendation = cpmf.recommend(u)
    idx = recommendation[1] <= recommendation[1].median()
    certain = recommendation[0][idx]
    uncertain = recommendation[0][torch.logical_not(idx)]

    hit_rate_certain.append(0)
    for recom in certain:
        if recom in targets:
            hit_rate_certain[-1] += 0.2
            with torch.no_grad():
                recom_var = cpmf._net.item_embeddings(recom)
            exp_surprise_certain.append((1. - torch.cosine_similarity(recom_var, rated_var, dim=-1)).min().item())
    hit_rate_uncertain.append(0)
    for recom in uncertain:
        if recom in targets:
            hit_rate_uncertain[-1] += 0.2
            with torch.no_grad():
                recom_var = cpmf._net.item_embeddings(recom)
            exp_surprise_uncertain.append((1. - torch.cosine_similarity(recom_var, rated_var, dim=-1)).min().item())






from matplotlib import pyplot as plt
f, ax = plt.subplots(1, 2)
ax[0].plot(range(1, 11), precision[0], label='No removal')
ax[0].plot(range(1, 11), precision[1], label='Remove 10%')
ax[0].plot(range(1, 11), precision[2], label='Remove 30%')
ax[0].plot(range(1, 11), precision[3], label='Remove 50%')
ax[0].legend()
ax[1].plot(range(2, 11), rri[0], label='No removal')
ax[1].plot(range(2, 11), rri[1], label='Remove 10%')
ax[1].plot(range(2, 11), rri[2], label='Remove 30%')
ax[1].plot(range(2, 11), rri[3], label='Remove 50%')
ax[1].legend()
f.tight_layout()