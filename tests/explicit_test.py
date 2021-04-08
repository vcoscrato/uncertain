import torch
import numpy as np
from pandas import factorize

from uncertain.models import Linear, FunkSVD, CPMF, OrdRec, Ensemble, Resample
from uncertain.models import LinearUncertainty, CVUncertainty, PlugIn
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split
from uncertain.metrics import recommendation_score, rmse_score, correlation, rpi_score, classification
from uncertain.utils import minibatch


def evaluate(model, test, train):
    out = {}
    loader = minibatch(test, batch_size=int(1e5))
    est = []
    if model.is_uncertain:
        unc = []
        for interactions, _ in loader:
            predictions = model.predict(interactions)
            est.append(predictions[0])
            unc.append(predictions[1])
        unc = torch.hstack(unc)
    else:
        for interactions, _ in loader:
            est.append(model.predict(interactions))
    est = torch.hstack(est)
    out['RMSE'] = rmse_score(est, test.ratings)

    out['Precision'], out['Recall'], out['RRI'] = recommendation_score(model, test, train, max_k=10)

    if model.is_uncertain:
        error = torch.abs(test.ratings - est)
        idx = torch.randperm(len(unc))[:int(1e5)]
        quantiles = torch.quantile(unc[idx], torch.linspace(0, 1, 21, device=unc.device, dtype=unc.dtype))
        out['Quantile RMSE'] = torch.zeros(20)
        for idx in range(20):
            ind = torch.bitwise_and(quantiles[idx] <= unc, unc < quantiles[idx + 1])
            out['Quantile RMSE'][idx] = torch.sqrt(torch.square(error[ind]).mean())
        out['Correlation'] = correlation(error, unc)
        out['RPI'] = rpi_score(error, unc)
        out['Classification'] = classification(error, unc)

    return out


ML = get_movielens_dataset(variant='100K').cuda()
train, test = random_train_test_split(ML, test_percentage=0.2, random_state=0)
test, val = random_train_test_split(test, test_percentage=0.5, random_state=0)

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'test_models/baseline.pth', 'use_cuda': True}
baseline = FunkSVD(**MF_params)
baseline.initialize(train)
baseline.load()
#baseline.fit(train, val)
baseline.evaluation = evaluate(baseline, test, train); print(baseline.evaluation)

CPMF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 4,
               'batch_size': 512, 'path': 'tests/test_models/cpmf.pth', 'use_cuda': True}
cpmf = CPMF(**CPMF_params)
cpmf.initialize(train)
cpmf.load()
cpmf.remove_uncertain = 0.5
cpmf.fit(train, val)
cpmf.evaluation = evaluate(cpmf, test, train); print(cpmf.evaluation)

'''
ensemble = Ensemble(base_model=baseline, n_models=5)
ensemble.fit(train, val)
ensemble.evaluation = evaluate(ensemble, test, train); print(ensemble.evaluation)

resample = Resample(base_model=baseline, n_models=5)
resample.fit(train, val)
resample.evaluation = evaluate(resample, test, train); print(resample.evaluation)

user_support = PlugIn(baseline, LinearUncertainty(-train.get_user_support(), torch.zeros(train.num_items).cuda()))
user_support.evaluation = evaluate(user_support, test, train); print(user_support.evaluation)

item_support = PlugIn(baseline, LinearUncertainty(torch.zeros(train.num_users).cuda(), -train.get_item_support()))
item_support.evaluation = evaluate(item_support, test, train); print(item_support.evaluation)

item_variance = PlugIn(baseline, LinearUncertainty(torch.zeros(train.num_users).cuda(), train.get_item_variance()))
item_variance.evaluation = evaluate(item_variance, test, train); print(item_variance.evaluation)

MF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 2,
             'batch_size': 512, 'path': 'test_models/funksvdcv.pth', 'use_cuda': True}
uncertainty = FunkSVD(**MF_params)
funksvdcv = CVUncertainty(recommender=baseline, uncertainty=uncertainty)
funksvdcv.fit(train, val)
funksvdcv = PlugIn(baseline, funksvdcv.uncertainty)
funksvdcv.evaluation = evaluate(funksvdcv, test, train); print(funksvdcv.evaluation)

MF_params = {'l2': 0, 'learning_rate': 2, 'batch_size': 512, 
             'path': 'test_models/biascv.pth', 'use_cuda': True}
uncertainty = Linear(**MF_params)
biascv = CVUncertainty(recommender=baseline, uncertainty=uncertainty)
biascv.fit(train, val)
biascv = PlugIn(baseline, biascv.uncertainty)
biascv.evaluation = evaluate(biascv, test, train); print(biascv.evaluation)

OrdRec_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 10,
                 'batch_size': 512, 'path': 'test_models/ordrec.pth', 'use_cuda': True}
factor = factorize(train.ratings.cpu(), sort=True)
rating_labels = torch.from_numpy(factor[1].astype(np.float64)).cuda()
train.ratings = torch.from_numpy(factor[0]).cuda()
val.ratings = torch.from_numpy(factorize(val.ratings.cpu(), sort=True)[0]).cuda()
ordrec = OrdRec(rating_labels, **OrdRec_params)
ordrec.initialize(train)
ordrec.fit(train, val)
ordrec.evaluation = evaluate(ordrec, test, train); print(ordrec.evaluation)
'''

from pandas import DataFrame as df
from matplotlib import pyplot as plt
from copy import deepcopy
from uncertain.data_structures import Recommendations
from uncertain.metrics import precision, hit_rate, surprise

unc = []
for u in range(test.num_users):
  users = torch.full([10], u).cuda()
  items = torch.randint(low=1, high=train.num_items, size=[10]).cuda()
  with torch.no_grad():
    unc.append(cpmf._net(users, items)[1])
unc = torch.hstack(unc).cpu().detach().numpy()
cuts = np.quantile(unc, [0.75, 0.5, 0.25])
unc = unc[unc < 5]
count, bins, _ = plt.hist(unc, 30, density=True)
count = [0] + list(count) + [0]
bins = list(bins) + [bins[-1] + (bins[-1] - bins[-2])]

cuts = {'Permissive': {'cut': cuts[0], 'color': 'g'},
        'Median': {'cut': cuts[1], 'color': 'b'},
        'Restrictive': {'cut': cuts[2], 'color': 'r'}}
f, ax = plt.subplots()
ax.plot(bins, count, color='k')
for key, value in cuts.items():
    ax.axvline(x=value['cut'], color=value['color'], linestyle='dashed', label=key)
ax.set_xlabel('Uncertainty')
ax.set_ylabel('Density')
ax.legend()
f.tight_layout()
#f.savefig('Movielens/distribution.pdf')
#f.show()

for quartile in cuts:
    cuts[quartile].update({'Drop': {'Coverage': 0, 'Hit rate': [], 'Surprise': []},
                           'Split certain': {'% Recommendations': 0, 'Hit rate': [], 'Surprise': []},
                           'Split uncertain': {'% Recommendations': 0, 'Hit rate': [], 'Surprise': []}})

baseline = {'cut': float('inf'), 'color': 'k', 'Hit rate': [], 'Surprise': []}

relevant_users = 0
for u in range(test.num_users):

    targets = test.items()[torch.logical_and(test.users() == u,
                                             test.ratings >= 4)]
    if not len(targets):
        continue
    else:
        relevant_users += 1

    rated = train.get_rated_items(u)
    with torch.no_grad():
        rated_var = cpmf._net.item_embeddings(rated)

    pred, unc = cpmf.predict_user(u)
    pred[rated] = -float('inf')
    ranking = pred.argsort(descending=True)[:-len(rated)]
    unc = unc[ranking]
    top10 = ranking[:10]
    unc_top10 = unc[top10]
    
    recommendations = Recommendations(u, top10, unc_top10)
    baseline['Hit rate'].append(hit_rate(recommendations, targets))
    baseline['Surprise'].append(surprise(recommendations, cpmf, rated_var))

    for quartile in cuts:

        idx = unc < cuts[quartile]['cut']
        ranking_ = ranking[idx]
        unc_ = unc[idx]
        if idx.sum() > 0:
            recommendations = Recommendations(u, ranking_[:10], unc_[:10])
            cuts[quartile]['Drop']['Hit rate'].append(hit_rate(recommendations, targets))
            cuts[quartile]['Drop']['Surprise'].append(surprise(recommendations, cpmf, rated_var))
        if idx.sum() >= 10:
            cuts[quartile]['Drop']['Coverage'] += 1

        if quartile != 'Free':
            is_certain = unc_top10 < cuts[quartile]['cut']
            if is_certain.sum() > 0:
                cuts[quartile]['Split certain']['% Recommendations'] += is_certain.sum()
                recommendations = Recommendations(u, top10[is_certain], unc_top10[is_certain])
                cuts[quartile]['Split certain']['Hit rate'].append(hit_rate(recommendations, targets))
                cuts[quartile]['Split certain']['Surprise'].append(surprise(recommendations, cpmf, rated_var))

            is_uncertain = torch.logical_not(is_certain)
            if is_uncertain.sum() > 0:
                cuts[quartile]['Split uncertain']['% Recommendations'] += is_uncertain.sum()
                recommendations = Recommendations(u, top10[is_uncertain], unc_top10[is_uncertain])
                cuts[quartile]['Split uncertain']['Hit rate'].append(hit_rate(recommendations, targets))
                cuts[quartile]['Split uncertain']['Surprise'].append(surprise(recommendations, cpmf, rated_var))

baseline['Hit rate'] = np.mean(baseline['Hit rate'])
baseline['Surprise'] = np.mean(np.hstack(baseline['Surprise']))
print(baseline)

dropping = df(([cuts[quartile]['Drop']['Coverage'] * 100 / relevant_users for quartile in cuts],
               [np.mean(cuts[quartile]['Drop']['Hit rate']) for quartile in cuts],
               [np.mean(np.hstack(cuts[quartile]['Drop']['Surprise'])) for quartile in cuts]),
              index=['Coverage', 'Hit rate', 'Surprise'], columns=cuts.keys()).T
print(dropping)

certain = df(([cuts[quartile]['Split certain']['% Recommendations'].item() * 10 / relevant_users for quartile in cuts],
              [np.mean(cuts[quartile]['Split certain']['Hit rate']) for quartile in cuts],
              [np.mean(np.hstack(cuts[quartile]['Split certain']['Surprise'])) for quartile in cuts]),
             index=['% Recommendations', 'Hit rate', 'Surprise'], columns=cuts.keys()).T
print(certain)

uncertain = df(([cuts[quartile]['Split uncertain']['% Recommendations'].item() * 10 / relevant_users for quartile in cuts],
                [np.mean(cuts[quartile]['Split uncertain']['Hit rate']) for quartile in cuts],
                [np.mean(np.hstack(cuts[quartile]['Split uncertain']['Surprise'])) for quartile in cuts]),
               index=['% Recommendations', 'Hit rate', 'Surprise'], columns=cuts.keys()).T
print(uncertain)
