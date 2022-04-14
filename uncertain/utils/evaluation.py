import torch
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
from scipy.special import comb
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from uncertain.core import VanillaRecommender, UncertainRecommender


def get_AP(hits):
    n_hits = hits.cumsum(0)
    if n_hits[-1] > 0:
        precision = n_hits / np.arange(1, len(hits) + 1)
        return np.sum(precision * hits) / n_hits[-1]
    else:
        return 0


def rmse(errors):
    return np.sqrt(np.square(errors).mean())


def rpi_score(errors, uncertainties):
    mae = errors.mean()
    errors_deviations = errors - mae
    uncertainties_deviations = uncertainties - uncertainties.mean()
    errors_std = np.abs(errors_deviations).mean()
    epsilon_std = np.abs(uncertainties_deviations).mean()
    num = np.mean(errors * errors_deviations * uncertainties_deviations)
    denom = errors_std * epsilon_std * mae
    return num / denom


def classification(errors, uncertainties):
    splitter = KFold(n_splits=2, shuffle=True, random_state=0)
    targets = errors > 1
    auc = 0
    for train_index, test_index in splitter.split(errors):
        mod = LogisticRegression().fit(uncertainties[train_index].reshape(-1, 1), targets[train_index])
        prob = mod.predict_proba(uncertainties[test_index].reshape(-1, 1))
        auc += roc_auc_score(targets[test_index], prob[:, 1]) / 2
    return auc


def quantile_score(errors, uncertainties):
    quantiles = np.quantile(uncertainties, np.linspace(0, 1, 21))
    q_rmse = np.zeros(20)
    for idx in range(20):
        ind = np.bitwise_and(quantiles[idx] <= uncertainties, uncertainties < quantiles[idx + 1])
        q_rmse[idx] = np.sqrt(np.square(errors[ind]).mean())
    return q_rmse


def accuracy_metrics(size, max_k, is_uncertain):
    out = {'MAP': np.zeros((size, max_k)),
           'Recall': np.zeros((size, max_k))}
    if is_uncertain:
        out['URI_rec'] = np.full(size, np.nan)
    return out


def test(model, data, max_k, name):

    metrics = {}
    Rating_rec = accuracy_metrics(len(data.test_users), max_k, isinstance(model, UncertainRecommender))

    pred = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test[:, 1]).long())
    if not isinstance(model, UncertainRecommender):
        pred = pred[~np.isnan(pred)]
        metrics['RMSE'] = rmse(pred - data.test[:, 2])
    if isinstance(model, UncertainRecommender):
        idx = ~np.isnan(pred[1])
        unc = pred[1][idx]
        pred = pred[0][idx]
        errors = np.abs(data.test[:, 2] - pred)
        metrics['RMSE'] = np.sqrt(np.square(errors).mean())
        metrics['RPI'] = rpi_score(errors, unc)
        metrics['Classification'] = classification(errors, unc)
        metrics['Quantile RMSE'] = quantile_score(errors, unc)
        metrics['Pearson error x unc'] = stats.pearsonr(errors, unc)[0]
        metrics['Spearman error x unc'] = stats.spearmanr(errors, unc)[0]
        metrics['dw'] = metrics['Quantile RMSE'][-1] - metrics['Quantile RMSE'][0]
        URI_rat = {'test': {'avg': unc.mean(), 'std': unc.std()},
                   'rec': {'hits': 0, 'hits_unc': 0,
                           'avg': np.zeros(len(data.test_users))}}

        rand_preds = model.predict(data.rand['users'], data.rand['items'])
        user_vars = data.user.loc[data.rand['users']]
        item_vars = data.item.loc[data.rand['items']]
        metrics['User_unc_corr'] = {column: np.corrcoef(rand_preds[1], user_vars[column].to_numpy().flatten())[0, 1]
                                    for column in data.user.columns}
        metrics['Item_unc_corr'] = {column: np.corrcoef(rand_preds[1], item_vars[column].to_numpy().flatten())[0, 1]
                                    for column in data.item.columns}
        metrics['Pred_unc_corr'] = np.corrcoef(rand_preds[0], rand_preds[1])[0, 1]

        Cuts = {'Values': np.quantile(rand_preds[1], np.linspace(0.8, 0.2, 4)),
                'Coverage': np.ones((len(data.test_users), 5)),
                'MAP': np.zeros((len(data.test_users), 5)),
                'Surprise': np.zeros((len(data.test_users), 5))}

    if hasattr(model, 'uncertain_predict_user'):
        Uncertain_rec = accuracy_metrics(len(data.test_users), max_k, False)
        unc = 1 - model.uncertain_predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test[:, 1]).long(),
                                          threshold=4)
        URI_unc = {'test': {'avg': unc.mean(), 'std': unc.std()},
                   'rec': {'hits': 0, 'hits_unc': 0,
                           'avg': np.zeros(len(data.test_users))}}

    precision_denom = np.arange(1, max_k + 1)

    for idxu, user in enumerate(tqdm(data.test_users, desc=name+' - Recommending')):
        targets = data.test[data.test[:, 0] == user, 1]
        rated = data.train_val.item[data.train_val.user == user].to_numpy()
        rec = model.recommend(user, remove_items=rated, n=data.n_item - len(rated))
        top_k = rec[:max_k]
        hits = top_k.index.isin(targets)
        n_hits = hits.cumsum(0)

        if n_hits[-1] > 0:
            precision = n_hits / precision_denom
            Rating_rec['MAP'][idxu] = np.cumsum(precision * hits) / np.maximum(1, n_hits)
            Rating_rec['Recall'][idxu] = n_hits / len(targets)

        if isinstance(model, UncertainRecommender):
            unc = rec.uncertainties.to_numpy()
            top_k_unc = unc[:max_k]
            URI_rat['rec']['avg'][idxu] = top_k_unc.mean()
            hits_unc = np.sum(top_k_unc * hits)
            URI_rat['rec']['hits'] += n_hits[-1]
            URI_rat['rec']['hits_unc'] += hits_unc
            if n_hits[-1] > 0:
                avg_hits_unc = hits_unc / n_hits[-1]
                Rating_rec['URI_rec'][idxu] = (URI_rat['rec']['avg'][idxu] - avg_hits_unc) / top_k_unc.std()

            profile_distances = data.distances[top_k.index][:, rated]
            Cuts['Surprise'][idxu, 0] = profile_distances.min(1).sum(0) / max_k

            Cuts['MAP'][idxu, 0] = Rating_rec['MAP'][idxu][-1]
            for idxc, cut in enumerate(Cuts['Values']):
                if np.sum(top_k_unc < cut) == len(top_k_unc):
                    Cuts['MAP'][idxu, idxc + 1] = Cuts['MAP'][idxu, idxc]
                    Cuts['Surprise'][idxu, idxc + 1] = Cuts['Surprise'][idxu, idxc]
                else:
                    top_k_constrained = rec[unc < cut][:max_k]
                    top_k_unc = top_k_constrained.uncertainties.to_numpy()
                    Cuts['Coverage'][idxu][idxc + 1] = len(top_k_unc) / max_k
                    if len(top_k_constrained > 0):
                        Cuts['MAP'][idxu, idxc + 1] = get_AP(top_k_constrained.index.isin(targets))
                        profile_distances = data.distances[top_k_constrained.index][:, rated]
                        Cuts['Surprise'][idxu, idxc + 1] = profile_distances.min(1).sum(0) / len(top_k_constrained)

        if hasattr(model, 'uncertain_predict_user'):
            top_k_unc = model.uncertain_recommend(user, threshold=4, remove_items=rated, n=max_k)
            top_k_unc_unc = 1 - top_k_unc.scores.to_numpy()
            URI_unc['rec']['avg'][idxu] = top_k_unc_unc.mean()
            if np.sum(top_k_unc.index.to_numpy() == top_k.index.to_numpy()) == max_k:
                Uncertain_rec['MAP'][idxu] = Rating_rec['MAP'][idxu]
                Uncertain_rec['Recall'][idxu] = Rating_rec['Recall'][idxu]
            else:
                hits = top_k_unc.index.isin(targets)
                n_hits = hits.cumsum(0)
            URI_unc['rec']['hits'] += n_hits[-1]
            URI_unc['rec']['hits_unc'] += np.sum(top_k_unc_unc * hits)

            if n_hits[-1] > 0:
                precision = n_hits / precision_denom
                Uncertain_rec['MAP'][idxu] = np.cumsum(precision * hits) / np.maximum(1, n_hits)
                Uncertain_rec['Recall'][idxu] = n_hits / len(targets)

    metrics['Rating_rec'] = {'MAP': np.mean(Rating_rec['MAP'], 0),
                             'Recall': np.mean(Rating_rec['Recall'], 0)}

    if isinstance(model, UncertainRecommender):
        avg_unc_hits = URI_rat['rec']['hits_unc'] / URI_rat['rec']['hits']
        metrics['Rating_rec']['URI_global'] = (URI_rat['test']['avg'] - avg_unc_hits) / URI_rat['test']['std']
        metrics['Rating_rec']['Unc_MAP_corr'] = np.corrcoef(Rating_rec['MAP'][:, -1], URI_rat['rec']['avg'])[0, 1]
        metrics['Rating_rec']['URI_rec'] = np.nanmean(Rating_rec['URI_rec'])

        metrics['Cuts'] = {'Values': Cuts['Values'],
                           'Coverage': Cuts['Coverage'].mean(0),
                           'MAP': Cuts['MAP'].mean(0),
                           'Novelty': Cuts['Surprise'].mean(0),
                           'Map*': np.array([Cuts['MAP'][Cuts['Coverage'][:, k] == 1, k].mean() for k in range(5)]),
                           'Novelty*': np.array([Cuts['Surprise'][Cuts['Coverage'][:, k] == 1, k].mean() for k in range(5)])}

    if hasattr(model, 'uncertain_predict_user'):
        metrics['Uncertain_rec'] = {'MAP': np.mean(Uncertain_rec['MAP'], 0),
                                    'Recall': np.mean(Uncertain_rec['Recall'], 0)}
        avg_unc_hits = URI_unc['rec']['hits_unc'] / URI_unc['rec']['hits']
        metrics['Uncertain_rec']['URI_global'] = (URI_unc['test']['avg'] - avg_unc_hits) / URI_unc['test']['std']
        metrics['Uncertain_rec']['Unc_MAP_corr'] = np.corrcoef(Uncertain_rec['MAP'][:, -1], URI_unc['rec']['avg'])[0, 1]

    if name is not None:
        with open('results/' + name + '.pkl', 'wb') as f:
            pickle.dump(metrics, file=f)

    return metrics
