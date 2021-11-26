import torch
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.special import comb
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_distances
from uncertain.core import VanillaRecommender, UncertainRecommender


def rpi_score(errors, uncertainties):
    mae = errors.mean()
    errors_deviations = errors - mae
    uncertainties_deviations = uncertainties - uncertainties.mean()
    errors_std = np.abs(errors_deviations).mean()
    epsilon_std = np.abs(uncertainties_deviations).mean()
    num = np.mean(errors * errors_deviations * uncertainties_deviations)
    denom = errors_std*epsilon_std*mae
    return num/denom


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


def test_ratings(model, data, use_baseline, out):
    predictions = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test[:, 1]).long())
    if not isinstance(model, UncertainRecommender) and not use_baseline:
        out['RMSE'] = np.sqrt(np.square(predictions - data.test[:, 2]).mean())
    if isinstance(model, UncertainRecommender):
        if not use_baseline:
            out['RMSE'] = np.sqrt(np.square(predictions[0] - data.test[:, 2]).mean())
        out['avg_unc'], out['std_unc'] = predictions[1].mean(), predictions[1].std()
        errors = np.abs(data.test[:, 2] - predictions[0])
        out['RPI'] = rpi_score(errors, predictions[1])
        out['Classification'] = classification(errors, predictions[1])
        out['Quantile RMSE'] = quantile_score(errors, predictions[1])


def get_cuts(model, data):
    users = torch.randint(0, data.n_user, (100000,))
    items = torch.randint(0, data.n_item, (100000,))
    unc = model.predict(users, items)[1]
    cuts = np.quantile(unc, [2 / 3, 1 / 3])
    f, ax = plt.subplots(figsize=(10, 5))
    ax.hist(unc, density=True, color='k')
    ax.axvline(x=cuts[0], color='g', label=r'$\frac{2}{3}$ Quantile', linewidth=6)
    ax.axvline(x=cuts[1], color='r', label=r'$\frac{1}{3}$ Quantile', linewidth=6)
    ax.set_ylabel('Density', fontsize=20)
    ax.set_xlabel('Uncertainty', fontsize=20)
    ax.legend()
    f.tight_layout()
    return {'values': cuts, 'plot': f}


def accuracy_metrics(data, max_k):
    return {'Precision': np.zeros((data.n_user, max_k)),
            'Recall': np.zeros((data.n_user, max_k)),
            'NDCG': np.zeros((data.n_user, max_k)),
            'Diversity': np.zeros((data.n_user, max_k - 1)) * np.NaN,
            'Expected surprise': np.zeros((data.n_user, max_k))}


def test_recommendations(user, index, hits, cache, out):
    n_hit = hits.cumsum(0)

    # Diversity
    inner_distances = np.triu(cosine_distances(cache['csr'][index]), 1) / 2
    out['Diversity'][user] = np.diag(inner_distances.cumsum(0).cumsum(1), 1) / cache['diversity_denom']

    if n_hit[-1] > 0:
        # Accuracy
        out['Precision'][user] = n_hit / cache['precision_denom']
        out['Recall'][user] = n_hit / cache['n_target']

        # NDCG
        dcg = (hits / cache['ndcg_denom']).cumsum(0)
        for k in range(len(index)):
            if dcg[k] > 0:
                idcg = np.sum(np.sort(hits[:k + 1]) / cache['ndcg_denom'][:k + 1])
                out['NDCG'][user][k] = dcg[k] / idcg

        # Expected surprise
        profile_distances = cosine_distances(cache['csr'][index], cache['csr_rated']) / 2
        out['Expected surprise'][user] = (profile_distances.min(1) * hits).cumsum(0) / hits.cumsum(0)


def test(model, data, name, threshold=4, max_k=10, use_baseline=False):
    metrics = {'ratings': {}}
    test_ratings(model, data, use_baseline, metrics['ratings'])

    if not use_baseline:
        metrics['accuracy'] = accuracy_metrics(data, max_k)

    if isinstance(model, UncertainRecommender):
        metrics['uncertainty'] = {'Uncertainty_TopK': np.zeros((data.n_user, max_k*4)),
                                  'RRI': np.zeros((data.n_user, max_k)) * np.NaN}

        metrics['cuts'] = get_cuts(model, data)
        metrics['cuts']['2/3'] = {**accuracy_metrics(data, max_k), **{'Coverage': np.ones((data.n_user, 1))}}
        metrics['cuts']['1/3'] = {**accuracy_metrics(data, max_k), **{'Coverage': np.ones((data.n_user, 1))}}

        if hasattr(model, 'uncertain_predict_user'):
            metrics['uncertain_accuracy'] = accuracy_metrics(data, max_k)

    cache = {'precision_denom': np.arange(1, max_k + 1),
             'ndcg_denom': np.log2(np.arange(2, max_k+2)),
             'diversity_denom': np.arange(1, max_k).cumsum(0),
             'csr': data.csr}

    for user in tqdm(range(data.n_user)):
        targets = data.test[data.test[:, 0] == user]
        targets = targets[targets[:, 2] >= threshold, 1]
        cache['n_target'] = len(targets)
        if not cache['n_target']:
            continue
        rated = data.train_val.item[data.train_val.user == user].to_numpy()
        cache['csr_rated'] = data.csr[rated]

        rec = model.recommend(user, remove_items=rated, n=data.n_item - len(rated))
        top_k = rec[:max_k]
        hits = top_k.index.isin(targets)

        if not use_baseline:
            test_recommendations(user, top_k.index, hits, cache, metrics['accuracy'])

        if isinstance(model, UncertainRecommender):
            metrics['uncertainty']['Uncertainty_TopK'][user] = rec.uncertainties[:max_k*4]

            # RRI
            unc = (metrics['ratings']['avg_unc'] - top_k.uncertainties) / metrics['ratings']['std_unc'] * hits
            metrics['uncertainty']['RRI'][user] = unc.cumsum(0) / cache['precision_denom']

            for idx, cut in zip([0, 1], ['2/3', '1/3']):
                top_k_constrained = rec[rec.uncertainties < metrics['cuts']['values'][idx]][:max_k]
                if not use_baseline and top_k_constrained.equals(top_k):
                    for key, value in metrics['cuts'][cut].items():
                        if key != 'Coverage':
                            value[user] = metrics['accuracy'][key][user]
                else:
                    n_rec = len(top_k_constrained)
                    if n_rec:
                        hits = top_k_constrained.index.isin(targets)
                        test_recommendations(user, top_k_constrained.index, hits, cache, metrics['cuts'][cut])
                        top_k = top_k_constrained
                        if n_rec < max_k:
                            metrics['cuts'][cut]['Coverage'][user] = 0

        if hasattr(model, 'uncertain_predict_user'):
            rec = model.uncertain_recommend(user, threshold=threshold, remove_items=rated, n=data.n_item - len(rated))
            top_k = rec[:max_k]
            hits = top_k.index.isin(targets)
            test_recommendations(user, top_k.index, hits, cache, metrics['uncertain_accuracy'])

    if not use_baseline:
        metrics['accuracy'] = {key: np.nanmean(value, 0) for key, value in metrics['accuracy'].items()}
    if isinstance(model, UncertainRecommender):
        metrics['uncertainty'] = {key: np.nanmean(value, 0) for key, value in metrics['uncertainty'].items()}
        for cut in ['2/3', '1/3']:
            metrics['cuts'][cut] = {key: np.nanmean(value, 0) for key, value in metrics['cuts'][cut].items()}
        if hasattr(model, 'uncertain_predict_user'):
            metrics['uncertain_accuracy'] = {key: np.nanmean(value, 0) for key, value in
                                             metrics['uncertain_accuracy'].items()}

    with open('results/'+name+'.pkl', 'wb') as f:
        pickle.dump(metrics, file=f)

    return metrics
