import torch
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.special import comb
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from uncertain.core import VanillaRecommender, UncertainRecommender


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


def test_ratings(model, data, use_baseline, out):
    predictions = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test[:, 1]).long())
    if not isinstance(model, UncertainRecommender) and not use_baseline:
        predictions = predictions[~np.isnan(predictions)]
        out['RMSE'] = np.sqrt(np.square(predictions - data.test[:, 2]).mean())
    if isinstance(model, UncertainRecommender):
        idx = ~np.isnan(predictions[1])
        pred = predictions[0][idx]
        unc = predictions[1][idx]
        if not use_baseline:
            out['RMSE'] = np.sqrt(np.square(pred - data.test[:, 2]).mean())
        out['avg_unc'], out['std_unc'] = unc.mean(), unc.std()
        errors = np.abs(data.test[:, 2] - pred)
        out['RPI'] = rpi_score(errors, unc)
        out['Classification'] = classification(errors, unc)
        out['Quantile RMSE'] = quantile_score(errors, unc)


def accuracy_metrics(data, max_k):
    return {'Precision': np.zeros((data.n_user, max_k)),
            'Recall': np.zeros((data.n_user, max_k)),
            'NDCG': np.zeros((data.n_user, max_k))}


def test_recommendations(user, index, hits, cache, out):
    n_hit = hits.cumsum(0)

    if n_hit[-1] > 0:
        # Accuracy
        out['Precision'][user][:len(index)] = n_hit / cache['precision_denom'][:len(index)]
        out['Recall'][user][:len(index)] = n_hit / cache['n_target']

        # NDCG
        dcg = (hits / cache['ndcg_denom'][:len(index)]).cumsum(0)
        for k in range(len(index)):
            if dcg[k] > 0:
                idcg = np.sum(np.sort(hits[:k + 1]) / cache['ndcg_denom'][:k + 1])
                out['NDCG'][user][k] = dcg[k] / idcg


def test(model, data, name, max_k, use_baseline=False):
    
    metrics = {}
    if not use_baseline:
        metrics['accuracy'] = accuracy_metrics(data, max_k)
        
    if isinstance(model, UncertainRecommender):
        preds = model.predict(data.rand['users'], data.rand['items'])
        quantiles = np.linspace(0.8, 0.2, 4)
        cuts = np.quantile(preds[1], quantiles)
        metrics['uncertainty'] = {'quantiles': quantiles, 'cut_values': cuts, 'preds': preds,
                                  'RRI': np.zeros((data.n_user, max_k)) * np.NaN}
        metrics['cuts'] = [{**accuracy_metrics(data, max_k), **{'Coverage': np.ones((data.n_user, 1))}}
                           for _ in range(len(quantiles))]

        if hasattr(model, 'uncertain_predict_user'):
            metrics['uncertain_accuracy'] = accuracy_metrics(data, max_k)
        
    cache = {'precision_denom': np.arange(1, max_k + 1),
             'ndcg_denom': np.log2(np.arange(2, max_k + 2))}
    
    for user in tqdm(data.test_users):
        targets = data.test[data.test[:, 0] == user, 1]
        cache['n_target'] = len(targets)
        if not cache['n_target']:
            continue
        cache['rated'] = data.train_val.item[data.train_val.user == user].to_numpy()

        rec = model.recommend(user, remove_items=cache['rated'], n=data.n_item - len(cache['rated']))
        top_k = rec[:max_k]
        hits = top_k.index.isin(targets)

        if not use_baseline:
            test_recommendations(user, top_k.index, hits, cache, metrics['accuracy'])

        if isinstance(model, UncertainRecommender):
            # RRI
            unc = (rec.uncertainties.mean() - top_k.uncertainties) / rec.uncertainties.std() * hits
            metrics['uncertainty']['RRI'][user] = unc.cumsum(0) / cache['precision_denom']

            for idx in range(len(metrics['uncertainty']['quantiles'])):
                top_k_constrained = rec[rec.uncertainties < metrics['uncertainty']['cut_values'][idx]][:max_k]
                n_rec = len(top_k_constrained)
                if n_rec:
                    hits = top_k_constrained.index.isin(targets)
                    test_recommendations(user, top_k_constrained.index, hits, cache, metrics['cuts'][idx])
                    if n_rec < max_k:
                        metrics['cuts'][idx]['Coverage'][user] = 0

        if hasattr(model, 'uncertain_predict_user'):
            rec = model.uncertain_recommend(user, remove_items=cache['rated'], n=data.n_item - len(cache['rated']))
            top_k = rec[:max_k]
            hits = top_k.index.isin(targets)
            test_recommendations(user, top_k.index, hits, cache, metrics['uncertain_accuracy'])

    if not use_baseline:
        metrics['accuracy'] = {key: np.nanmean(value, 0) for key, value in metrics['accuracy'].items()}
    if isinstance(model, UncertainRecommender):
        metrics['uncertainty']['RRI'] = np.nanmean(metrics['uncertainty']['RRI'], 0)
        for idx in range(len(metrics['uncertainty']['quantiles'])):
            metrics['cuts'][idx] = {key: np.nanmean(value, 0) for key, value in metrics['cuts'][idx].items()}
        if hasattr(model, 'uncertain_predict_user'):
            metrics['uncertain_accuracy'] = {key: np.nanmean(value, 0) for key, value in
                                             metrics['uncertain_accuracy'].items()}

    with open('results/' + name + '.pkl', 'wb') as f:
        pickle.dump(metrics, file=f)

    return 'Success!'
        
    
def unc_distribution(model):
    f, ax = plt.subplots(ncols=2)
    preds = model.predict(data.rand['users'], data.rand['items'])
    ax[0].hist(preds[1])
    ax[0].set_xlabel('Uncertainty')
    ax[0].set_ylabel('Density')
    ax[1].plot(preds[0], preds[1], 'o')
    ax[1].set_xlabel('Score')
    ax[1].set_ylabel('Uncertainty')
    f.tight_layout()
