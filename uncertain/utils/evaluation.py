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


def test_recommendations(model, data, name, max_k=4):

    rating_metrics = {}
    metrics = {'Precision': np.zeros((data.n_user, max_k)),
               'Recall': np.zeros((data.n_user, max_k)),
               'NDCG': np.zeros((data.n_user, max_k))}

    predictions = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test[:, 1]).long())
    if type(predictions) == tuple:
        metrics['Uncertainty'] = np.zeros((data.n_user, max_k*5))
        metrics['RRI'] = np.zeros((data.n_user, max_k)) * np.NaN
        avg_unc, std_unc = predictions[1].mean(), predictions[1].std()
        users = torch.randint(0, data.n_user, (100000,))
        items = torch.randint(0, data.n_item, (100000,))
        unc = model.predict(users, items)[1]
        cuts = np.quantile(unc, [1, 2/3, 1/3])
        cuts_metrics = {'Precision': np.zeros((data.n_user, 3)) * np.NaN,
                        'Coverage': np.zeros((data.n_user, 3)),
                        'Diversity': np.zeros((data.n_user, 3)) * np.NaN,
                        'Expected surprise': np.zeros((data.n_user, 3)) * np.NaN}

        f, ax = plt.subplots(figsize=(10, 5))
        ax.hist(unc, density=True, color='k')
        ax.axvline(x=cuts[1], color='g', label=r'$\frac{2}{3}$ Quantile', linewidth=6)
        ax.axvline(x=cuts[2], color='r', label=r'$\frac{1}{3}$ Quantile', linewidth=6)
        ax.set_ylabel('Density', fontsize=20)
        ax.set_xlabel('Uncertainty', fontsize=20)
        f.tight_layout()
        ax.legend()
        f.savefig('plots/' + name + ' distribution.pdf')

        if not data.implicit:
            rating_metrics['RMSE'] = np.sqrt(np.square(predictions[0] - data.test[:, 2]).mean())
            errors = np.abs(data.test[:, 2] - predictions[0])
            rating_metrics['RPI'] = rpi_score(errors, predictions[1])
            rating_metrics['Classification'] = classification(errors, predictions[1])
            rating_metrics['Quantile RMSE'] = quantile_score(errors, predictions[1])
            avg_unc, std_unc = predictions[1].mean(), predictions[1].std()

    elif not data.implicit:
        rating_metrics['RMSE'] = np.sqrt(np.square(predictions - data.test[:, 2]).mean())

    precision_denom = np.arange(1, max_k + 1)
    ndcg_denom = np.log2(np.arange(2, max_k+2))

    for user in tqdm(range(data.n_user)):
        targets = data.test[:, 1][data.test[:, 0] == user]
        n_target = len(targets)
        if not n_target:
            continue

        rated = data.train_val.item[data.train_val.user == user].to_numpy()
        rec = model.recommend(user, remove_items=rated, n=data.n_item-len(rated))
        hits = rec.index[:max_k].isin(targets)
        n_hit = hits.cumsum(0)

        if n_hit[-1] > 0:
            # Accuracy
            metrics['Precision'][user] = n_hit / precision_denom
            metrics['Recall'][user] = n_hit / n_target

            # NDCG
            dcg = (hits / ndcg_denom).cumsum(0)
            for k in range(max_k):
                if dcg[k] > 0:
                    idcg = np.sum(np.sort(hits[:k+1]) / ndcg_denom[:k+1])
                    metrics['NDCG'][user][k] = dcg[k] / idcg

            # RRI
            if hasattr(rec, 'uncertainties'):
                unc = (avg_unc - rec.uncertainties[:max_k]) / std_unc * hits
                metrics['RRI'][user] = unc.cumsum(0) / precision_denom

        if hasattr(rec, 'uncertainties'):
            metrics['Uncertainty'][user] = rec.uncertainties[:max_k*5]
            for i in [0, 1, 2]:
                certain_idx = rec[rec.uncertainties < cuts[i]][:max_k].index
                hits = certain_idx.isin(targets)
                n_hit = hits.sum()
                if len(certain_idx >= max_k):
                    cuts_metrics['Precision'][user, i] = n_hit / max_k
                    cuts_metrics['Coverage'][user, i] = 1
                    sum_distance = np.sum(cosine_distances(data.csr[certain_idx]) / 2)
                    cuts_metrics['Diversity'][user, i] = sum_distance / (max_k * max_k - max_k)
                if n_hit > 0:
                    min_distance = cosine_distances(data.csr[certain_idx], data.csr[rated]).min(1) / 2
                    cuts_metrics['Expected surprise'][user, i] = (min_distance * hits).sum() / n_hit

        out = {**rating_metrics, **{key: np.nanmean(value, 0) for key, value in metrics.items()}}
        try:
            out.update({**{'Cuts '+key: np.nanmean(value, 0) for key, value in cuts_metrics.items()}})
        except:
            pass
        with open('results/'+name+'.pkl', 'wb') as f:
            pickle.dump(out, file=f)

    return out
