import torch
import numpy as np
import pandas as pd
from scipy.special import comb
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_distances


def rpi_score(errors, uncertainties):

    MAE = errors.mean()
    errors_deviations = errors - MAE
    uncertainties_deviations = uncertainties - uncertainties.mean()
    errors_std = np.abs(errors_deviations).mean()
    epsilon_std = np.abs(uncertainties_deviations).mean()

    num = np.mean(errors * errors_deviations * uncertainties_deviations)
    denom = errors_std*epsilon_std*MAE

    return num/denom


def classification(errors, uncertainties):

    errors = errors
    uncertainties = uncertainties

    splitter = KFold(n_splits=2, shuffle=True, random_state=0)
    targets = errors > 1
    auc = 0

    for train_index, test_index in splitter.split(errors):
        mod = LogisticRegression().fit(uncertainties[train_index].reshape(-1, 1), targets[train_index])
        probs = mod.predict_proba(uncertainties[test_index].reshape(-1, 1))
        auc += roc_auc_score(targets[test_index], probs[:, 1]) / 2

    return auc


def quantile_score(errors, uncertainties):
    quantiles = np.quantile(uncertainties, np.linspace(0, 1, 21))
    q_rmse = np.zeros(20)
    for idx in range(20):
        ind = np.bitwise_and(quantiles[idx] <= uncertainties, uncertainties < quantiles[idx + 1])
        q_rmse[idx] = np.sqrt(np.square(errors[ind]).mean())
    return q_rmse


def test_recommendations(model, data, max_k=10):

    metrics = {'Precision': np.zeros((data.n_user, max_k)),
               'Recall': np.zeros((data.n_user, max_k)),
               'NDCG': np.zeros((data.n_user, max_k)),
               'Diversity': np.zeros((data.n_user, max_k - 1)),
               'Novelty': np.zeros((data.n_user, max_k)) * np.NaN}

    predictions = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test[:, 1]).long())
    rating_metrics = {}
    if type(predictions) == tuple:
        metrics['Uncertainty'] = np.zeros((data.n_user, max_k*5))
        metrics['RRI'] = np.zeros((data.n_user, max_k)) * np.NaN
        avg_unc, std_unc = predictions[1].mean(), predictions[1].std()
        if not data.implicit:
            rating_metrics['RMSE'] = np.sqrt(np.square(predictions[0] - data.test[:, 2]).mean())
            errors = np.abs(data.test[:, 2] - predictions[0])
            rating_metrics['RPI'] = rpi_score(errors, predictions[1])
            rating_metrics['Classification'] = classification(errors, predictions[1])
            rating_metrics['Quantile RMSE'] = quantile_score(errors, predictions[1])
            avg_unc, std_unc = predictions[1].mean(), predictions[1].std()
    elif not data.implicit:
        rating_metrics['RMSE'] = np.sqrt(np.square(predictions - data.test[:, 2]).mean())

    diversity_denom = comb(np.arange(2, max_k+1), 2)
    precision_denom = np.arange(1, max_k + 1)
    ndcg_denom = np.log2(np.arange(2, max_k+2))

    for user in range(data.n_user):
        targets = data.test[:, 1][data.test[:, 0] == user]
        n_target = len(targets)
        if not n_target:
            continue

        rated = data.train_val.item[data.train_val.user == user].to_numpy()
        rec = model.recommend(user, remove_items=rated, n=data.n_item-len(rated))
        hits = rec.index[:max_k].isin(targets)
        n_hit = hits.cumsum(0)

        if hasattr(rec, 'uncertainties'):
            metrics['Uncertainty'][user] = rec.uncertainties[:max_k*5]
        '''
        # AUC
        targets_pos = sorted([rec.index.get_loc(i) for i in targets])
        n_negative = data.n_item - n_target
        n_after_target = [n_negative - pos + i for i, pos in enumerate(targets_pos)]
        metrics['AUC'] = sum(n_after_target) / (len(n_after_target) * n_negative)
        '''
        # Diversity
        distance = np.triu(cosine_distances(data.csr[rec.index[:max_k]]), 1) / 2
        metrics['Diversity'][user] = np.diag(distance.cumsum(0).cumsum(1), 1) / diversity_denom

        if hits.sum() > 0:
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

            # Expected surprise (novelty)
            distance = cosine_distances(data.csr[rec.index[:max_k]], data.csr[rated]) / 2
            metrics['Novelty'][user] = (distance.min(1) * hits).cumsum(0) / hits.cumsum(0)

    return {**rating_metrics, **{key: np.nanmean(value, 0) for key, value in metrics.items()}}


def uncertainty_distributions(model, size=10000):
    users = torch.randint(0, model.n_user, (size,))
    items = torch.randint(0, model.n_item, (size,))
    with torch.no_grad():
        preds = model.forward(users, items)
    try:
        f, ax = plt.subplots(ncols=2)
        ax[0].hist(preds[0].numpy(), density=True)
        ax[1].hist(preds[1].numpy(), density=True)
    except:
        ax[0].text(0, 0, 'Error', ha='center', fontsize=20)
        ax[1].text(0, 0, 'Error', ha='center', fontsize=20)
    ax[0].set_xlabel('Relevance', fontsize=20)
    ax[0].set_ylabel('Density', fontsize=20)
    ax[1].set_xlabel('Uncertainty', fontsize=20)
    ax[1].set_ylabel('Density', fontsize=20)
    f.tight_layout()
    return f
