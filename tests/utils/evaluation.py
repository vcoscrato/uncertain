import torch
import numpy as np
import pandas as pd
from scipy.special import comb
from matplotlib import pyplot as plt
from uncertain.metrics import rpi_score, classification, quantile_score


def test_recommendations(model, data, max_k=10):

    metrics = {'Precision': np.zeros((data.n_user, max_k)),
               'Recall': np.zeros((data.n_user, max_k)),
               'NDCG': np.zeros((data.n_user, max_k)),
               'Diversity': np.zeros((data.n_user, max_k - 1)),
               'Novelty': np.zeros((data.n_user, max_k)) * np.NaN}

    predictions = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test[:, 1]).long())
    rating_metrics = {}
    if type(predictions) == tuple:
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

        rated = data.train[:, 1][data.train[:, 0] == user].astype('int')
        rec = model.recommend(user, remove_items=rated, n=data.n_item-len(rated))
        hits = rec.index[:max_k].isin(targets)
        n_hit = hits.cumsum(0)
        with torch.no_grad():
            rec_distance = (1 - data.item_similarity[rec.index[:max_k]]) / 2
        '''
        # AUC
        targets_pos = sorted([rec.index.get_loc(i) for i in targets])
        n_negative = data.n_item - n_target
        n_after_target = [n_negative - pos + i for i, pos in enumerate(targets_pos)]
        metrics['AUC'] = sum(n_after_target) / (len(n_after_target) * n_negative)
        '''
        # Diversity
        distance = np.triu(rec_distance[:, rec.index[:max_k]], 1)
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
            metrics['Novelty'][user] = (rec_distance[:, rated].min(1) * hits).cumsum(0) / hits.cumsum(0)

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