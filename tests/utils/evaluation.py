import torch
import numpy as np
import pandas as pd
from scipy.special import comb
from matplotlib import pyplot as plt
from uncertain.metrics import rpi_score, classification, quantile_score


def test_ratings(model, data):
    out = {}
    predictions = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test[:, 1]).long())
    if type(predictions) == tuple:
        out['RMSE'] = np.sqrt(np.square(predictions[0] - data.test[:, 2]).mean())
        errors = np.abs(data.test[:, 2] - predictions[0])
        out['RPI'] = rpi_score(errors, predictions[1])
        out['Classification'] = classification(errors, predictions[1])
        out['Quantile RMSE'] = quantile_score(errors, predictions[1])
    else:
        out['RMSE'] = np.sqrt(np.square(predictions - data.test[:, 2]).mean())
    return out


def test_recommendations(model, data, max_k=10):

    metrics = {'Precision': np.zeros((data.n_user, max_k)),
               'Recall': np.zeros((data.n_user, max_k)),
               'NDCG': np.zeros((data.n_user, max_k)),
               'Diversity': np.zeros((data.n_user, max_k - 1)),
               'Novelty': np.zeros((data.n_user, max_k)) * np.NaN}
    if hasattr(model.recommend(0), 'uncertainties'):
        metrics['RRI'] = np.zeros((data.n_user, max_k - 1)) * np.NaN

    diversity_denom = comb(np.arange(2, max_k+1), 2)
    precision_denom = np.arange(1, max_k + 1)
    ndcg_denom = np.log2(np.arange(2, max_k+2))

    for user in range(data.n_user):
        targets = data.test[:, 1][data.test[:, 0] == user]
        n_target = len(targets)
        if not n_target:
            continue

        rated = data.train[:, 1][data.train[:, 0] == user]
        rec = model.recommend(user, remove_items=rated, n=data.n_item-len(rated))
        hits = rec.index[:max_k].isin(targets)
        n_hit = hits.cumsum(0)
        with torch.no_grad():
            rec_embeddings = model.item_embeddings(torch.tensor(rec.index[:max_k]).long())
        '''
        # AUC
        targets_pos = sorted([rec.index.get_loc(i) for i in targets])
        n_negative = data.n_item - n_target
        n_after_target = [n_negative - pos + i for i, pos in enumerate(targets_pos)]
        metrics['AUC'] = sum(n_after_target) / (len(n_after_target) * n_negative)
        '''
        # Diversity
        distance = torch.triu(1 - torch.cosine_similarity(rec_embeddings, rec_embeddings.unsqueeze(1), dim=-1), 1) / 2
        metrics['Diversity'][user] = distance.cumsum(0).cumsum(1).diag(1).numpy() / diversity_denom

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
                for k in range(1, max_k):
                    unc = rec.uncertainties[:k+1]
                    metrics['RRI'][user][k-1] = (unc.mean() - unc[hits[:k+1]].mean()) / unc.std()

            # Expected surprise (novelty)
            with torch.no_grad():
                rated_embeddings = model.item_embeddings(torch.tensor(rated).long())
            for k in range(max_k):
                if hits[:k+1].sum() > 0:
                    hits_embeddings = rec_embeddings[:k+1][torch.tensor(hits[:k+1])]
                    distance = 1 - torch.cosine_similarity(hits_embeddings, rated_embeddings.unsqueeze(1), dim=-1)
                    metrics['Novelty'][user][k] = distance.max(0).values.mean().item() / 2

    return {key: np.nanmean(value, 0) for key, value in metrics.items()}


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
