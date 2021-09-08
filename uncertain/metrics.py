import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from matplotlib import pyplot as plt


def rmse_score(predictions, ratings):

    return torch.sqrt(((ratings - predictions) ** 2).mean()).item()


def hit_rate(recommendations, targets):

    hits = 0
    for item in recommendations.items:
        if item in targets:
            hits += 1

    return hits / len(recommendations.items)


def surprise(recommendations, model, rated_factor):

    out = []
    for item in recommendations.items:
        with torch.no_grad():
            item_factor = model._net.item_embeddings(item)
        out.append(torch.cosine_similarity(item_factor, rated_factor, dim=-1).min().item())

    return out


def pearson_correlation(x, y):

    x_deviations = x - x.mean()
    y_deviations = y - y.mean()
    x_std = torch.sqrt((x_deviations**2).sum())
    y_std = torch.sqrt((y_deviations**2).sum())

    return ((x_deviations * y_deviations).sum() / (x_std * y_std)).item()


def correlation(errors, uncertainties):

    pearson = pearson_correlation(errors, uncertainties)
    spearman = pearson_correlation(errors.argsort().argsort().float(), uncertainties.argsort().argsort().float())

    return {'Pearson': pearson, 'Spearman': spearman}


def rpi_score(errors, uncertainties):

    MAE = errors.mean()
    errors_deviations = errors - MAE
    uncertainties_deviations = uncertainties - uncertainties.mean()
    errors_std = torch.abs(errors_deviations).mean()
    epsilon_std = torch.abs(uncertainties_deviations).mean()

    num = torch.mean(errors * errors_deviations * uncertainties_deviations)
    denom = errors_std*epsilon_std*MAE

    return (num/denom).item()


def classification(errors, uncertainties):

    errors = errors.cpu().detach().numpy()
    uncertainties = uncertainties.cpu().detach().numpy()

    splitter = KFold(n_splits=2, shuffle=True, random_state=0)
    targets = errors > 1
    auc = 0

    for train_index, test_index in splitter.split(errors):
        mod = LogisticRegression().fit(uncertainties[train_index].reshape(-1, 1), targets[train_index])
        probs = mod.predict_proba(uncertainties[test_index].reshape(-1, 1))
        auc += roc_auc_score(targets[test_index], probs[:, 1]) / 2

    return auc


def determination(errors, uncertainties):

    errors = errors.cpu().detach().numpy()
    uncertainties = uncertainties.cpu().detach().numpy()

    splitter = KFold(n_splits=2, shuffle=True, random_state=0)
    r2 = 0

    for train_index, test_index in splitter.split(errors):
        mod = LinearRegression().fit(uncertainties[train_index].reshape(-1, 1), errors[train_index])
        preds = mod.predict(uncertainties[test_index].reshape(-1, 1))
        r2 += r2_score(errors[test_index], preds) / 2

    plt.scatter(uncertainties[test_index], errors[test_index], color='black')
    plt.plot(uncertainties[test_index], preds, color='blue', linewidth=3)
    plt.show()

    return r2


def quantile_score(errors, uncertainties):
    quantiles = torch.quantile(uncertainties[torch.randperm(len(uncertainties))[:int(1e5)]],
                               torch.linspace(0, 1, 21, dtype=uncertainties.dtype, device=uncertainties.device))
    q_rmse = torch.zeros(20)
    for idx in range(20):
        ind = torch.bitwise_and(quantiles[idx] <= uncertainties, uncertainties < quantiles[idx + 1])
        q_rmse[idx] = torch.sqrt(torch.square(errors[ind]).mean()).item()
    return q_rmse


def get_hits(recommendations, targets):

    hits = torch.zeros_like(recommendations.items)
    for idx, item in enumerate(recommendations.items):
        if item in targets:
            hits[idx] = 1

    return hits


def ndcg(hits, denom):
    if hits.sum() == 0:
        return torch.zeros_like(hits)
    dcg = torch.cumsum(hits / denom, 0)
    sorted_hits = hits.sort(descending=True).values
    idcg = torch.cumsum(sorted_hits / denom, 0)
    return dcg/idcg
