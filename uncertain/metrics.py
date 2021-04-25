import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from matplotlib import pyplot as plt


def rmse_score(predictions, ratings):

    return torch.sqrt(((ratings - predictions) ** 2).mean())


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

    return (x_deviations * y_deviations).sum() / (x_std * y_std)
    

def error_uncertainty_correlation(predictions):

    errors = predictions[0]
    uncertainties = predictions[1]
    pearson = pearson_correlation(errors, uncertainties).item()
    spearman = pearson_correlation(errors.argsort().argsort().float(), uncertainties.argsort().argsort().float()).item()

    return {'Pearson': pearson, 'Spearman': spearman}


def rpi_score(errors, uncertainty):

    MAE = errors.mean()
    error_deviations = errors - MAE
    uncertainty_deviations = uncertainty - uncertainty.mean()
    errors_std = torch.abs(error_deviations).mean()
    epsilon_std = torch.abs(uncertainty_deviations).mean()

    return (errors*error_deviations*uncertainty_deviations).mean()/(errors_std*epsilon_std*MAE)


def classification(error, uncertainty):

    error = error.cpu().detach().numpy()
    uncertainty = uncertainty.cpu().detach().numpy()

    splitter = KFold(n_splits=2, shuffle=True, random_state=0)
    targets = error > 1
    auc = 0

    for train_index, test_index in splitter.split(error):
        mod = LogisticRegression().fit(uncertainty[train_index].reshape(-1, 1), targets[train_index])
        probs = mod.predict_proba(uncertainty[test_index].reshape(-1, 1))
        auc += roc_auc_score(targets[test_index], probs[:, 1]) / 2

    return auc


def determination(error, uncertainty):

    error = error.cpu().detach().numpy()
    uncertainty = uncertainty.cpu().detach().numpy()

    splitter = KFold(n_splits=2, shuffle=True, random_state=0)
    r2 = 0

    for train_index, test_index in splitter.split(error):
        mod = LinearRegression().fit(uncertainty[train_index].reshape(-1, 1), error[train_index])
        preds = mod.predict(uncertainty[test_index].reshape(-1, 1))
        r2 += r2_score(error[test_index], preds) / 2

    plt.scatter(uncertainty[test_index], error[test_index], color='black')
    plt.plot(uncertainty[test_index], preds, color='blue', linewidth=3)
    plt.show()

    return r2


def get_hits(recommendations, targets):

    hits = torch.zeros_like(recommendations.items)
    for idx, item in enumerate(recommendations.items):
        if item in targets:
            hits[idx] = 1

    return hits
