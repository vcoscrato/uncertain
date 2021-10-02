import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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
