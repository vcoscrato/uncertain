import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def rmse_score(predictions, ratings):

    return torch.sqrt(((ratings - predictions) ** 2).mean())


def pearson_correlation(x, y):
    
    x_deviations = x - x.mean()
    y_deviations = y - y.mean()
    x_std = torch.sqrt((x_deviations**2).sum())
    y_std = torch.sqrt((y_deviations**2).sum())

    return (x_deviations * y_deviations).sum() / (x_std * y_std)
    

def correlation(error, uncertainty):

    pearson = pearson_correlation(error, uncertainty)

    a = error.clone()
    b = uncertainty.clone()
    a[a.argsort()] = torch.arange(len(a), dtype=a.dtype, device=a.device)
    b[b.argsort()] = torch.arange(len(b), dtype=b.dtype, device=b.device)
    spearman = pearson_correlation(a, b)

    return pearson, spearman


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
    likelihood = 0
    auc = 0

    for train_index, test_index in splitter.split(error):
        mod = LogisticRegression().fit(uncertainty[train_index].reshape(-1, 1), targets[train_index])
        probs = mod.predict_proba(uncertainty[test_index].reshape(-1, 1))
        likelihood += np.log(probs[range(len(probs)), targets[test_index].astype(int)]).mean() / 2
        auc += roc_auc_score(targets[test_index], probs[:, 1]) / 2

    return likelihood, auc


def recommendation_score(model, test, train=None, relevance_threshold=4, max_k=10):
    """
    Compute Precision@k and Recall@k scores. One score
    is given for every user with interactions in the test
    set, representing the Precision@k and Recall@k of all their
    test items.
    Parameters
    ----------
    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will not affect the computed metrics.
    k: int or tensor of int,
        The maximum number of predicted items
    Returns
    -------
    (Precision@k, Recall@k): numpy tensor of shape (num_users, len(k))
        A tuple of Precisions@k and Recalls@k for each user in test.
        If k is a scalar, will return a tuple of vectors. If k is an
        tensor, will return a tuple of tensors, where each row corresponds
        to a user and each column corresponds to a value of k.
    """

    precision = []
    recall = []
    average_uncertainty = []
    rri = []
    precision_denom = torch.arange(1, max_k+1, device=test.user_ids.device)

    for user_id in range(test.num_users):

        targets = test.item_ids[torch.logical_and(test.user_ids == user_id, test.ratings >= relevance_threshold)]

        if not len(targets):
            continue

        predictions, uncertainties = model.recommend(user_id, train)
        if model._is_uncertain:
            average_uncertainty.append(uncertainties.mean())

        hits = torch.zeros_like(predictions, dtype=torch.bool)
        for elem in targets:
            hits = hits | (predictions == elem)
        num_hit = hits.cumsum(0)

        precision.append(num_hit / precision_denom)
        recall.append(num_hit / len(targets))
        if uncertainties is not None and hits.sum().item() > 0:
            rri_ = torch.empty(max_k - 1)
            for i in range(2, max_k+1):
                unc = uncertainties[:i]
                rri_[i-2] = (unc.mean() - unc[hits[:i]]).mean() / unc.std()
            rri.append(rri_)

    precision = torch.vstack(precision)
    recall = torch.vstack(recall)
    if len(rri) > 0:
        average_uncertainty = torch.tensor(average_uncertainty, device=precision.device)
        rri = torch.vstack(rri)

    return precision, recall, average_uncertainty, rri
