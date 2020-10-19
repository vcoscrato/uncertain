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


def quantiles(error, uncertainty):

    quantiles = torch.quantile(uncertainty, torch.linspace(0, 1, 21, device=uncertainty.device, dtype=uncertainty.dtype))
    rmse = torch.zeros(20)

    for idx in range(20):
        ind = torch.bitwise_and(quantiles[idx] <= uncertainty, uncertainty < quantiles[idx+1])
        rmse[idx] = torch.sqrt(torch.square(error[ind]).mean())

    return rmse


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


def precision_recall_score(model, test, train=None, max_k=10):
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
    precision_denom = torch.arange(1, max_k+1, device=test.user_ids.device)

    for user_id in range(test.num_users):

        targets = test.item_ids[test.user_ids == user_id]

        if not len(targets):
            continue

        predictions = model.predict(user_id)

        if type(predictions) is not tuple:
            predictions = -predictions
        else:
            predictions = -predictions[0]

        if train is not None:
            rated = train.item_ids[train.user_ids == user_id]
            predictions[rated] = float('inf')

        predictions = predictions.argsort()[:max_k]

        indices = torch.zeros_like(predictions, dtype=torch.bool)
        for elem in targets:
            indices = indices | (predictions == elem)
        num_hit = indices.cumsum(0)

        precision.append(num_hit / precision_denom)
        recall.append(num_hit / len(targets))

    precision = torch.vstack(precision)
    recall = torch.vstack(recall)

    return precision, recall
