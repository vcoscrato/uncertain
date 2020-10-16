import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def rmse_score(predictions, ratings):

    return np.sqrt(((ratings - predictions) ** 2).mean())


def rpi_score(predictions, ratings):

    predictions, confidence = predictions
    errors = np.abs(predictions - ratings)

    MAE = errors.mean()
    error_deviations = errors - MAE
    confidence_deviations = confidence.mean() - confidence
    errors_std = np.abs(error_deviations).mean()
    epsilon_std = np.abs(confidence_deviations).mean()

    return (errors*error_deviations*confidence_deviations).mean()/(errors_std*epsilon_std*MAE)


def graphs_score(predictions, ratings):

    predictions, epsilons = predictions
    quantiles = np.quantile(epsilons, np.linspace(start=0, stop=1, num=21, endpoint=True))

    rmse = []
    width = []

    for idx in range(1, 21):
        ind = np.bitwise_and(quantiles[idx-1] <= epsilons, epsilons <= quantiles[idx])
        errors = np.abs(predictions[ind] - ratings[ind])
        rmse.append(np.sqrt(np.square(errors).mean()))
        width.append(np.quantile(errors, 0.95))

    return np.array(rmse), np.array(width)


def _get_rri(predictions, reliabilities, avg_rel, std_rel, targets, k):

    predictions = predictions[:k]
    reliabilities = reliabilities[:k]
    hit = list(set(predictions).intersection(set(targets)))
    num_hit = float(len(hit))
    if num_hit == 0:
        return np.nan
    deviations = reliabilities[np.isin(predictions, hit)] - avg_rel

    return (deviations.sum() / std_rel) / num_hit


def rri_score(model, test, train=None, k=10):

    reliabilities = model.predict(test.user_ids, test.item_ids)[1]
    avg_reliability, std_reliability = reliabilities.mean(), reliabilities.std()

    if np.isscalar(k):
        k = np.array([k])

    idx = test.ratings >= 4
    test_ = deepcopy(test)
    test_.user_ids = test_.user_ids[idx]
    test_.item_ids = test_.item_ids[idx]
    test_.ratings = test_.ratings[idx]
    test_ = test_.tocsr()
    if train is not None:
        train = train.tocsr()

    rri = []

    for user_id, row in enumerate(test_):

        if not len(row.indices):
            continue

        predictions, reliabilities = model.predict(user_id)
        predictions *= -1

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = np.infty

        predictions = predictions.argsort()
        targets = row.indices

        user_rri = [
            _get_rri(predictions, reliabilities, avg_reliability, std_reliability, targets, x) for x in k]

        rri.append(user_rri)

    rri = np.array(rri).squeeze()

    return rri


def _get_precision_recall_rri(predictions, reliabilities, avg_rel, std_rel, targets, k):

    predictions = predictions[:k]
    reliabilities = reliabilities[:k]
    hit = list(set(predictions).intersection(set(targets)))
    num_hit = float(len(hit))
    if num_hit == 0:
        return 0, 0, np.nan
    deviations = reliabilities[np.isin(predictions, hit)] - avg_rel

    return num_hit / len(predictions), num_hit / len(targets), (deviations.sum() / std_rel) / num_hit


def precision_recall_rri_score(model, test, train=None, k=10):

    idx = test.ratings >= 4
    test_ = deepcopy(test)
    test_.user_ids = test_.user_ids[idx]
    test_.item_ids = test_.item_ids[idx]
    test_.ratings = test_.ratings[idx]

    reliabilities = model.predict(test_.user_ids, test_.item_ids)[1]
    avg_reliability, std_reliability = reliabilities.mean(), reliabilities.std()

    test_ = test_.tocsr()
    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []
    rri = []

    for user_id, row in enumerate(test_):

        if not len(row.indices):
            continue

        predictions, reliabilities = model.predict(user_id)
        predictions *= -1

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = np.infty

        predictions = predictions.argsort()

        targets = row.indices

        user_precision, user_recall, user_rri = zip(*[
            _get_precision_recall_rri(predictions, reliabilities, avg_reliability, std_reliability, targets, x)
            for x in k
        ])

        precision.append(user_precision)
        recall.append(user_recall)
        rri.append(user_rri)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()
    rri = np.array(rri).squeeze()

    return precision, recall, rri


def classification(preds, error, test):

    splitter = KFold(n_splits=2, shuffle=True, random_state=0)
    targets = error > 1
    likelihood = 0
    auc = 0

    for train_index, test_index in splitter.split(test.ratings):
        mod = LogisticRegression().fit(preds[1][train_index].reshape(-1, 1), targets[train_index])
        probs = mod.predict_proba(preds[1][test_index].reshape(-1, 1))
        likelihood += np.log(probs[range(len(probs)), targets[test_index].astype(int)]).mean() / 2
        auc += roc_auc_score(targets[test_index], probs[:, 1]) / 2

    return likelihood, auc


def _get_precision_recall(predictions, targets, k):

    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))

    return float(num_hit) / len(predictions), float(num_hit) / len(targets)


def precision_recall_score(model, test, train=None, k=10):
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
    k: int or array of int,
        The maximum number of predicted items
    Returns
    -------
    (Precision@k, Recall@k): numpy array of shape (num_users, len(k))
        A tuple of Precisions@k and Recalls@k for each user in test.
        If k is a scalar, will return a tuple of vectors. If k is an
        array, will return a tuple of arrays, where each row corresponds
        to a user and each column corresponds to a value of k.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX

        predictions = predictions.argsort()

        targets = row.indices

        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])

        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()

    return precision, recall
