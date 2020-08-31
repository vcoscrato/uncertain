import numpy as np
from copy import deepcopy


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
        ind = np.bitwise_and(quantiles[idx-1] < epsilons, epsilons < quantiles[idx])
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
