import numpy as np
from copy import deepcopy


def graphs_score(model, test):

    predictions, reliabilities = model.predict(test.user_ids, test.item_ids)
    quantiles = np.quantile(reliabilities, np.linspace(start=0, stop=1, num=21, endpoint=True))
    rmse = np.empty(20)
    width = np.empty(20)
    for idx in range(20):
        ind = reliabilities > quantiles[idx]
        rmse[idx] = np.sqrt(np.square(predictions[ind] - test.ratings[ind]).mean())
        ind = np.bitwise_and(ind, reliabilities < quantiles[idx+1])
        errors = np.abs(predictions[ind] - test.ratings[ind])
        width[idx] = np.quantile(errors, 0.95)

    return rmse, width


def rpi_score(model, test):

    predictions, reliabilities = model.predict(test.user_ids, test.item_ids)
    errors = np.abs(predictions - test.ratings)
    MAE = errors.mean()
    error_deviations = errors - MAE
    reliability_deviations = reliabilities.mean() - reliabilities
    errors_std = np.abs(error_deviations).mean()
    reliabilities_std = np.abs(reliability_deviations).mean()
    RPI = (errors*error_deviations*reliability_deviations).mean()/(errors_std*reliabilities_std*MAE)

    return RPI


def rmse_rpi_score(model, test):

    predictions, reliabilities = model.predict(test.user_ids, test.item_ids)
    errors = np.abs(predictions - test.ratings)
    mae = errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    error_deviations = errors - mae
    reliability_deviations = reliabilities.mean() - reliabilities
    errors_std = np.abs(error_deviations).mean()
    reliabilities_std = np.abs(reliability_deviations).mean()
    rpi = (errors * error_deviations * reliability_deviations).mean() / (errors_std * reliabilities_std * mae)

    return rmse, rpi


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

    reliabilities = model.predict(test.user_ids, test.item_ids)[1]
    avg_reliability, std_reliability = reliabilities.mean(), reliabilities.std()

    idx = test.ratings >= 4
    test_ = deepcopy(test)
    test_.user_ids = test_.user_ids[idx]
    test_.item_ids = test_.item_ids[idx]
    test_.ratings = test_.ratings[idx]
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
