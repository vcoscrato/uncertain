import numpy as np


def rpi(model, test):
    predictions, reliabilities = model.predict(test.user_ids, test.item_ids)
    errors = np.abs(predictions - test.ratings)
    MAE = errors.mean()
    error_deviations = errors - MAE
    reliability_deviations = reliabilities.mean() - reliabilities
    errors_std = np.abs(error_deviations).mean()
    reliabilities_std = np.abs(reliability_deviations).mean()
    RPI = (errors*error_deviations*reliability_deviations).mean()/(errors_std*reliabilities_std*MAE)
    return RPI


def rmse_rpi_wrapper(model, test):
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


def _get_precision_recall_rri(predictions, reliabilities, targets, k):
    predictions = predictions[:k]
    reliabilities = reliabilities[:k]
    avg_rel = reliabilities.mean()
    std_rel = reliabilities.std()
    hit = list(set(predictions).intersection(set(targets)))
    num_hit = float(len(hit))
    if num_hit == 0:
        return 0, 0, np.nan
    reliability_relevants = reliabilities[np.isin(predictions, hit)]
    precision = float(num_hit) / len(predictions)
    recall = float(num_hit) / len(targets)
    if std_rel == 0:
        return predictions, recall, 0
    rri = ((reliability_relevants - avg_rel).sum()/std_rel) / num_hit
    return precision, recall, rri


def precision_recall_rri(model, test, train=None, k=10):

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []
    rri = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions, reliabilities = model.predict(user_id)
        predictions *= -1

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = np.infty

        predictions = predictions.argsort()
        reliabilities = reliabilities[predictions]
        targets = row.indices[row.data >= 4]

        user_precision, user_recall, user_rpi = zip(*[
            _get_precision_recall_rri(predictions, reliabilities, targets, x)
            for x in k
        ])

        precision.append(user_precision)
        recall.append(user_recall)
        rri.append(user_rpi)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()
    rri = np.array(rri).squeeze()

    return precision, recall, rri