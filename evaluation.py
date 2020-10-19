from torch import abs
from uncertain.metrics import rmse_score, precision_recall_score, correlation, rpi_score, classification, quantiles


def evaluate(model, test, train, accuracy, uncertainty):

    out = {}
    est = model.predict(test.user_ids, test.item_ids)
    if type(est) is tuple:
        est, unc = est
    if accuracy:
        p, r = precision_recall_score(model, test, train, max_k=10)
        out['RMSE'] = rmse_score(est, test.ratings)
        out['Precision'] = p.mean(axis=0)
        out['Recall'] = r.mean(axis=0)

    if uncertainty:
        error = abs(test.ratings - est)
        out['Correlation'] = correlation(error, unc)
        out['Quantiles'] = quantiles(error, unc)
        out['RPI'] = rpi_score(error, unc)
        out['Classification'] = classification(error, unc)

    return out
