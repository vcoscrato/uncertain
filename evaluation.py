import torch
from uncertain.metrics import rmse_score, recommendation_score, correlation, rpi_score, classification


def evaluate(model, test, train, uncertainty):

    out = {}
    est = model.predict(test.user_ids, test.item_ids)
    if type(est) is tuple:
        est, unc = est
    p, r, a, s = recommendation_score(model, test, train, max_k=10)

    out['RMSE'] = rmse_score(est, test.ratings)
    out['Precision'] = p.mean(axis=0)
    out['Recall'] = r.mean(axis=0)

    if uncertainty:
        error = torch.abs(test.ratings - est)
        quantiles = torch.quantile(unc, torch.linspace(0, 1, 21, device=unc.device, dtype=unc.dtype))
        out['Quantile RMSE'] = torch.zeros(20)
        for idx in range(20):
            ind = torch.bitwise_and(quantiles[idx] <= unc, unc < quantiles[idx + 1])
            out['Quantile RMSE'][idx] = torch.sqrt(torch.square(error[ind]).mean())
        quantiles = torch.quantile(a, torch.linspace(0, 1, 21, device=a.device, dtype=a.dtype))
        out['Quantile MAP'] = torch.zeros(20)
        for idx in range(20):
            ind = torch.bitwise_and(quantiles[idx] <= a, a < quantiles[idx + 1])
            out['Quantile MAP'][idx] = p[ind, -1].mean()
        out['RRI'] = s.nansum(0) / (~s.isnan()).float().sum(0)
        out['Correlation'] = correlation(error, unc)
        out['RPI'] = rpi_score(error, unc)
        out['Classification'] = classification(error, unc)

    return out
