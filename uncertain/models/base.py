import torch
from uncertain.utils import minibatch
from uncertain.metrics import rmse_score, recommendation_score, correlation, rpi_score, classification


class BaseRecommender(object):

    def __init__(self, num_users=None, num_items=None, num_ratings=None, desc=None):
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_ratings = num_ratings
        self._desc = desc
        
    def recommend(self, user_id, train=None, top=10):

        predictions = self.predict(user_id)

        if not self.is_uncertain:
            predictions = -predictions
            uncertainties = None
        else:
            uncertainties = predictions[1]
            predictions = -predictions[0]

        if train is not None:
            rated = train.interactions[:, 1][train.interactions[:, 0] == user_id]
            predictions[rated] = float('inf')

        idx = predictions.argsort()
        predictions = idx[:top]
        if self.is_uncertain:
            uncertainties = uncertainties[idx][:top]

        return predictions, uncertainties

    def evaluate(self, test, train):

        out = {}
        loader = minibatch(test, batch_size=int(1e5))
        est = []
        if self.is_uncertain:
            unc = []
            for interactions, _ in loader:
                predictions = self.predict(interactions[:, 0], interactions[:, 1])
                est.append(predictions[0])
                unc.append(predictions[1])
            unc = torch.hstack(unc)
        else:
            for interactions, _ in loader:
                est.append(self.predict(interactions[:, 0], interactions[:, 1]))
        est = torch.hstack(est)

        p, r, a, s = recommendation_score(self, test, train, max_k=10)

        out['RMSE'] = rmse_score(est, test.ratings)
        out['Precision'] = p.mean(axis=0)
        out['Recall'] = r.mean(axis=0)

        if self.is_uncertain:
            error = torch.abs(test.ratings - est)
            idx = torch.randperm(len(unc))[:int(1e5)]
            quantiles = torch.quantile(unc[idx], torch.linspace(0, 1, 21, device=unc.device, dtype=unc.dtype))
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
