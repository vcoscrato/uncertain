import torch
from uncertain.metrics import rmse_score, recommendation_score, correlation, rpi_score, classification
from uncertain.utils import minibatch


class LinearUncertaintyEstimator(object):
    """
    Basic uncertainty estimator that uses the
    sum of static user and/or item coefficients.


    Parameters
    ----------
    user_uncertainty: tensor
        A tensor containing the uncertainty coefficient for each user.
    item_uncertainty: tensor
        A tensor containing the uncertainty coefficient for each user.
    """

    def __init__(self,
                 user_uncertainty,
                 item_uncertainty):

        self.user = user_uncertainty
        self.item = item_uncertainty

    def predict(self, user_ids, item_ids):

        user_uncertainty = self.user[user_ids] if self.user is not None else 0
        item_uncertainty = self.item[item_ids] if self.item is not None else 0

        return user_uncertainty + item_uncertainty


class UncertainWrapper(object):
    """
    Wraps a rating estimator with an uncertainty estimator.

    Parameters
    ----------
    ratings: :class:`uncertain.models.BaseRecommender`
        A rating estimator.
    uncertainty: :class:`uncertain.UncertaintyWrapper.LinearUncertaintyEstimator
        An uncertainty estimator: A class containing a predict
        function that returns an uncertainty estimate for the
        given user, item pairs.
    """

    def __init__(self,
                 ratings,
                 uncertainty):

        self.ratings = ratings
        self.uncertainty = uncertainty

    @property
    def is_uncertain(self):
        return True

    def predict(self, user_ids, item_ids=None):

        user_ids, item_ids = self.ratings._predict_process_ids(user_ids, item_ids)

        ratings = self.ratings.predict(user_ids, item_ids)
        uncertainty = self.uncertainty.predict(user_ids, item_ids)

        return ratings, uncertainty

    def recommend(self, user_id, train=None, top=10):

        predictions = self.predict(user_id)

        if not self.is_uncertain:
            predictions = -predictions
            uncertainties = None
        else:
            uncertainties = predictions[1]
            predictions = -predictions[0]

        if train is not None:
            rated = train.item_ids[train.user_ids == user_id]
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
            for u, i, _ in loader:
                predictions = self.predict(u, i)
                est.append(predictions[0])
                unc.append(predictions[1])
            unc = torch.hstack(unc)
        else:
            for u, i, _ in loader:
                est.append(self.predict(u, i))
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