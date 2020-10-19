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

        return (user_uncertainty + item_uncertainty)


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

    def predict(self, user_ids, item_ids=None):

        user_ids, item_ids = self.ratings._predict_process_ids(user_ids, item_ids)

        ratings = self.ratings.predict(user_ids, item_ids)
        uncertainty = self.uncertainty.predict(user_ids, item_ids)

        return ratings, uncertainty
