import torch


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterion don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def funk_svd_loss(predicted_ratings, observed_ratings=None, predicted_negative=None):
    """
    Funk SVD loss - If explicit, uses regression
    loss, otherwise uses logistic loss.

    Parameters
    ----------
    predicted_ratings: tensor
        Tensor containing rating predictions.
    observed_ratings: tensor
        If explicit feedback: Tensor containing observed ratings.
    predicted_negative: tensor
        If implicit feedback: Tensor containing rating
        predictions for the sampled negative instances.

    Returns
    -------
    loss, float
        The mean value of the loss function.
    """

    if observed_ratings is not None:
        assert_no_grad(observed_ratings)
        return ((observed_ratings - predicted_ratings) ** 2).mean()
    else:
        positive = (1.0 - torch.sigmoid(predicted_ratings)).mean()
        negative = torch.sigmoid(predicted_negative).mean() if predicted_negative is not None else 0
        return (positive + negative) / 2


def cpmf_loss(predicted_ratings, observed_ratings=None, predicted_negative=None):
    """
    Gaussian loss for explicit and implicit feedback.

    Parameters
    ----------
    predicted_ratings: tensor
        Tensor containing rating predictions.
    observed_ratings: tensor
        If explicit feedback: Tensor containing observed ratings.
    predicted_negative: tensor
        If implicit feedback: Tensor containing rating
        predictions for the sampled negative instances.

    Returns
    -------
    loss, float
        The mean value of the loss function.
    """

    if observed_ratings is not None:
        assert_no_grad(observed_ratings)
        mean, variance = predicted_ratings
        return (((observed_ratings - mean) ** 2) / variance).mean() + torch.log(variance).mean()
    else:
        mean, variance = predicted_ratings
        positive = (((1.0 - mean) ** 2) / variance).mean() + torch.log(variance).mean()
        if predicted_negative is not None:
            mean, variance = predicted_negative
            negative = ((mean ** 2) / variance).mean() + torch.log(variance).mean()
        else:
            negative = 0
        return (positive + negative) / 2


def max_prob_loss(predicted_ratings, observed_ratings):
    """
    Maximum probability loss for ordinal data.

    Parameters
    ----------
    observed_ratings: tensor
        Tensor (n_obs) containing the observed ordinal rating labels.
    predicted_ratings: tensor
        Tensor (n_obs, n_rating_labels) containing the probabilities for each rating label.

        Returns
    -------
    loss: float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)
    return -predicted_ratings[range(len(-predicted_ratings)), observed_ratings].log().mean()

