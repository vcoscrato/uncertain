import torch
from uncertain.utils import assert_no_grad


def regression_loss(observed_ratings, predicted_ratings):
    """
    Regression loss.

    Parameters
    ----------
    observed_ratings: tensor
        Tensor containing observed ratings.
    predicted_ratings: tensor
        Tensor containing rating predictions.

    Returns
    -------
    loss, float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    return ((observed_ratings - predicted_ratings) ** 2).mean()


def gaussian_loss(observed_ratings, predicted_ratings):
    """
    Gaussian loss.

    Parameters
    ----------
    observed_ratings: tensor
        Tensor (n_obs) containing the observed ratings.
    predicted_ratings: tuple
        A tuple containing 2 tensors: the estimated averages (n_obs) and variances (n_obs).

        Returns
    -------
    loss: float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    mean, variance = predicted_ratings

    return (((observed_ratings - mean) ** 2) / variance).mean() + torch.log(variance).mean()


def max_prob_loss(observed_ratings, predicted_ratings):
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

    return -predicted_ratings[range(len(-predicted_ratings)), observed_ratings].mean()


def pointwise_loss(positive_predictions, negative_predictions, mask=None):
    """
    Logistic loss function.
    Parameters
    ----------
    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.
    Returns
    -------
    loss, float
        The mean value of the loss function.
    """

    positives_loss = (1.0 - torch.sigmoid(positive_predictions))
    negatives_loss = torch.sigmoid(negative_predictions)

    loss = (positives_loss + negatives_loss)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()
