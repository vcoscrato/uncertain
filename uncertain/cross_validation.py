"""
Module with functionality for splitting and shuffling datasets.
"""

import torch
import numpy as np
from uncertain.data_structures import Interactions
from copy import deepcopy as dc


def random_train_test_split(data, test_percentage=0.2, random_state=None):
    """
    Randomly split interactions between training and testing.

    Parameters
    ----------

    data: :class:`uncertain.data_structures.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`uncertain.interactions.Interactions`,
                    :class:`uncertain.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    user_labels = data.user_labels
    item_labels = data.item_labels
    data.shuffle(random_state)

    cutoff = int((1.0 - test_percentage) * len(data))

    train_interactions = data.interactions[:cutoff]
    test_interactions = data.interactions[cutoff:]

    train_ratings = None if data.ratings is None else data.ratings[:cutoff]
    test_ratings = None if data.ratings is None else data.ratings[cutoff:]

    train = Interactions(train_interactions, train_ratings, None, user_labels, item_labels)
    test = Interactions(test_interactions, test_ratings, None, user_labels, item_labels)

    return train, test


def user_based_split(data, test_percentage=0.2, seed=None):
    """
    Split interactions between training and testing, guarantee
    that each user have a 'test_percentage' fraction of ratings
    in the test set. If timestamps are provided, the latest
    ratings are placed in the test set.

    Parameters
    ----------

    data: :class:`uncertain.data_structures.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    seed: int
        Seed to pass to RNG. Ignored if timestamps are provided.

    Returns
    -------

    (train, test): (:class:`uncertain.interactions.Interactions`,
                    :class:`uncertain.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    train_idx = []
    test_idx = []

    if seed is not None:
        torch.manual_seed(seed)

    for u in range(data.num_users):

        idx = torch.where(data.users == u)[0]

        if data.timestamps is not None:
            idx = idx[data.timestamps[idx].argsort()]
        else:
            idx = idx[torch.randperm(len(idx), device=idx.device)]

        cutoff = int((1.0 - test_percentage) * len(idx))

        train_idx.append(idx[:cutoff])
        test_idx.append(idx[cutoff:])

    train = data[torch.cat(train_idx)]
    train = Interactions(interactions=train[0], ratings=train[1],
                         user_labels=data.user_labels, item_labels=data.item_labels)
    test = data[torch.cat(test_idx)]
    test = Interactions(interactions=test[0], ratings=test[1],
                        user_labels=data.user_labels, item_labels=data.item_labels)

    return train, test
