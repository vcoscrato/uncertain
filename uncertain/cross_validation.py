"""
Module with functionality for splitting and shuffling datasets.
"""

import torch
import numpy as np
from uncertain.interactions import ExplicitInteractions
from copy import deepcopy as dc


def random_train_test_split(interactions,
                            test_percentage=0.2,
                            random_state=None):
    """
    Randomly split interactions between training and testing.

    Parameters
    ----------

    interactions: :class:`uncertain.interactions.Interactions`
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

    n_users = interactions.num_users
    n_items = interactions.num_items
    interactions.shuffle(random_state)

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train = interactions[:cutoff]
    train = ExplicitInteractions(train[0], train[1], train[2],
                                 n_users, n_items)
    test = interactions[cutoff:]
    test = ExplicitInteractions(test[0], test[1], test[2],
                                n_users, n_items)

    return train, test


def user_based_split(interactions, n_validation, n_test):

    train_user_ids = []
    train_item_ids = []
    train_ratings = []

    if n_validation > 0:
        validation_user_ids = []
        validation_item_ids = []
        validation_ratings = []
    
    test_user_ids = []
    test_item_ids = []
    test_ratings = []

    for u in range(1, interactions.num_users):
        idx = interactions.user_ids == u
        if idx.sum() <= n_validation + n_test:
            continue
        user_ids = interactions.user_ids[idx]
        item_ids = interactions.item_ids[idx]
        ratings = interactions.ratings[idx]

        test_user_ids.append(user_ids[-n_test:])
        test_item_ids.append(item_ids[-n_test:])
        test_ratings.append(ratings[-n_test:])

        if n_validation > 0:
            validation_user_ids.append(user_ids[-(n_test + n_validation):-n_test])
            validation_item_ids.append(item_ids[-(n_test+n_validation):-n_test])
            validation_ratings.append(ratings[-(n_test + n_validation):-n_test])

        train_user_ids.append(user_ids[:-(n_test + n_validation)])
        train_item_ids.append(item_ids[:-(n_test+n_validation)])
        train_ratings.append(ratings[:-(n_test+n_validation)])

    train = ExplicitInteractions(torch.cat(train_user_ids).long(),
                         torch.cat(train_item_ids).long(),
                         torch.cat(train_ratings), num_items=interactions.num_items)

    test = ExplicitInteractions(torch.cat(test_user_ids).long(),
                        torch.cat(test_item_ids).long(),
                        torch.cat(test_ratings), num_items=interactions.num_items)

    if n_validation > 0:
        validation = ExplicitInteractions(torch.cat(validation_user_ids).long(),
                                  torch.cat(validation_item_ids).long(),
                                  torch.cat(validation_ratings), num_items=interactions.num_items)

        return train, validation, test

    return train, test