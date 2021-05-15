import torch
import numpy as np
from uncertain.core import Interactions
from copy import deepcopy as dc


def random_train_test_split(interactions, test_percentage=0.2, seed=0):

    interactions.shuffle(seed)
    cutoff = int((1.0 - test_percentage) * len(interactions))

    u, i, s = interactions[:cutoff]
    train = Interactions(u, i, s, **interactions.pass_args())
    u, i, s = interactions[cutoff:]
    test = Interactions(u, i, s, **interactions.pass_args())

    return train, test


def user_based_split(interactions, test_percentage=0.2, min_profile_length=2, seed=None):

    train_idx = []
    test_idx = []

    if seed is not None:
        torch.manual_seed(seed)

    for u in range(interactions.num_users):

        idx = torch.where(interactions.users == u)[0]

        if len(idx) < min_profile_length:
            train_idx.append(idx)
            continue

        if hasattr(interactions, 'timestamps'):
            idx = idx[interactions.timestamps[idx].argsort()]
        else:
            idx = idx[torch.randperm(len(idx), device=idx.device)]

        cutoff = int((1.0 - test_percentage) * len(idx))

        train_idx.append(idx[:cutoff])
        test_idx.append(idx[cutoff:])

    u, i, s = interactions[torch.cat(train_idx)]
    train = Interactions(u, i, s, **interactions.pass_args())
    u, i, s = interactions[torch.cat(test_idx)]
    test = Interactions(u, i, s, **interactions.pass_args())

    return train, test
