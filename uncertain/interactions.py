import numpy as np
import scipy.sparse as sp
from uncertain.utils import gpu, cpu
from torch import from_numpy, randperm, manual_seed, empty, zeros, isnan


def _index_or_none(tensor, shuffle_index):
    if tensor is None:
        return None
    else:
        return tensor[shuffle_index]


def _func_or_none(tensor, func):
    if tensor is None:
        return None
    else:
        return func(tensor)


def _tensor_or_array(tensor):
    if tensor is None:
        return None
    if 'torch' in str(tensor.__class__):
        return tensor
    else:
        if tensor.dtype == np.int32:
            return from_numpy(tensor.astype(np.int64))
        return from_numpy(tensor)


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions, but can also be enriched with ratings, timestamps,
    and interaction weights.
    For *implicit feedback* scenarios, user ids and item ids should
    only be provided for user-item pairs where an interaction was
    observed. All pairs that are not provided are treated as missing
    observations, and often interpreted as (implicit) negative
    signals.
    For *explicit feedback* scenarios, user ids, item ids, and
    ratings should be provided for all user-item-rating triplets
    that were observed in the dataset.

    Parameters
    ----------
    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
        Must be larger than the maximum user id
        in user_ids.
    num_items: int, optional
        Number of distinct items in the dataset.
        Must be larger than the maximum item id
        in item_ids.

    Attributes
    ----------
    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
    num_items: int, optional
        Number of distinct items in the dataset.
    """

    def __init__(self, user_ids, item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 num_users=None,
                 num_items=None):

        self.num_users = num_users or int(user_ids.max() + 1)
        self.num_items = num_items or int(item_ids.max() + 1)

        self.user_ids = _tensor_or_array(user_ids)
        self.item_ids = _tensor_or_array(item_ids)
        self.ratings = _tensor_or_array(ratings)
        self.timestamps = _tensor_or_array(timestamps)
        self.weights = _tensor_or_array(weights)
        self._check()

    def __repr__(self):

        return ('<Interactions dataset ({num_users} users x {num_items} items '
                'x {num_interactions} interactions)>'
                .format(
                    num_users=self.num_users,
                    num_items=self.num_items,
                    num_interactions=len(self)
                ))

    def __len__(self):

        return len(self.user_ids)

    def _check(self):

        if self.user_ids.max() >= self.num_users:
            raise ValueError('Maximum user id greater '
                             'than declared number of users.')
        if self.item_ids.max() >= self.num_items:
            raise ValueError('Maximum item id greater '
                             'than declared number of items.')

        num_interactions = len(self.user_ids)

        for name, value in (('item IDs', self.item_ids),
                            ('ratings', self.ratings),
                            ('timestamps', self.timestamps),
                            ('weights', self.weights)):

            if value is None:
                continue

            if len(value) != num_interactions:
                raise ValueError('Invalid {} dimensions: length '
                                 'must be equal to number of interactions'
                                 .format(name))

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = cpu(self.user_ids)
        col = cpu(self.item_ids)
        data = cpu(self.ratings if self.ratings is not None else np.ones(len(self)))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def gpu(self):
        """
        Move data to gpu.
        """

        self.user_ids = gpu(self.user_ids, True)
        self.item_ids = gpu(self.item_ids, True)
        self.ratings = _func_or_none(self.ratings, lambda x: gpu(x, use_cuda=True))
        self.timestamps = _func_or_none(self.timestamps, lambda x: gpu(x, use_cuda=True))
        self.weights = _func_or_none(self.timestamps, lambda x: gpu(x, use_cuda=True))

    def cpu(self):
        """
        Move data to cpu.
        """

        self.user_ids = cpu(self.user_ids)
        self.item_ids = cpu(self.item_ids)
        self.ratings = _func_or_none(self.ratings, cpu)
        self.timestamps = _func_or_none(self.timestamps, cpu)
        self.weights = _func_or_none(self.timestamps, cpu)

    def shuffle(self, seed):
        """
        Shuffle interaction data.
        """

        manual_seed(seed)
        shuffle_indices = randperm(len(self), device=self.user_ids.device)
        
        self.user_ids = self.user_ids[shuffle_indices]
        self.item_ids = self.item_ids[shuffle_indices]
        self.ratings = _index_or_none(self.ratings, shuffle_indices)
        self.timestamps = _index_or_none(self.timestamps, shuffle_indices)
        self.weights = _index_or_none(self.weights, shuffle_indices)

    def get_user_support(self):

        count = zeros(self.num_users, device=self.user_ids.device)
        for i in range(self.num_users):
            count[i] += (self.user_ids == i).sum()

        return count

    def get_item_support(self):

        count = zeros(self.num_items, device=self.item_ids.device)
        for i in range(self.num_items):
            count[i] += (self.item_ids == i).sum()

        return count
    
    def get_item_variance(self):

        variances = empty(self.num_items, device=self.item_ids.device)
        for i in range(self.num_items):
            variances[i] = self.ratings[self.item_ids == i].var()
        variances[isnan(variances)] = 0

        return variances
