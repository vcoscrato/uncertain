import torch
import numpy as np
import scipy.sparse as sp
from uncertain.utils import gpu, cpu


def _index_or_none(tensor, shuffle_index):
    if tensor is None:
        return None
    else:
        return tensor[shuffle_index]


def _tensor_or_array(tensor):
    if tensor is None:
        return None
    if 'torch' in str(tensor.__class__):
        return tensor
    else:
        if tensor.dtype == np.int32:
            return torch.from_numpy(tensor.astype(np.int64))
        return torch.from_numpy(tensor)


class ExplicitInteractions(object):
    """
    For *explicit feedback* scenarios: user ids, item ids, and
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
    num_users: int, optional
        Number of distinct users in the dataset.
    num_items: int, optional
        Number of distinct items in the dataset.
    """

    def __init__(self, user_ids, item_ids, ratings,
                 num_users=None,
                 num_items=None):

        self.num_users = num_users or int(user_ids.max() + 1)
        self.num_items = num_items or int(item_ids.max() + 1)

        self.user_ids = _tensor_or_array(user_ids)
        self.item_ids = _tensor_or_array(item_ids)
        self.ratings = _tensor_or_array(ratings)

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

    def __getitem__(self, idx):

        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

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
        self.ratings = gpu(self.ratings, True)

    def cpu(self):
        """
        Move data to cpu.
        """

        self.user_ids = cpu(self.user_ids)
        self.item_ids = cpu(self.item_ids)
        self.ratings = cpu(self.ratings)

    def shuffle(self, seed=None):
        """
        Shuffle interaction data.
        """

        if seed is not None:
            torch.manual_seed(seed)

        shuffle_indices = torch.randperm(len(self), device=self.user_ids.device)
        
        self.user_ids = self.user_ids[shuffle_indices]
        self.item_ids = self.item_ids[shuffle_indices]
        self.ratings = _index_or_none(self.ratings, shuffle_indices)

    def get_user_support(self):

        count = torch.zeros(self.num_users, device=self.user_ids.device)
        for i in range(self.num_users):
            count[i] += (self.user_ids == i).sum()

        return count

    def get_item_support(self):

        count = torch.zeros(self.num_items, device=self.item_ids.device)
        for i in range(self.num_items):
            count[i] += (self.item_ids == i).sum()

        return count
    
    def get_item_variance(self):

        variances = torch.empty(self.num_items, device=self.item_ids.device)
        for i in range(self.num_items):
            variances[i] = self.ratings[self.item_ids == i].var()
        variances[torch.isnan(variances)] = 0

        return variances
