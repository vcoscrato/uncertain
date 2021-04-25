import torch
import numpy as np
import scipy.sparse as sp


class Interactions(object):
    """
    For *explicit feedback* scenarios: interaction and
    ratings should be provided for all user-item-rating
    triplets that were observed in the dataset.

    Parameters
    ----------
    interactions: tensor or tuple
        A tensor in which lines correspond to
        interaction instances in the format
        (user_id, item_id). Or a tuple containing
        the user_ids and item_ids arrays or tensors.
    ratings: tensor or array
        A tensor or array containing the ratings
        for each interaction. If None implicit
        signals are assumed.
    user_labels: tensor, optional
        Tensor of unique identifiers for the
        users in the dataset. If passed, then
        the user interactions are not factorized.
        This can allow users with no known
        interactions.
    item_labels: tensor, optional
        Tensor of unique identifiers for the
        items in the dataset. If passed, then
        the item interactions are not factorized.
        This can allow items with no known
        interactions.

    Attributes
    ----------
    data: tensor
        A tensor in which lines correspond to
        data instances in the format (user_id,
        item_id).
    ratings: tensor
        A tensor or array containing the ratings
        for each interaction. If None implicit
        signals are assumed.
    user_labels: tensor
        Tensor of unique identifiers for the
        users in the dataset.
    item_labels: tensor
        Tensor of unique identifiers for the
        items in the dataset.
    """

    def __init__(self, interactions, ratings=None, timestamps=None, user_labels=None, item_labels=None):

        self.interactions = (interactions if torch.is_tensor(interactions) else torch.tensor(interactions).T).long()
        assert self.interactions.shape[1] == 2, "Interactions should be (n_instances, 2) shaped."

        if ratings is not None:
            self.ratings = ratings if torch.is_tensor(ratings) else torch.tensor(ratings)
            assert len(self.ratings) == len(self), "Interaction and ratings should have same length."
        else:
            self.ratings = None

        if timestamps is not None:
            self.timestamps = timestamps if torch.is_tensor(timestamps) else torch.tensor(timestamps)
            assert len(self.timestamps) == len(self), "Interaction and timestamps should have same length."
        else:
            self.timestamps = None

        if user_labels is None:
            self.user_labels, self.interactions[:, 0] = torch.unique(self.interactions[:, 0], return_inverse=True)
        else:
            self.user_labels = user_labels
        if item_labels is None:
            self.item_labels, self.interactions[:, 1] = torch.unique(self.interactions[:, 1], return_inverse=True)
        else:
            self.item_labels = item_labels

    @property
    def device(self):
        return self.interactions.device

    @property
    def is_cuda(self):
        return self.interactions.is_cuda

    @property
    def users(self):
        return self.interactions[:, 0]

    @property
    def items(self):
        return self.interactions[:, 1]

    @property
    def num_users(self):
        return len(self.user_labels)

    @property
    def num_items(self):
        return len(self.item_labels)

    @property
    def type(self):
        return 'Explicit' if self.ratings is not None else 'Implicit'

    def __len__(self):
        return self.interactions.shape[0]

    def __repr__(self):

        return ('<{type} interactions dataset ({num_users} users x {num_items} items '
                'x {num_interactions} interactions).>'
                .format(
                    type=self.type,
                    num_users=self.num_users,
                    num_items=self.num_items,
                    num_interactions=len(self)
                ))

    def __getitem__(self, idx):

        return self.interactions[idx], self.ratings[idx]

    def cuda(self):
        """
        Move data to gpu.
        """

        self.interactions = self.interactions.cuda()
        if self.ratings is not None:
            self.ratings = self.ratings.cuda()
        if self.timestamps is not None:
            self.timestamps = self.timestamps.cuda()

        return self

    def cpu(self):
        """
        Move data to cpu.
        """

        self.interactions = self.interactions.cpu()
        if self.ratings is not None:
            self.ratings = self.ratings.cpu()
        if self.timestamps is not None:
            self.timestamps = self.timestamps.cpu()

        return self

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.interactions[:, 0].cpu()
        col = self.interactions[:, 1].cpu()
        data = self.ratings.cpu() if self.type == 'Explicit' else torch.ones_like(col)

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def get_rated_items(self, user_id, threshold=None):

        idx = self.interactions[:, 0] == user_id
        if threshold is None:
            return self.interactions[idx, 1]
        else:
            return self.interactions[torch.logical_and(idx, self.ratings >= threshold), 1]

    def get_negative_items(self, user_id):

        rated_items = self.get_rated_items(user_id)
        negative_items = torch.tensor([i for i in range(self.num_items) if i not in rated_items],
                                      device=rated_items.device)
        return negative_items

    def shuffle(self, seed=None):
        """
        Shuffle interaction data.
        """

        if seed is not None:
            torch.manual_seed(seed)

        shuffle_indices = torch.randperm(len(self), device=self.device)
        
        self.interactions = self.interactions[shuffle_indices]
        if self.ratings is not None:
            self.ratings = self.ratings[shuffle_indices]
        if self.timestamps is not None:
            self.timestamps = self.timestamps[shuffle_indices]

    def get_user_profile_length(self):

        user_profile_length = torch.zeros(self.num_users, device=self.device)
        count = torch.bincount(self.users)
        user_profile_length[:len(count)] = count

        return user_profile_length

    def get_item_popularity(self):

        item_pop = torch.zeros(self.num_items, device=self.device)
        count = torch.bincount(self.items)
        item_pop[:len(count)] = count

        return item_pop
    
    def get_item_variance(self):

        if self.type == 'Implicit':
            raise TypeError('Item variance not defined for implicit ratings.')

        variances = torch.empty(self.num_items, device=self.device)
        for i in range(self.num_items):
            variances[i] = self.ratings[self.interactions[:, 1] == i].float().var()
        variances[torch.isnan(variances)] = 0

        return variances

    def filter_users(self, min_profile_length):

        profile_len = self.get_user_profile_length()
        idx = profile_len[self.users] >= min_profile_length

        self.interactions = self.interactions[idx]
        if self.ratings is not None:
            self.ratings = self.ratings[idx]
        if self.timestamps is not None:
            self.timestamps = self.timestamps[idx]

        return self

    def minibatch(self, batch_size):

        if self.type == 'Explicit':
            for i in range(0, len(self), batch_size):
                yield self.interactions[i:i + batch_size], self.ratings[i:i + batch_size]
        else:
            for i in range(0, len(self), batch_size):
                yield self.interactions[i:i + batch_size], None


class Recommendations(object):
    """
    This object should be used for an easier and better
    visualization and evaluation of a recommendation list.

    Parameters
    ----------
    ID: int
        user identifier
    items: tensor
        A tensor containing the recommended item ids.
    item_labels: tensor
        The labels of the recommended items.
    uncertainties: tensor
        The estimated uncertainty for each of the
        recommended items.

    """

    def __init__(self, ID, items, item_labels, uncertainties=None):

        self.ID = ID
        self.items = items
        self.item_labels = item_labels
        self.uncertainties = uncertainties

    def __repr__(self):

        s = 'Recommendation list for user {}: \n'.format(self.ID)
        for i in range(len(self.items)):
            s += 'Item: {}'.format(self.item_labels[i])
            if self.uncertainties is not None:
                s += '; Uncertainty: {unc:1.2f}'.format(unc=self.uncertainties[i])
            s += '.\n'

        return s
