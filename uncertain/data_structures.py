import torch
import numpy as np
import scipy.sparse as sp
from uncertain.utils import gpu, cpu


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
    data: tensor
        A tensor in which lines correspond to
        data instances in the format (user_id,
        item_id).
    ratings: tensor
        A tensor or array containing the ratings
        for each interaction. If None implicit
        signals are assumed.
    num_users: int
        Number of distinct users in the dataset.
    num_items: int
        Number of distinct items in the dataset.
    """

    def __init__(self, interactions, ratings=None, num_users=None, num_items=None):

        self.interactions = (interactions if torch.is_tensor(interactions) else torch.tensor(interactions).T).long()
        assert self.interactions.shape[1] == 2, "Interactions should be (n_instances, 2) shaped."

        if ratings is not None:
            self.ratings = ratings if torch.is_tensor(ratings) else torch.tensor(ratings)
            assert len(self.ratings) == self.interactions.shape[0], "Interaction and ratings should have same length."
        else:
            self.ratings = None

        self.num_users = num_users or int(self.interactions[:, 0].max() + 1)
        self.num_items = num_items or int(self.interactions[:, 1].max() + 1)

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

    def users(self):

        return self.interactions[:, 0]

    def items(self):

        return self.interactions[:, 1]

    def cuda(self):
        """
        Move data to gpu.
        """

        self.interactions = self.interactions.cuda()
        if self.ratings is not None:
            self.ratings = self.ratings.cuda()

        return self

    def cpu(self):
        """
        Move data to cpu.
        """

        self.interactions = self.interactions.cpu()
        if self.ratings is not None:
            self.ratings = self.ratings.cpu()

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

        shuffle_indices = torch.randperm(len(self), device=self.interactions.device)
        
        self.interactions = self.interactions[shuffle_indices]
        if self.type == 'Explicit':
            self.ratings = self.ratings[shuffle_indices]

    def get_user_support(self):

        count = torch.zeros(self.num_users, device=self.interactions.device)
        for i in range(self.num_users):
            count[i] += (self.interactions[:, 0] == i).sum()

        return count

    def get_item_support(self):

        count = torch.zeros(self.num_items, device=self.interactions.device)
        for i in range(self.num_items):
            count[i] += (self.interactions[:, 1] == i).sum()

        return count
    
    def get_item_variance(self):

        if self.type == 'Implicit':
            raise TypeError('Item variance not defined for implicit ratings.')

        variances = torch.empty(self.num_items, device=self.interactions.device)
        for i in range(self.num_items):
            variances[i] = self.ratings[self.interactions[:, 1] == i].float().var()
        variances[torch.isnan(variances)] = 0

        return variances


class Recommendations(object):
    """
    This object should be used for an easier and better
    visualization and evaluation of a recommendation list.

    Parameters
    ----------
    ID: int
        user identifier
    items: list
        A list containing the recommended item ids.
    uncertainties: list
        The estimated uncertainty for each of the
        recommended items.
    """

    def __init__(self, ID, items, uncertainties=None):

        self.ID = ID
        self.items = items
        self.uncertainties = uncertainties

    def __repr__(self):

        s = 'Recommendation list for user {}: \n'.format(self.ID)
        for i in range(len(self.items)):
            s += 'Item: {}'.format(self.items[i])
            if self.uncertainties is not None:
                s += '; Uncertainty: {unc:1.2f}'.format(unc=self.uncertainties[i])
            s += '.\n'

        return s
