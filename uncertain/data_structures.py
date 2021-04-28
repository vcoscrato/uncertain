import torch
import numpy as np
import scipy.sparse as sp


class Interactions(object):
    """
    Base data structure for Recommendation Systems.

    Parameters
    ----------
    users: tensor
        The interactions user ids.
    items: tensor
        The interactions item ids.
    scores: tensor
        For explicit or feedback feedback: The interaction values. If None, Implicit feedback is assumed.
    timestamps: tensor
        The interactions timestamps.
    user_labels: tensor, optional
        Tensor of unique identifiers for the users in the dataset. If passed, then the interactions users are not
        factorized (this can allow users with no known interactions).
    item_labels: tensor, optional
        Tensor of unique identifiers for the items in the dataset. If passed, then the interaction items are not
        factorized (this can allow items with no known interactions).
    score_labels: tensor
        For ordinal feedback only: The ordered score labels.
    device: torch.device
        The machine device in which data is stored.
    """

    def __init__(self, users, items, scores=None, timestamps=None,
                 user_labels=None, item_labels=None, score_labels=None, device=torch.device('cpu')):

        self.device = device
        
        self.users = self.make_tensor(users).long()
        self.user_labels = user_labels
        if self.user_labels is None:
            self.user_labels, self.users = torch.unique(self.users, return_inverse=True)
            
        self.items = self.make_tensor(items).long()
        self.item_labels = item_labels
        if self.item_labels is None:
            self.item_labels, self.items = torch.unique(self.items, return_inverse=True)
        assert len(self.items) == len(self.items), 'items and items should have same length'

        if scores is not None:
            assert len(self.users) == len(scores), 'users, items and scores should have same length'
            if score_labels is not None:
                self.scores = self.make_tensor(scores).long()
                self.score_labels = score_labels
            else:
                self.scores = self.make_tensor(scores).float()

        if timestamps is not None:
            assert len(self.users) == len(timestamps), 'users, items and timestamps should have same length'
            self.timestamps = self.make_tensor(timestamps)

    def make_tensor(self, x):

        if torch.is_tensor(x):
            return x.to(self.device)
        else:
            return torch.tensor(x).to(self.device)

    @property
    def num_users(self):
        return len(self.user_labels)

    @property
    def num_items(self):
        return len(self.item_labels)

    @property
    def type(self):
        if not hasattr(self, 'scores'):
            return 'Implicit'
        if not hasattr(self, 'score_labels'):
            return 'Explicit'
        else:
            return 'Ordinal'

    def __len__(self):
        return len(self.users)

    def __repr__(self):
        return ('<{type} interactions ({num_users} users x {num_items} items x {num_interactions} interactions)>'
                .format(type=self.type, num_users=self.num_users, num_items=self.num_items, num_interactions=len(self)))

    def __getitem__(self, idx):
        if hasattr(self, 'scores'):
            return self.users[idx], self.items[idx], self.scores[idx]
        else:
            return self.users[idx], self.items[idx], None

    def minibatch(self, batch_size):

        for i in range(0, len(self), batch_size):
            yield self[i:i + batch_size]

    def cuda(self):

        for key in self.__dict__:
            attr = getattr(self, key)
            if torch.is_tensor(attr):
                attr = attr.to('cuda')
        return self

    def cpu(self):

        for key in self.__dict__:
            attr = getattr(self, key)
            if torch.is_tensor(attr):
                attr = attr.to('cpu')
        return self

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.users.cpu()
        col = self.items.cpu()
        data = self.scores.cpu() if self.type == 'Explicit' else torch.ones_like(col)

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def get_rated_items(self, user_id, threshold=None):

        idx = self.users == user_id
        if threshold is None:
            return self.items[idx]
        else:
            return self.items[torch.logical_and(idx, self.scores >= threshold)]

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

        self.users = self.users[shuffle_indices]
        self.items = self.items[shuffle_indices]
        if hasattr(self, 'scores'):
            self.scores = self.scores[shuffle_indices]
        if hasattr(self, 'timestamps'):
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
            raise TypeError('Item variance not defined for implicit scores.')

        variances = torch.empty(self.num_items, device=self.device)
        for i in range(self.num_items):
            variances[i] = self.scores[self.items == i].float().var()
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