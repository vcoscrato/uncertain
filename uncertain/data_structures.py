import torch
import numpy as np
import scipy.sparse as sp


class Interactions(object):
    """
    Base data structure for Recommendation Systems.

    Parameters
    ----------
    users: tensor of ints
        The interactions user ids. Should be integers in [0, n_users].
    items: tensor
        The interactions item ids. Should be integers in [0, items].
    scores: tensor
        For explicit or ordinal feedback feedback: The interaction values. If None, Implicit feedback is assumed.
    timestamps: tensor
        The interactions timestamps.
    n_users: int, optional
        The number of users. If none, inferred from max(users)+1.
    n_items: int, optional
        The number of items. If none, inferred from max(items)+1.
    user_labels: tensor, optional
        Tensor of unique identifiers for the users in the dataset. user_labels[u] is the label of the u-th user.
    item_labels: tensor, optional
        Tensor of unique identifiers for the items in the dataset. item_labels[i] is the label of the i-th user
    score_labels: tensor
        For ordinal feedback only: The ordered score labels.
    device: torch.device
        The computing device in which data should be stored.
    """

    def __init__(self, users, items, scores=None, timestamps=None, num_users=None, num_items=None,
                 user_labels=None, item_labels=None, score_labels=None, device=torch.device('cpu')):

        self.device = device
        
        self.users = self.make_tensor(users).long()
        if num_users is None:
            self.num_users = torch.max(self.users).item() + 1
        else:
            assert num_users > torch.max(self.users), 'num_users should be > max(users)'
            self.num_users = num_users
        if user_labels is not None:
            self.user_labels = list(user_labels)
            
        self.items = self.make_tensor(items).long()
        assert len(self.items) == len(self.items), 'items and items should have same length'
        if num_items is None:
            self.num_items = torch.max(self.items).item() + 1
        else:
            assert num_items > torch.max(self.items), 'num_items should be > max(items)'
            self.num_items = num_items
        if item_labels is not None:
            self.item_labels = list(item_labels)

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

    def to_device(self, device):

        self.device = torch.device('cuda')
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                setattr(self, key, value.to(device))
        return self

    def pass_args(self):

        kwargs = {'num_users': self.num_users, 'num_items': self.num_items, 'device': self.device}
        if hasattr(self, 'user_labels'):
            kwargs['user_labels'] = self.user_labels
        if hasattr(self, 'item_labels'):
            kwargs['item_labels'] = self.item_labels
        if hasattr(self, 'score_labels'):
            kwargs['score_labels'] = self.score_labels

        return kwargs

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.users.cpu()
        col = self.items.cpu()
        data = self.scores.cpu() if self.type == 'Explicit' else torch.ones_like(col)

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def get_rated_items(self, user, threshold=None):

        if isinstance(a, str):
            user = self.user_labels.index(user)

        idx = self.users == user
        if threshold is None:
            return self.items[idx]
        else:
            return self.items[torch.logical_and(idx, self.scores >= threshold)]

    def get_negative_items(self, user):

        if isinstance(a, str):
            user = self.user_labels.index(user)

        rated_items = self.get_rated_items(user)
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
    user: int
        user identifier
    items: tensor
        A tensor containing the recommended item ids.
    user_label: str

    item_labels: tensor
        The labels of the recommended items.
    uncertainties: tensor
        The estimated uncertainty for each of the
        recommended items.

    """

    def __init__(self, user, items, user_label=None, item_labels=None, uncertainties=None):

        self.user = user
        self.items = items
        if user_label is None:
            self.user_label = user
        else:
            self.user_label = user_label
        if item_labels is None:
            self.item_labels = items
        else:
            self.item_labels = item_labels
        self.uncertainties = uncertainties

    def __repr__(self):

        s = 'Recommendation list for user {}: \n'.format(self.user_label)
        for i in range(len(self.items)):
            s += 'Item: {}'.format(self.item_labels[i])
            if self.uncertainties is not None:
                s += '; Uncertainty: {unc:1.2f}'.format(unc=self.uncertainties[i])
            s += '.\n'

        return s