import torch
import scipy.sparse as sp


def make_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)


class Interactions(torch.utils.data.Dataset):

    def __init__(self, users, items, scores=None, timestamps=None, num_users=None, num_items=None,
                 user_labels=None, item_labels=None, score_labels=None):

        super().__init__()
        self.users = make_tensor(users).long()
        self.num_users = num_users or torch.max(self.users).item() + 1
        if user_labels is not None:
            self.user_labels = list(user_labels)

        self.items = make_tensor(items).long()
        assert len(self.items) == len(self.items), 'items and items should have same length'
        self.num_items = num_items or torch.max(self.items).item() + 1
        if item_labels is not None:
            self.item_labels = list(item_labels)

        if scores is not None:
            if score_labels is not None:
                self.scores = make_tensor(scores).long()
                self.score_labels = score_labels
            else:
                self.scores = make_tensor(scores).float()
            assert len(self.users) == len(scores), 'users, items and scores should have same length'

        if timestamps is not None:
            assert len(self.users) == len(timestamps), 'users, items and timestamps should have same length'
            self.timestamps = make_tensor(timestamps)

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
            return self.users[idx], self.items[idx]

    def shuffle(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        shuffle_indices = torch.randperm(len(self))
        self.users = self.users[shuffle_indices]
        self.items = self.items[shuffle_indices]
        if hasattr(self, 'scores'):
            self.scores = self.scores[shuffle_indices]
        if hasattr(self, 'timestamps'):
            self.timestamps = self.timestamps[shuffle_indices]

    def pass_args(self):
        kwargs = {'num_users': self.num_users, 'num_items': self.num_items}
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
        row = self.users
        col = self.items
        data = self.scores.cpu() if self.type == 'Explicit' else torch.ones_like(col)
        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def get_rated_items(self, user, threshold=None):
        if isinstance(user, str):
            user = self.user_labels.index(user)
        idx = self.users == user
        if threshold is None:
            return self.items[idx]
        else:
            return self.items[torch.logical_and(idx, self.scores >= threshold)]

    def get_negative_items(self, user):
        if isinstance(user, str):
            user = self.user_labels.index(user)

        rated_items = self.get_rated_items(user)
        negative_items = torch.tensor([i for i in range(self.num_items) if i not in rated_items])
        return negative_items

    def get_user_profile_length(self):
        user_profile_length = torch.zeros(self.num_users)
        count = torch.bincount(self.users)
        user_profile_length[:len(count)] = count
        return user_profile_length

    def get_item_popularity(self):
        item_pop = torch.zeros(self.num_items)
        count = torch.bincount(self.items)
        item_pop[:len(count)] = count
        return item_pop

    def get_item_variance(self):
        if self.type == 'Implicit':
            raise TypeError('Item variance not defined for implicit scores.')
        variances = torch.empty(self.num_items)
        for i in range(self.num_items):
            variances[i] = self.scores[self.items == i].float().var()
        variances[torch.isnan(variances)] = 0
        return variances

    def dataloader(self, batch_size):
        return torch.utils.data.DataLoader(self, batch_size=batch_size,
                                           drop_last=True, shuffle=True, num_workers=torch.get_num_threads())

    def split(self, validation_percentage, test_percentage, min_profile_length=0, seed=0):
        train_idx = []
        val_idx = []
        test_idx = []
        if seed is not None:
            torch.manual_seed(seed)
        for u in range(self.num_users):
            idx = torch.where(self.users == u)[0]
            if len(idx) < min_profile_length:
                train_idx.append(idx)
                continue
            if hasattr(self, 'timestamps'):
                idx = idx[self.timestamps[idx].argsort()]
            else:
                idx = idx[torch.randperm(len(idx))]
            cut_train = int((1.0 - validation_percentage - test_percentage) * len(idx))
            cut_val = int((1.0 - test_percentage) * len(idx))
            train_idx.append(idx[:cut_train])
            test_idx.append(idx[cut_train:cut_val])
            val_idx.append(idx[cut_val:])
        train = Interactions(*self[torch.cat(train_idx)], **self.pass_args())
        validation = Interactions(*self[torch.cat(val_idx)], **self.pass_args())
        test = Interactions(*self[torch.cat(test_idx)], **self.pass_args())
        return train, validation, test


class Recommendations(object):

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


class Recommender(object):

    def pass_args(self, interactions):
        for key, value in interactions.pass_args().items():
            setattr(self, key, value)

    def recommend(self, user, remove_items=None, top=10):

        if isinstance(user, str):
            user = self.user_labels.index(user)

        predictions = self.predict(user)

        if not self.is_uncertain:
            predictions = predictions
        else:
            uncertainties = predictions[1]
            predictions = predictions[0]

        if remove_items is not None:
            predictions[remove_items] = -float('inf')
            ranking = predictions.argsort(descending=True)[:-len(remove_items)][:top]
        else:
            ranking = predictions.argsort(descending=True)[:top]

        kwargs = {'user': user, 'items': ranking}
        if self.is_uncertain:
            kwargs['uncertainties'] = uncertainties[ranking]

        if hasattr(self, 'user_labels'):
            kwargs['user_label'] = self.user_labels[user]
        if hasattr(self, 'item_labels'):
            kwargs['item_labels'] = [self.item_labels[i] for i in ranking.cpu().tolist()]

        return Recommendations(**kwargs)

    def test_ratings(self, test_interactions):
        with torch.no_grad():
            out = {}
            predictions = self.forward(test_interactions.users, test_interactions.items)
            out['loss'] = self.loss_func(predictions, test_interactions.scores).item()
            if not self.is_uncertain:
                out['RMSE'] = rmse_score(predictions, test_interactions.scores)
            else:
                out['RMSE'] = rmse_score(predictions[0], test_interactions.scores)
                errors = torch.abs(test_interactions.scores - predictions[0])
                out['RPI'] = rpi_score(errors, predictions[1])
                out['Classification'] = classification(errors, predictions[1])
                out['Correlation'] = correlation(errors, predictions[1])
                out['Quantile RMSE'] = quantile_score(errors, predictions[1])
            return out

    def test_recommendations(self, test_interactions, train_interactions, max_k=10, relevance_threshold=None):
        out = {}
        precision = []
        recall = []
        ndcg_ = []
        rri = []
        precision_denom = torch.arange(1, max_k + 1, dtype=torch.float64)
        ndcg_denom = torch.log2(precision_denom + 1)
        for user in range(test_interactions.num_users):
            targets = test_interactions.get_rated_items(user, threshold=relevance_threshold)
            if not len(targets):
                continue
            rec = self.recommend(user, train_interactions.get_rated_items(user))
            hits = get_hits(rec, targets)
            num_hit = hits.cumsum(0)
            precision.append(num_hit / precision_denom)
            recall.append(num_hit / len(targets))
            ndcg_.append(ndcg(hits, ndcg_denom))
            if self.is_uncertain and hits.sum().item() > 0:
                with torch.no_grad():
                    rri_ = torch.empty(max_k - 1)
                    for i in range(2, max_k + 1):
                        unc = rec.uncertainties[:i]
                        rri_[i - 2] = (unc.mean() - unc[hits[:i]]).mean() / unc.std()
                    rri.append(rri_)
        out['Precision'] = torch.vstack(precision).mean(0)
        out['Recall'] = torch.vstack(recall).mean(0)
        out['NDCG'] = torch.vstack(ndcg_).mean(0)
        if len(rri) > 0:
            rri = torch.vstack(rri)
            out['RRI'] = rri.nansum(0) / (~rri.isnan()).float().sum(0)
        return out
