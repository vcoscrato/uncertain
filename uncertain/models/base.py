import torch
import numpy as np
from uncertain.data_structures import Recommendations


class Recommender(object):

    def pass_args(self, interactions):
        for key, value in interactions.pass_args().items():
            setattr(self, key, value)

    def recommend(self, user, remove_items=None, top=10):

        if isinstance(user, str):
            user = self.user_labels.index(user)

        predictions = self.predict(user_id=user)

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
