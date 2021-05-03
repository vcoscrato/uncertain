import numpy as np
from .base import Recommendations


class ItemKNN(object):

    def __init__(self, item_similarities, k=10, weighted=True):

        self.item_similarities = item_similarities
        self.k = k
        self.weighted = weighted

    def recommend(self, user_profile):

        relevance = np.empty(len(item_similarities))
        for idx, item in self.item_similarities:
            if idx in user_profile:
                relevance[idx] = -np.inf
            else:
                relevance[idx] = np.sum(item[user_profile][:self.k])

        return Recommendations(user='', items=relevance.argsort()[:10])