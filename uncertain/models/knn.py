import torch
from uncertain.core import Recommender


class ItemKNN(Recommender):

    def __init__(self, item_similarities, k=10, weighted=True):

        self.item_similarities = torch.tensor(item_similarities)
        self.k = k
        self.weighted = weighted

    def predict(self, user_profile):

        relevance = torch.empty(len(self.item_similarities))
        for idx, item in enumerate(self.item_similarities):
            if idx in user_profile:
                relevance[idx] = -float('inf')
            else:
                relevance[idx] = torch.sum(item[user_profile][:self.k])

        return relevance
