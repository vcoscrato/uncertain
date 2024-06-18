import torch
from ..core import UncertainRecommender
from .base import Implicit


class HeuristicUncertainty(Implicit, UncertainRecommender):
    
    def __init__(self, baseline, user_uncertainty=None, item_uncertainty=None):
        
        super().__init__()
        self.baseline = baseline
        self.n_user = self.baseline.n_user
        self.n_item = self.baseline.n_item
        self.user_uncertainty = torch.tensor(user_uncertainty) if user_uncertainty is not None else None
        self.item_uncertainty = torch.tensor(item_uncertainty) if item_uncertainty is not None else None

    def get_user_embeddings(self, user_ids):
        if self.user_uncertainty is not None:
            unc = self.user_uncertainty[user_ids]
        else:
            unc = 0
        return self.baseline.get_user_embeddings(user_ids), unc

    def get_item_embeddings(self, item_ids=None):
        if self.item_uncertainty is not None and item_ids is not None:
            emb = self.baseline.get_item_embeddings(item_ids)
            unc = self.item_uncertainty[item_ids]
        elif item_ids is not None:
            emb = self.baseline.get_item_embeddings(item_ids)
            unc = 0
        elif self.item_uncertainty is not None:
            emb = self.baseline.item_embeddings.weight
            unc = self.item_uncertainty
        else:
            emb = self.baseline.item_embeddings.weight
            unc = 0
        return emb, unc
    
    def interact(self, user_embeddings, item_embeddings):
        score = self.baseline.interact(user_embeddings[0], item_embeddings[0])
        unc = user_embeddings[1] + item_embeddings[1]
        if score.numel() != unc.numel():
            unc = unc.repeat(len(score))
        return score, unc
