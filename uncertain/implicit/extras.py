from ..core import UncertainRecommender
from .base import Implicit


class HeuristicUncertainty(Implicit, UncertainRecommender):
    
    def __init__(self, baseline):
        
        super().__init__()
        self.baseline = baseline
        self.n_user = self.baseline.n_user
        self.n_item = self.baseline.n_item
        self.get_user_embeddings = self.baseline.get_user_embeddings
        self.get_item_embeddings = self.baseline.get_item_embeddings
        
    def interact(self, user_embeddings, item_embeddings):
        score = self.baseline.interact(user_embeddings, item_embeddings)
        return score, 1 - score.abs()
