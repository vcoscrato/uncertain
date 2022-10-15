import torch
from .base import Implicit


class MF(Implicit):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, n_negatives):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.n_negatives = n_negatives
        if type(weight_decay) is tuple:
            self.weight_decay_user = weight_decay[0]
            self.weight_decay_item = weight_decay[1]
            self.weight_decay_negative = weight_decay[2]
        else:
            self.weight_decay_user = weight_decay
            self.weight_decay_item = weight_decay
            self.weight_decay_negative = weight_decay
    
        # Init embeddings
        self.user_embeddings = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.01)
        
        self.save_hyperparameters()
    
    def get_user_embeddings(self, user_ids):
        return self.user_embeddings(user_ids)
    
    def get_item_embeddings(self, item_ids=None):
        if item_ids is not None:
            item_embeddings = self.item_embeddings(item_ids)
        else:
            item_embeddings = self.item_embeddings.weight
        return item_embeddings
    
    def interact(self, user_embeddings, item_embeddings):
        return (user_embeddings * item_embeddings).sum(1).sigmoid()
    
    def get_penalty(self, user_embeddings, item_embeddings, neg_item_embeddings):
        user_penalty = self.weight_decay_user * (user_embeddings ** 2).sum()
        item_penalty = self.weight_decay_item * (item_embeddings ** 2).sum()
        neg_item_penalty = self.weight_decay_negative * (neg_item_embeddings ** 2).sum()
        return user_penalty + item_penalty + neg_item_penalty
