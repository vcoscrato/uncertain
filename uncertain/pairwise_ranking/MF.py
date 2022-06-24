import math
import torch
from pytorch_lightning import LightningModule
from ..core import VanillaRecommender, UncertainRecommender
from .base import BPR, GPR


class bprMF(BPR):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
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
        return (user_embeddings * item_embeddings).sum(1)
    
    def get_penalty(self, user_embeddings, item_embeddings, neg_item_embeddings):
        user_penalty = self.weight_decay_user * (user_embeddings ** 2).sum()
        item_penalty = self.weight_decay_item * (item_embeddings ** 2).sum()
        neg_item_penalty = self.weight_decay_negative * (neg_item_embeddings ** 2).sum()
        return user_penalty + item_penalty + neg_item_penalty
            
            

class UncertainMF(GPR):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, embedding_dim_var=None):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        if embedding_dim_var is None:
            self.embedding_dim_var = embedding_dim
        else:
            self.embedding_dim_var = embedding_dim_var
        self.lr = lr
        if type(weight_decay) is tuple:
            self.weight_decay_user = weight_decay[0]
            self.weight_decay_item = weight_decay[1]
            self.weight_decay_negative = weight_decay[2]
        else:
            self.weight_decay_user = weight_decay
            self.weight_decay_item = weight_decay
            self.weight_decay_negative = weight_decay
    
        # Init embeddings
        self.user_mean = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_mean = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_mean.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_mean.weight, mean=0, std=0.01)
        
        self.user_var = torch.nn.Embedding(self.n_user, self.embedding_dim_var)
        self.item_var = torch.nn.Embedding(self.n_item, self.embedding_dim_var)
        torch.nn.init.constant_(self.user_var.weight, 1/math.sqrt(self.embedding_dim_var))
        torch.nn.init.constant_(self.item_var.weight, 1/math.sqrt(self.embedding_dim_var))
        self.var_activation = torch.nn.Softplus()
        
        self.save_hyperparameters()
    
    def get_user_embeddings(self, user_ids):
        return self.user_mean(user_ids), self.user_var(user_ids)
    
    def get_item_embeddings(self, item_ids=None):
        if item_ids is not None:
            item_mean = self.item_mean(item_ids)
            item_var = self.item_var(item_ids)
        else:
            item_mean = self.item_mean.weight
            item_var = self.item_var.weight
        return item_mean, item_var
    
    def interact(self, user_embeddings, item_embeddings):
        mean = (user_embeddings[0] * item_embeddings[0]).sum(1)
        var = self.var_activation((user_embeddings[1] * item_embeddings[1]).sum(1))
        return mean, var
    
    def get_penalty(self, user_embeddings, item_embeddings, neg_item_embeddings):
        user_penalty = self.weight_decay_user * ((user_embeddings[0] ** 2).sum() + (user_embeddings[1] ** 2).sum())
        item_penalty = self.weight_decay_item * ((item_embeddings[0] ** 2).sum() + (item_embeddings[1] ** 2).sum())
        neg_item_penalty = self.weight_decay_negative * ((neg_item_embeddings[0] ** 2).sum() + (neg_item_embeddings[1] ** 2).sum())
        return user_penalty + item_penalty + neg_item_penalty

    
    
class PretrainedUncertainMF(GPR):
    
    def __init__(self, baseline, embedding_dim, lr, weight_decay):
        
        super().__init__()
        self.baseline = baseline
        '''
        for param in self.baseline.parameters():
            param.requires_grad = False
        '''
        
        self.n_user = baseline.n_user
        self.n_item = baseline.n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        if type(weight_decay) is tuple:
            self.weight_decay_user = weight_decay[0]
            self.weight_decay_item = weight_decay[1]
            self.weight_decay_negative = weight_decay[2]
        else:
            self.weight_decay_user = weight_decay
            self.weight_decay_item = weight_decay
            self.weight_decay_negative = weight_decay
            
        # Init embeddings
        self.user_embeddings = torch.nn.Embedding(self.n_user, embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, embedding_dim)
        torch.nn.init.constant_(self.user_embeddings.weight, 1/math.sqrt(self.embedding_dim))
        torch.nn.init.constant_(self.item_embeddings.weight, 1/math.sqrt(self.embedding_dim))
        self.var_activation = torch.nn.Softplus()
        self.save_hyperparameters()
    
    def get_user_embeddings(self, user_ids):
        return self.baseline.get_user_embeddings(user_ids), self.user_embeddings(user_ids)
    
    def get_item_embeddings(self, item_ids=None):
        if item_ids is not None:
            item_embeddings = self.item_embeddings(item_ids)
        else:
            item_embeddings = self.item_embeddings.weight
        return self.baseline.get_item_embeddings(item_ids), item_embeddings
    
    def interact(self, user_embeddings, item_embeddings):
        mean = self.baseline.interact(user_embeddings[0], item_embeddings[0])
        var = self.var_activation((user_embeddings[1] * item_embeddings[1]).sum(1))
        return mean, var
    
    def get_penalty(self, user_embeddings, item_embeddings, neg_item_embeddings):
        user_penalty = self.weight_decay_user * (user_embeddings[1] ** 2).sum()
        item_penalty = self.weight_decay_item * (item_embeddings[1] ** 2).sum()
        neg_item_penalty = self.weight_decay_negative * (neg_item_embeddings[1] ** 2).sum()
        return user_penalty + item_penalty + neg_item_penalty
