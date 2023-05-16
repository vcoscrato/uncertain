import math
import torch
from ..core import UncertainRecommender
from .base import Implicit
from ..loss import GPR, GBR


class CPMF(Implicit, UncertainRecommender):
    
    def __init__(self, n_user, n_item, embedding_dim=128, lr=1e-4, weight_decay=0, init_var=1, n_negatives=None, loss='GBR'):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.n_negatives = n_negatives
        self.init_var=init_var
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
        self.user_gammas = torch.nn.Embedding(self.n_user, 1)
        self.item_gammas = torch.nn.Embedding(self.n_item, 1)
        torch.nn.init.constant_(self.user_gammas.weight, math.sqrt(self.init_var))
        torch.nn.init.constant_(self.item_gammas.weight, math.sqrt(self.init_var))
        self.var_activation = torch.nn.Softplus()
        
        if loss == 'GBR':
            self.loss = GBR
        elif loss == 'GPR': 
            self.loss = GPR
        elif loss == 'AUR':
            self.loss = AUR
        self.save_hyperparameters()
    
    def get_user_embeddings(self, user_ids):
        return self.user_embeddings(user_ids), self.user_gammas(user_ids)
    
    def get_item_embeddings(self, item_ids=None):
        if item_ids is not None:
            return self.item_embeddings(item_ids), self.item_gammas(item_ids)
        else:
            return self.item_embeddings.weight, self.item_gammas.weight
    
    def interact(self, user_embeddings, item_embeddings):
        user_embeddings, user_gammas = user_embeddings
        item_embeddings, item_gammas = item_embeddings
        mean = (user_embeddings * item_embeddings).sum(1)
        var = self.var_activation(user_gammas * item_gammas).flatten()
        return mean, var
    
    def uncertain_transform(self, obj):
        mean, var = obj
        return 1 - 0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))
    
    def get_penalty(self, user_embeddings, item_embeddings, neg_item_embeddings):
        user_penalty = self.weight_decay_user * (user_embeddings[0] ** 2).sum()
        item_penalty = self.weight_decay_item * (item_embeddings[0] ** 2).sum()
        neg_item_penalty = self.weight_decay_negative * (neg_item_embeddings[0] ** 2).sum()
        return user_penalty + item_penalty + neg_item_penalty
    
    def get_user_uncertainty(self, user=None):
        with torch.no_grad():
            if user is None:
                return self.user_gammas.weight.flatten().numpy()
            else:
                return self.user_gammas.weight.flatten().numpy()[user]
            

'''
Pretrained version of CPMF that shouldn't be used because the current validation protocol won't work

class PretrainedCPMF(Implicit, UncertainRecommender):
    
    def __init__(self, baseline, lr, n_negatives=1, loss='GBR'):
        
        super().__init__()
        self.baseline = baseline
        
        for param in self.baseline.parameters():
            param.requires_grad = False
        
        self.n_user = baseline.n_user
        self.n_item = baseline.n_item
        self.lr = lr
        self.n_negatives = n_negatives
            
        # Init embeddings
        self.user_embeddings = torch.nn.Embedding(self.n_user, 1)
        self.item_embeddings = torch.nn.Embedding(self.n_item, 1)
        torch.nn.init.constant_(self.user_embeddings.weight, 10)
        torch.nn.init.constant_(self.item_embeddings.weight, 10)
        self.var_activation = torch.nn.Softplus()
        
        if loss == 'GBR':
            self.loss = GBR
        elif loss == 'GPR': 
            self.loss = GPR
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
        var = self.var_activation((user_embeddings[1] * item_embeddings[1]).flatten())
        return mean, var
    
    def uncertain_transform(self, obj):
        mean, var = obj
        return 1 - 0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))
'''
