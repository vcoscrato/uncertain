import math
import torch
from ..core import UncertainRecommender
from .base import Implicit
from .loss import GPR, GBR


def get_prior(mean, var, bias=None):
    '''
    Returns log prior
    '''
    precision = var.reciprocal()
    out = (- 3/2 * precision.log()).sum() - ((4 + mean**2) / (precision * 2)).sum()
    if bias is not None:
        out += (bias**2).sum()
    return out


class GER(Implicit, UncertainRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        # self.n_interaction = n_interaction
        self.embedding_dim = embedding_dim
        self.lr = lr

        self.user_mean = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_mean = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_mean.weight, 0, 1)
        torch.nn.init.normal_(self.item_mean.weight, 0, 1)
        
        self.user_var = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_var = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.constant_(self.user_var.weight, 1)
        torch.nn.init.constant_(self.item_var.weight, 1)
        
        self.item_bias = torch.nn.Embedding(self.n_item, 1)
        torch.nn.init.constant_(self.item_bias.weight, 0)
        
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{"params": self.user_mean.parameters()}, {"params": self.user_var.parameters()},
                                      {"params": self.item_mean.parameters()}, {"params": self.item_var.parameters()},
                                      {"params": self.item_bias.parameters()}], lr=self.lr)
        return optimizer
    
    def get_user_embeddings(self, user_ids):
        return self.user_mean(user_ids), self.user_var(user_ids)
    
    def get_item_embeddings(self, item_ids):
        if item_ids is not None:
            return self.item_mean(item_ids), self.item_var(item_ids), self.item_bias(item_ids)
        else:
            return self.item_mean.weight, self.item_var.weight, self.item_bias.weight
        
    def interact(self, user_embeddings, item_embeddings):
        user_mean, user_var = user_embeddings
        item_mean, item_var, item_bias = item_embeddings
        
        mean = (user_mean * item_mean).sum(1).flatten() + item_bias.flatten()
        var = (user_mean**2 * user_var + item_mean**2 * user_var + user_var * item_var).sum(1).flatten()
        return mean, var
    
    def uncertain_transform(self, obj):
        mean, var = obj
        return 1 - 0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))

    def training_step(self, batch, batch_idx):
        user_ids, item_ids = batch[:, 0], batch[:, 1]
        neg_item_ids = torch.randint(0, self.n_item, (len(item_ids),), device=item_ids.device)

        # Selecting
        user_mean = self.user_mean(user_ids)
        item_mean = self.item_mean(item_ids)
        neg_item_mean = self.item_mean(neg_item_ids)
        
        user_var = self.user_var(user_ids)
        item_var = self.item_var(item_ids)
        neg_item_var = self.item_var(neg_item_ids)
        
        item_bias = self.item_bias(item_ids)
        neg_item_bias = self.item_bias(neg_item_ids)

        # Moment matching
        item_mean_diff = neg_item_mean - item_mean
        item_var_sum = item_var + neg_item_var
        user_factor = 2 * user_mean**2 + user_var
        
        mean = (user_mean * item_mean_diff).sum(1)
        var = (item_var_sum * user_factor + user_var * item_mean_diff**2).sum(1)
        
        # Prior
        user_prior = get_prior(user_mean, user_var)
        item_prior = get_prior(item_mean, item_var, item_bias)
        neg_item_prior = get_prior(neg_item_mean, neg_item_var, neg_item_bias)
        prior = user_prior + item_prior + neg_item_prior
        # scale = self.n_interaction / len(batch) # Commented because now calc. prior only on used params
    
        # Data Likelihood
        bias_diff = (item_bias - neg_item_bias).flatten()
        prob_ij = 0.5 * (1 + torch.erf((bias_diff - mean) * var.sqrt().reciprocal() / math.sqrt(2)))
        
        self.log('mean_train_prob', prob_ij.mean(), prog_bar=True)
        nll = - (prob_ij.log().sum() + prior)
        self.log('loss', nll)
        return nll
        
        
class BinaryGER(Implicit, UncertainRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, n_negatives=None, loss='GBR'):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.n_negatives = n_negatives

        self.user_mean = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_mean = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_mean.weight, 0, 1)
        torch.nn.init.normal_(self.item_mean.weight, 0, 1)
        
        self.user_var = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_var = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.constant_(self.user_var.weight, 1)
        torch.nn.init.constant_(self.item_var.weight, 1)
        
        # self.item_bias = torch.nn.Embedding(self.n_item, 1)
        # torch.nn.init.constant_(self.item_bias.weight, 0)
        
        if loss == 'GBR':
            self.loss = GBR
        elif loss == 'GPR': 
            self.loss = GPR
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{"params": self.user_mean.parameters()}, {"params": self.user_var.parameters()},
                                      {"params": self.item_mean.parameters()}, {"params": self.item_var.parameters()}], lr=self.lr)
                                      # {"params": self.item_bias.parameters()}], lr=self.lr)
        return optimizer

    def get_user_embeddings(self, user_ids):
        return self.user_mean(user_ids), self.user_var(user_ids)
    
    def get_item_embeddings(self, item_ids):
        if item_ids is not None:
            return self.item_mean(item_ids), self.item_var(item_ids) #, self.item_bias(item_ids)
        else:
            return self.item_mean.weight, self.item_var.weight #, self.item_bias.weight
        
    def interact(self, user_embeddings, item_embeddings):
        user_mean, user_var = user_embeddings
        item_mean, item_var = item_embeddings # No bias
        
        mean = (user_mean * item_mean).sum(1).flatten() # + item_bias.flatten()
        var = (2*(user_mean**2 * user_var) + (item_mean**2 * user_var) + user_var * item_var).sum(1).flatten()
        return mean, var
    
    def uncertain_transform(self, obj):
        mean, var = obj
        return 1 - 0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))
        
    def get_penalty(self, user_embeddings, item_embeddings, neg_item_embeddings):
        user_mean, user_var = user_embeddings
        item_mean, item_var = item_embeddings # No bias
        neg_item_mean, neg_item_var = neg_item_embeddings # No bias
        
        user_prior = get_prior(user_mean, user_var)
        item_prior = get_prior(item_mean, item_var)
        neg_item_prior = get_prior(neg_item_mean, neg_item_var)
        return user_prior + item_prior + neg_item_prior
