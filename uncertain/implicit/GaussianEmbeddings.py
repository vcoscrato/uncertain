import math
import torch
from .base import Implicit


class GER(Implicit):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, n_negatives):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
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
            return self.item_mean(item_ids), self.item_var(item_ids)
        else:
            return self.item_mean.weight, self.item_var.weight
        
    def interact(self, user_embeddings, item_embeddings):
        user_mean, user_var = user_embeddings
        item_mean, item_var = item_embeddings
        
        mean = (user_mean * item_mean).sum(1).flatten() # + item_bias.flatten()
        # var = torch.sum(item_var * (2 * user_mean ** 2 + user_var), 1) + torch.sum(user_var * item_mean, 1)
        var =  2 * (user_mean * item_var * user_mean).sum(1).flatten() + \
               (item_mean * user_var * item_mean).sum(1).flatten() + \
               (user_var * item_var).sum(1).flatten()
        return 1 - (0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))), var
        
    def get_penalty(self, user_embeddings, item_embeddings, neg_item_embeddings):
        user_prior = (- 3/2 * self.user_var.weight.reciprocal().log()).sum() - \
                     ((4 + self.user_mean.weight**2) / (self.user_var.weight.reciprocal() * 2)).sum()
        item_prior = (- 3/2 * self.item_var.weight.reciprocal().log()).sum() - \
                     ((4 + self.item_mean.weight**2) / (self.item_var.weight.reciprocal() * 2)).sum()
        return - self.weight_decay * (user_prior + item_prior)

    
    
    
# Code currently on pairwise verion, need to unify

import math
import torch
import numpy as np
from pytorch_lightning import LightningModule
from torch.distributions import Normal, Gamma
from ..core import VanillaRecommender, UncertainRecommender


class GER(LightningModule, UncertainRecommender):

    def __init__(self, n_user, n_item, n_interaction, embedding_dim, lr, weight_decay):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_interaction = n_interaction
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay

        # Init embeddings (3 dimensional - (user or item), embedding factors and (mean, viariance) - respectively)
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

    def forward(self, user_ids, item_ids=None):
        user_mean = self.user_mean(user_ids)
        user_var = self.user_var(user_ids)
        
        if item_ids is not None:
            item_mean = self.item_mean(item_ids)
            item_var = self.item_var(item_ids)
            item_bias = self.item_bias(item_ids)
        else:
            item_mean = self.item_mean.weight
            item_var = self.item_var.weight
            item_bias = self.item_bias.weight
    
        mean = (user_mean * item_mean).sum(1).flatten() + item_bias.flatten()
        var = (user_mean**2 * user_var + item_mean**2 * user_var + user_var * user_var).sum(1).flatten()
        return mean, var
    
    def train_val_step(self, user_ids, item_ids):
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

        # Data Likelihood
        bias_diff = (item_bias - neg_item_bias).flatten()
        prob_ij = 0.5 * (1 + torch.erf((bias_diff - mean) * var.sqrt().reciprocal() / math.sqrt(2)))
        
        # Prior
        user_prior = (- 3/2 * self.user_var.weight.reciprocal().log()).sum() - ((4 + self.user_mean.weight**2) / (self.user_var.weight.reciprocal() * 2)).sum()
        item_prior = (- 3/2 * self.item_var.weight.reciprocal().log()).sum() - ((4 + self.item_mean.weight**2) / (self.item_var.weight.reciprocal() * 2)).sum()
        bias_prior = - self.weight_decay * (self.item_bias.weight**2).sum()
        prior = user_prior + item_prior
    
        return prob_ij, prior    
        
    def training_step(self, batch, batch_idx):
        prob_ij, prior = self.train_val_step(batch[:, 0], batch[:, 1])
        self.log('mean_train_prob', prob_ij.mean(), prog_bar=True)
        scale = self.n_interaction / len(batch)
        
        nll = - (scale * prob_ij.log().sum() + prior)
        self.log('train_nll', nll)
        return nll

    def validation_step(self, batch, batch_idx):
        user, rated, targets = batch
        
        prob_ij, prior = self.train_val_step(user.expand(len(targets[0])), targets[0])
        self.log('mean_val_prob', prob_ij.mean(), prog_bar=True)
        
        rec, _, _ = self.rank(user, ignored_item_ids=rated[0], top_n=5)
        hits = torch.isin(rec[:5], targets[0], assume_unique=True)
        n_hits = hits.cumsum(0)
        if n_hits[-1] > 0:
            precision = n_hits / torch.arange(1, 6, device=n_hits.device)
            self.log('val_MAP', torch.sum(precision * hits) / n_hits[-1], prog_bar=True)
        else:
            self.log('val_MAP', torch.tensor(0, dtype=torch.float32), prog_bar=True)
                         
    def uncertain_predict(self, user, threshold=None):
        with torch.no_grad():
            mean, var = self(user)
            return 1 - 0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))
