import math
import torch
from pytorch_lightning import LightningModule
from ..core import VanillaRecommender, UncertainRecommender
from .base import BPR, GPR


class bprGMF(LightningModule, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Init embeddings
        self.user_embeddings = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.01)
        
        # Init linear layer
        self.linear = torch.nn.Linear(self.embedding_dim, 1, bias=True)
        torch.nn.init.normal_(self.linear.weight, mean=1, std=0.01)
        torch.nn.init.normal_(self.linear.bias, mean=0, std=0.01)
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{"params": self.user_embeddings.parameters(), 'weight_decay': self.weight_decay},
                                      {"params": self.item_embeddings.parameters(), 'weight_decay': self.weight_decay},
                                      {"params": self.linear.parameters()}], lr=self.lr)
        return optimizer
    
    def forward(self, user_ids, item_ids=None):
        user_embeddings = self.user_embeddings(user_ids)
        
        if item_ids is not None:
            item_embeddings = self.item_embeddings(item_ids)
        else:
            item_embeddings = self.item_embeddings.weight
        
        return self.linear(user_embeddings * item_embeddings).flatten()

    def train_val_step(self, user_ids, item_ids):
        neg_item_ids = torch.randint(0, self.n_item, (len(item_ids),), device=item_ids.device)
        
        # Selecting
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        neg_item_embeddings = self.item_embeddings(neg_item_ids)
        
        # Data likelihood
        positives_dot = self.linear(user_embeddings * item_embeddings).flatten()
        negatives_dot = self.linear(user_embeddings * neg_item_embeddings).flatten()
        
        prob_ij = torch.sigmoid(positives_dot - negatives_dot)
        
        return prob_ij
    
    def training_step(self, batch, batch_idx):
        prob_ij = self.train_val_step(batch[:, 0], batch[:, 1])
        self.log('mean_train_prob', prob_ij.mean(), prog_bar=True)
        
        nll = - prob_ij.log().sum()
        self.log('train_nll', nll)
        return nll

    def validation_step(self, batch, batch_idx):
        user, rated, targets = batch
        
        prob_ij = self.train_val_step(user.expand(len(targets[0])), targets[0])
        self.log('mean_val_prob', prob_ij.mean(), prog_bar=True)
        
        rec, _ = self.rank(user, ignored_item_ids=rated[0], top_n=5)
        hits = torch.isin(rec, targets[0], assume_unique=True)
        n_hits = hits.cumsum(0)
        if n_hits[-1] > 0:
            precision = n_hits / torch.arange(1, 6, device=n_hits.device)
            self.log('val_MAP', torch.sum(precision * hits) / n_hits[-1], prog_bar=True)
        else:
            self.log('val_MAP', torch.tensor(0, dtype=torch.float32), prog_bar=True)



class UncertainGMF(GPR):

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
        
        # Init linear layer / Softplus variance activation
        self.linear = torch.nn.Linear(self.embedding_dim, 2, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=1, std=0.01)
        self.var_activation = torch.nn.Softplus()
        
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
        out = self.linear(user_embeddings * item_embeddings)
        return out[:, 0].flatten(), self.var_activation(out[:, 1].flatten())
    
    def get_penalty(self, user_embeddings, item_embeddings, neg_item_embeddings):
        user_penalty = self.weight_decay_user * (user_embeddings ** 2).sum()
        item_penalty = self.weight_decay_item * (item_embeddings ** 2).sum()
        neg_item_penalty = self.weight_decay_negative * (neg_item_embeddings ** 2).sum()
        return user_penalty + item_penalty + neg_item_penalty
