import torch
from .base import Implicit
from ..core import UncertainRecommender


class MLP(Implicit):

    def __init__(self, n_user, n_item, embedding_dim, lr, n_negatives=4, dropout=0, n_hidden=3):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.n_negatives = n_negatives
        
        self.n_hidden = n_hidden
        self.dropl = torch.nn.Dropout(p=dropout)
        self.ReLU = torch.nn.ReLU()
        
        # Init embeddings
        self.user_embeddings = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.01)
        
        # Init linear layers
        self.layers = torch.nn.ModuleList()
        size = self.embedding_dim*2
        while len(self.layers) < self.n_hidden:
            self.layers.append(torch.nn.Linear(size, int(size/2), bias=True))
            torch.nn.init.normal_(self.layers[-1].weight, mean=0, std=0.01)
            torch.nn.init.normal_(self.layers[-1].bias, mean=0, std=0.01)
            size = int(size/2)
        self.out_layer = torch.nn.Linear(size, 1, bias=True)
        torch.nn.init.normal_(self.out_layer.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.out_layer.bias, mean=0, std=0.01)
        
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
        if not len(user_embeddings) == len(item_embeddings):
            user_embeddings = user_embeddings.repeat(len(item_embeddings), 1)
        x = torch.cat((user_embeddings, item_embeddings), dim=1)  
        for layer in self.layers:
            x = self.ReLU(layer(x))
            x = self.dropl(x)
        return self.out_layer(x).sigmoid().flatten()


    
class MCDropout(UncertainRecommender):

    def __init__(self, base_model, mc_iteration):
        
        super().__init__()
        self.base_model = base_model
        self.mc_iteration = mc_iteration
        self.base_model.dropl.train()
        self.n_user = base_model.n_user
        self.n_item = base_model.n_item
    
    def __call__(self, user_ids, item_ids=None):
        pred = torch.vstack([self.base_model(user_ids, item_ids) for _ in range(self.mc_iteration)])
        return pred.mean(0), pred.var(0)

    

class Ensemble(UncertainRecommender):

    def __init__(self, models):
        self.models = models

    def __call__(self, user_ids, item_ids=None):
        pred = torch.vstack([model(user_ids, item_ids) for model in self.models])
        return pred.mean(0), pred.var(0)

    

class ItemSupport(UncertainRecommender):

    def __init__(self, base_MF, uncertainty):
        self.MF = base_MF
        self.uncertainty = torch.tensor(uncertainty)

    def __call__(self, user_ids, item_ids=None):
        pred = self.MF(user_ids, item_ids)
        unc = self.uncertainty[item_ids] if item_ids is not None else self.uncertainty
        return pred, unc

    
class UserSupport(UncertainRecommender):

    def __init__(self, base_MF, uncertainty):
        self.MF = base_MF
        self.uncertainty = torch.tensor(uncertainty)

    def __call__(self, user_ids, item_ids=None):
        pred = self.MF(user_ids, item_ids)
        unc = self.uncertainty[user_ids] if item_ids is not None else self.uncertainty[user_ids.expand(self.MF.n_item)]
        return pred, unc

    
    
'''

# MLP stuff with pairwise losses.

import math
import torch
from pytorch_lightning import LightningModule
from ..core import VanillaRecommender, UncertainRecommender


class BprMlp(LightningModule, VanillaRecommender):

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
        
        # Init linear layers
        self.linear0 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=True)
        self.linear1 = torch.nn.Linear(self.embedding_dim, int(self.embedding_dim/2), bias=True)
        self.linear2 = torch.nn.Linear(int(self.embedding_dim/2), 1, bias=True)
        torch.nn.init.normal_(self.linear0.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear0.bias, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=0.01)
        self.var_activation = torch.nn.Softplus()
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{"params": self.user_embeddings.parameters(), 'weight_decay': self.weight_decay},
                                      {"params": self.item_embeddings.parameters(), 'weight_decay': self.weight_decay},
                                      {"params": self.linear0.parameters()}, {"params": self.linear1.parameters()},
                                      {"params": self.linear2.parameters()}], lr=self.lr)
        return optimizer
    
    def forward(self, user_ids, item_ids=None):
        user_embeddings = self.user_embeddings(user_ids)
        
        if item_ids is not None:
            item_embeddings = self.item_embeddings(item_ids)
        else:
            item_embeddings = self.item_embeddings.weight
            user_embeddings = user_embeddings.expand(self.n_item, self.embedding_dim)
            
        input_embeddings = torch.cat((user_embeddings, item_embeddings), dim=1)
        out = self.linear2(self.linear1(self.linear0(input_embeddings)))
        
        return out.flatten()

    def train_val_step(self, user_ids, item_ids):
        neg_item_ids = torch.randint(0, self.n_item, (len(item_ids),), device=item_ids.device)
        
        # Data likelihood
        positives = self(user_ids, item_ids)
        negatives = self(user_ids, neg_item_ids)
        
        prob_ij = torch.sigmoid(positives - negatives)
        
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
        
        
        
class GprMlp(LightningModule, UncertainRecommender):

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
        
        # Init linear layers / Softplus variance activation
        self.linear0 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=True)
        self.linear1 = torch.nn.Linear(self.embedding_dim, int(self.embedding_dim/2), bias=True)
        self.linear2 = torch.nn.Linear(int(self.embedding_dim/2), 2, bias=True)
        torch.nn.init.normal_(self.linear0.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear0.bias, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=0.01)
        self.var_activation = torch.nn.Softplus()

        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{"params": self.user_embeddings.parameters(), 'weight_decay': self.weight_decay},
                                      {"params": self.item_embeddings.parameters(), 'weight_decay': self.weight_decay},
                                      {"params": self.linear0.parameters()}, {"params": self.linear1.parameters()},
                                      {"params": self.linear2.parameters()}], lr=self.lr)
        return optimizer
    
    def forward(self, user_ids, item_ids=None):
        user_embeddings = self.user_embeddings(user_ids)
        
        if item_ids is not None:
            item_embeddings = self.item_embeddings(item_ids)
        else:
            item_embeddings = self.item_embeddings.weight
            user_embeddings = user_embeddings.expand(self.n_item, self.embedding_dim)
        
        input_embedding = torch.cat((user_embeddings, item_embeddings), dim=1)
        out = self.linear2(self.linear1(self.linear0(input_embedding)))
        
        return out[:, 0].flatten(), self.var_activation(out[:, 1].flatten())
        
    def train_val_step(self, user_ids, item_ids):
        neg_item_ids = torch.randint(0, self.n_item, (len(item_ids),), device=item_ids.device)
        
        # Data likelihood
        positives = self(user_ids, item_ids)
        negatives = self(user_ids, neg_item_ids)
        
        mean_diff = positives[0].flatten() - negatives[0].flatten()
        var_sum = self.var_activation(positives[1].flatten()) + self.var_activation(negatives[1].flatten())
        prob_ij = 1 - 0.5 * (1 + torch.erf((-mean_diff) * var_sum.sqrt().reciprocal() / math.sqrt(2)))
        
        return prob_ij
    
    def training_step(self, batch, batch_idx):
        prob_ij = self.train_val_step(batch[:, 0], batch[:, 1])
        self.log('mean_train_prob', prob_ij.mean(), prog_bar=True)
        
        loss = - prob_ij.sum()
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user, rated, targets = batch
        
        prob_ij = self.train_val_step(user.expand(len(targets[0])), targets[0])
        self.log('mean_val_prob', prob_ij.mean(), prog_bar=True)
        
        rec, _, _ = self.rank(user, ignored_item_ids=rated[0], top_n=5)
        hits = torch.isin(rec, targets[0], assume_unique=True)
        n_hits = hits.cumsum(0)
        if n_hits[-1] > 0:
            precision = n_hits / torch.arange(1, 6, device=n_hits.device)
            self.log('val_MAP', torch.sum(precision * hits) / n_hits[-1], prog_bar=True)
        else:
            self.log('val_MAP', torch.tensor(0, dtype=torch.float32), prog_bar=True)
        
    def uncertain_predict(self, user_ids, item_ids=None, threshold=None):
        with torch.no_grad():
            mean, var = self(user_ids, item_ids)
            return 1 - 0.5 * (1 + torch.erf(-mean * var.sqrt().reciprocal() / math.sqrt(2)))
'''


'''

# GMF code with pairwise losses, prob not worth using.

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

'''