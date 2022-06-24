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
