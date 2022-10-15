import torch
from pytorch_lightning import LightningModule
from ..core import FactorizationModel, VanillaRecommender, UncertainRecommender


class Implicit(LightningModule, VanillaRecommender):
    
    def __init__(self):
        super().__init__()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, user_ids, item_ids=None):
        user_embeddings = self.get_user_embeddings(user_ids)
        item_embeddings = self.get_item_embeddings(item_ids)
        return self.interact(user_embeddings, item_embeddings)
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids = batch[:, 0], batch[:, 1]
        
        # Selecting
        user_embeddings = self.get_user_embeddings(user_ids)
        item_embeddings = self.get_item_embeddings(item_ids)
        neg_item_ids = torch.randint(0, self.n_item, (len(user_ids) * self.n_negatives,), device=item_ids.device)
        neg_item_embeddings = self.get_item_embeddings(neg_item_ids)
        
        # Data likelihood
        prob_ij = torch.cat((self.interact(user_embeddings, item_embeddings), 
                             1 - self.interact(user_embeddings.repeat(self.n_negatives, 1), neg_item_embeddings)))
        self.log('train_likelihood', prob_ij.mean(), prog_bar=True)
        loss = - prob_ij.log().sum()
            
        # Penalty
        if hasattr(self, 'get_penalty'):
            penalty = self.get_penalty(user_embeddings, item_embeddings, neg_item_embeddings)
            loss += penalty
            
        # Loss
        self.log('train_nll', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        user, rated, targets = batch
        rec, _ = self.rank(user, ignored_item_ids=rated[0], top_n=5)
        hits = torch.isin(rec, targets[0], assume_unique=True)
        n_hits = hits.cumsum(0)
        if n_hits[-1] > 0:
            precision = n_hits / torch.arange(1, 6, device=n_hits.device)
            self.log('val_MAP', torch.sum(precision * hits) / n_hits[-1], prog_bar=True)
        else:
            self.log('val_MAP', torch.tensor(0, dtype=torch.float32), prog_bar=True)

            
            
class UncertainImplicit(LightningModule, UncertainRecommender):
    
    def __init__(self):
        super().__init__()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, user_ids, item_ids=None):
        user_embeddings = self.get_user_embeddings(user_ids)
        item_embeddings = self.get_item_embeddings(item_ids)
        return self.interact(user_embeddings, item_embeddings)
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids = batch[:, 0], batch[:, 1]
        
        # Selecting
        user_embeddings = self.get_user_embeddings(user_ids)
        item_embeddings = self.get_item_embeddings(item_ids)
        neg_item_ids = torch.randint(0, self.n_item, (len(user_ids) * self.n_negatives,), device=item_ids.device)
        neg_item_embeddings = self.get_item_embeddings(neg_item_ids)
        
        # Data likelihood
        pos_prob, pos_unc = self.interact(user_embeddings, item_embeddings)
        neg_prob, neg_unc = self.interact(user_embeddings.repeat(self.n_negatives, 1), neg_item_embeddings)
        prob_ij = torch.cat((pos_prob, 1 - neg_prob))
        self.log('train_likelihood', prob_ij.mean(), prog_bar=True)
        loss = - prob_ij.log().sum()
            
        # Penalty
        if hasattr(self, 'get_penalty'):
            penalty = self.get_penalty(user_embeddings, item_embeddings, neg_item_embeddings)
            loss += penalty
            
        # Loss
        self.log('train_nll', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        user, rated, targets = batch
        rec, _, _ = self.rank(user, ignored_item_ids=rated[0], top_n=5)
        hits = torch.isin(rec, targets[0], assume_unique=True)
        n_hits = hits.cumsum(0)
        if n_hits[-1] > 0:
            precision = n_hits / torch.arange(1, 6, device=n_hits.device)
            self.log('val_MAP', torch.sum(precision * hits) / n_hits[-1], prog_bar=True)
        else:
            self.log('val_MAP', torch.tensor(0, dtype=torch.float32), prog_bar=True)

            
            
