import math
import torch
from pytorch_lightning import LightningModule
from ..core import FactorizationModel, VanillaRecommender, UncertainRecommender


def BCE(positive_scores, negative_scores):
    '''
    Binary cross entropy. Returns the average likelihood and the NLL.
    '''
    probs = torch.cat((positive_scores.sigmoid(), 1 - neg_scores.sigmoid()))
    return probs.mean(), - probs.log().sum()


def BPR(positive_scores, negative_scores):
    '''
    Bayesian Personalized Ranking. Returns the average likelihood and the NLL.
    '''
    probs = torch.sigmoid(positive_scores - negative_scores)
    return probs.mean(), - probs.log().sum()


def GPR(positive_scores, negative_scores):
    '''
    Gaussian Pairwise Ranking. Returns the average likelihood and the NLL.
    '''
    pos_mean, pos_var = positive_scores
    neg_mean, neg_var = negative_scores
    mean_diff = pos_mean - neg_mean
    var_sum = pos_var + neg_var
    probs = 1 - 0.5 * (1 + torch.erf((-mean_diff) * var_sum.sqrt().reciprocal() / math.sqrt(2)))
    return probs.mean(), - probs.log().sum()


def GBR(positive_scores, negative_scores):
    '''
    Gaussian Binary Ranking. Returns the average likelihood and the NLL.
    '''
    pos_mean, pos_var = positive_scores
    pos_probs = 1 - 0.5 * (1 + torch.erf((0 - pos_mean) * pos_var.sqrt().reciprocal() / math.sqrt(2)))
    neg_mean, neg_var = negative_scores
    neg_probs = 1 - 0.5 * (1 + torch.erf((0 - neg_mean) * neg_var.sqrt().reciprocal() / math.sqrt(2)))
    var_sum = pos_var + neg_var
    probs = torch.cat((positive_scores, neg_scores))
    return probs.mean(), - probs.log().sum()


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
        if self.n_negative == 1 or not hasattr(self, 'n_negative'):
            neg_user_embeddings = user_embeddings
            neg_item_ids = torch.randint(0, self.n_item, (len(item_ids),), device=item_ids.device)
        else:
            if type(user_embeddings) is tuple:
                neg_user_embeddings = (user_embeddings[0].repeat(self.n_negatives, 1), 
                                       user_embeddings[1].repeat(self.n_negatives, 1))
            else:
                neg_user_embeddings = user_embeddings.repeat(self.n_negatives, 1)
            neg_item_ids = torch.randint(0, self.n_item, (len(user_ids) * self.n_negatives,), device=item_ids.device)
        neg_item_embeddings = self.get_item_embeddings(neg_item_ids)
        
        # Data likelihood
        pos_scores = self.interact(user_embeddings, item_embeddings)
        neg_scores = self.interact(neg_user_embeddings, neg_item_embeddings)
        
        train_likelihood, loss = self.loss(pos_scores, neg_scores)
        self.log('train_likelihood', train_likelihood, prog_bar=True)
            
        # Penalty
        if hasattr(self, 'get_penalty'):
            penalty = self.get_penalty(user_embeddings, item_embeddings, neg_item_embeddings)
            loss += penalty
            
        # Loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        user, rated, targets = batch
        rec = self.rank(user, ignored_item_ids=rated[0], top_n=5)[0]
        hits = torch.isin(rec, targets[0], assume_unique=True)
        n_hits = hits.cumsum(0)
        if n_hits[-1] > 0:
            precision = n_hits / torch.arange(1, 6, device=n_hits.device)
            self.log('val_MAP', torch.sum(precision * hits) / n_hits[-1], prog_bar=True)
        else:
            self.log('val_MAP', torch.tensor(0, dtype=torch.float32), prog_bar=True)

            
'''       
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
        if type(user_embeddings) is tuple:
            neg_user_embeddings = (user_embeddings[0].repeat(self.n_negatives, 1), 
                                   user_embeddings[1].repeat(self.n_negatives, 1))
        else:
            neg_user_embeddings = user_embeddings.repeat(self.n_negatives, 1)
        neg_prob, neg_unc = self.interact(neg_user_embeddings, neg_item_embeddings)
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
'''
            
            
########################################################### OLD CODE

        
class biasMF(UncertainBPR):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, loss)
        self.user_bias = torch.nn.Embedding(self.n_user, 1)
        self.item_bias = torch.nn.Embedding(self.n_item, 1)
        torch.nn.init.normal_(self.user_bias.weight, mean=1, std=0.01)
        torch.nn.init.normal_(self.item_bias.weight, mean=1, std=0.01)
        self.rho_activation = torch.nn.Softplus()
        self.save_hyperparameters()
        
    def forward(self, user_ids, item_ids):
        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)
        return self.dot(user_ids, item_ids), self.rho_activation(user_bias + item_bias).flatten() + self.padding
    
    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            out = self(user_ids, item_ids)
            return out[0].numpy(), out[1].numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            user_bias = self.user_bias(user)
            mean = (user_embedding * self.item_embeddings.weight).sum(1)
            unc = self.rho_activation(user_bias + self.item_bias.weight).flatten() + self.padding
            return mean, unc
        
    def uncertain_predict_user(self, user, threshold=None):
        with torch.no_grad():
            mean, var = self.predict_user(user)
            return 1 - 0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))

        
class TwoWayMF(UncertainBPR):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, loss)
        self.user_embeddings_rho = torch.nn.Embedding(self.n_user, embedding_dim)
        self.item_embeddings_rho = torch.nn.Embedding(self.n_item, embedding_dim)
        torch.nn.init.normal_(self.user_embeddings_rho.weight, mean=1/self.embedding_dim, std=0.01)
        torch.nn.init.normal_(self.item_embeddings_rho.weight, mean=1/self.embedding_dim, std=0.01)
        self.rho_activation = torch.nn.Softplus()
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        params = [{'params': self.user_embeddings.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay},
                  {'params': self.item_embeddings.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay},
                  {'params': self.user_embeddings_rho.parameters(), 'lr': self.lr*10, 'weight_decay': self.weight_decay},
                  {'params': self.item_embeddings_rho.parameters(), 'lr': self.lr*10, 'weight_decay': self.weight_decay},]
        return torch.optim.Adam(params)

    def forward(self, user_ids, item_ids):
        user_embedding_rho = self.user_embeddings_rho(user_ids)
        item_embedding_rho = self.item_embeddings_rho(item_ids)
        return self.dot(user_ids, item_ids), self.rho_activation((user_embedding_rho * item_embedding_rho).sum(1)) + self.padding
    
    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            out = self(user_ids, item_ids)
            return out[0].numpy(), out[1].numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            user_embedding_rho = self.user_embeddings_rho(user)
            mean = (user_embedding * self.item_embeddings.weight).sum(1)
            unc = self.rho_activation((user_embedding_rho * self.item_embeddings_rho.weight).sum(1)) + self.padding
            return mean, unc
        
    def uncertain_predict_user(self, user, threshold=None):
        with torch.no_grad():
            mean, var = self.predict_user(user)
            return 1 - 0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))
        
        
class MFUncertainty(Implicit, FactorizationModel, UncertainRecommender):
    
    def __init__(self, baseline, embedding_dim, lr, weight_decay, loss):
        super().__init__(baseline.n_user, baseline.n_item, embedding_dim, lr, weight_decay, **dict(loss_func=loss, n_negative=1, padding=0))
        self.baseline = baseline
        self.user_embeddings_rho = torch.nn.Embedding(self.n_user, embedding_dim)
        self.item_embeddings_rho = torch.nn.Embedding(self.n_item, embedding_dim)
        torch.nn.init.normal_(self.user_embeddings_rho.weight, mean=1/self.embedding_dim, std=0.01)
        torch.nn.init.normal_(self.item_embeddings_rho.weight, mean=1/self.embedding_dim, std=0.01)
        self.rho_activation = torch.nn.Softplus()
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        params = [{'params': self.user_embeddings_rho.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay},
                  {'params': self.item_embeddings_rho.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay}]
        return torch.optim.Adam(params)
    
    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings_rho(user_ids)
        item_embedding = self.item_embeddings_rho(item_ids)
        return self.baseline.dot(user_ids, item_ids), self.rho_activation((user_embedding * item_embedding).sum(1))
    
    def validation_step(self, batch, batch_idx):
        user, rated, targets = batch
        targets = targets[0]
        negative_samples = torch.randint(0, self.n_item, (len(targets),), device=targets.device)
        negatives = self.forward(user.expand(len(targets)), negative_samples)
        positives = self.forward(user.expand(len(targets)), targets)
        loss = self.loss_func(positives, negatives)
        self.log('val_MAP', - loss, prog_bar=True)
    
    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            out = self(user_ids, item_ids)
            return out[0].numpy(), out[1].numpy()
        
    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings_rho(user)
            unc = self.rho_activation((user_embedding * self.item_embeddings_rho.weight).sum(1))
            return self.baseline.predict_user(user), unc
        
    def uncertain_predict_user(self, user, threshold=None):
        with torch.no_grad():
            mean, var = self.predict_user(user)
            return 1 - 0.5 * (1 + torch.erf((0 - torch.tensor(mean)) * var.sqrt().reciprocal() / math.sqrt(2)))


class bprGMF(bprMF):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay)
        self.linear = torch.nn.Linear(self.embedding_dim, 1, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=1, std=0.01)
        self.save_hyperparameters()

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        return self.linear(user_embeddings * item_embeddings).flatten()

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids).numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embeddings = self.user_embeddings(user)
            return self.linear(user_embeddings * self.item_embeddings.weight).flatten()
        
        
class gprGMF(UncertainBPR):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, loss)
        self.linear = torch.nn.Linear(self.embedding_dim, 2, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=1, std=0.01)
        self.rho_activation = torch.nn.Softplus()
        self.save_hyperparameters()

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        out = self.linear(user_embeddings * item_embeddings)
        return out[:, 0], self.rho_activation(out[:, 1])

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            out = self(user_ids, item_ids)
            return out[0].numpy(), out[1].numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            out = self.linear(user_embedding * self.item_embeddings.weight)
            return out[:, 0], self.rho_activation(out[:, 1])
        
    def uncertain_predict_user(self, user, threshold=None):
        with torch.no_grad():
            mean, var = self.predict_user(user)
            return 1 - 0.5 * (1 + torch.erf((0 - torch.tensor(mean)) * var.sqrt().reciprocal() / math.sqrt(2)))


class bprMLP(UncertainBPR):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, loss)
        self.init_net()

    def init_net(self):
        self.linear0 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=False)
        self.linear1 = torch.nn.Linear(self.embedding_dim, int(self.embedding_dim/2), bias=False)
        self.linear2 = torch.nn.Linear(int(self.embedding_dim/2), 2, bias=False)
        torch.nn.init.normal_(self.linear0.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=0.01)
        self.rho_activation = torch.nn.Softplus()
        self.save_hyperparameters()

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        input_embeddings = torch.cat((user_embeddings, item_embeddings), dim=1)
        out = self.linear2(self.linear1(self.linear0(input_embeddings)))
        return out[:, 0], self.rho_activation(out[:, 1])

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            out = self(user_ids, item_ids)
            return out[0].numpy(), out[1].numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user).expand(self.n_item, self.embedding_dim)
            input_embedding = torch.cat((user_embedding, self.item_embeddings.weight), dim=1)
            out = self.linear2(self.linear1(self.linear0(input_embedding)))
            return out[:, 0], self.rho_activation(out[:, 1])
        
    def uncertain_predict_user(self, user, threshold=None):
        with torch.no_grad():
            mean, var = self.predict_user(user)
            return 1 - 0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))
