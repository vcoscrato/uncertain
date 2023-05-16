import math
import torch
from pytorch_lightning import LightningModule
from ..core import VanillaRecommender


# Losses
def BCE(positive_scores, negative_scores):
    '''
    Binary cross entropy loss.
    Returns the average likelihood and the NLL.
    '''
    probs = torch.cat((positive_scores.sigmoid(), 1 - negative_scores.sigmoid()))
    return probs.mean(), - probs.log().sum()


def Flat(positive_scores, negative_scores):
    '''
    Flat probabilistic loss.
    Returns the average likelihood and the NLL.
    '''
    probs = torch.cat((positive_scores.sigmoid(), 1 - negative_scores.sigmoid()))
    return probs.mean(), (1-probs).sum()
    

def BPR(positive_scores, negative_scores):
    '''
    Bayesian Personalized Ranking opt.
    Returns the average likelihood and the NLL.
    '''
    probs = torch.sigmoid(positive_scores - negative_scores)
    return probs.mean(), - probs.log().sum()


def normal_cdf(mean, var, value=0):
    '''
    Returns the P(N(mean, var) <= value)
    '''
    return 0.5 * (1 + torch.erf((value - mean) * var.sqrt().reciprocal() / math.sqrt(2)))


def MSE(positive_scores, negative_scores):
    '''
    Mean squared error loss.
    Returns the average likelihood and the NLL.
    '''
    positives_loglike = -((1 - positive_scores) ** 2) / 2
    negatives_loglike = -((0 - negative_scores) ** 2) / 2
    log_likes = torch.cat((positives_loglike, negatives_loglike))
    return torch.exp(log_likes).mean(), -log_likes.sum()


def Pointwise(positive_scores, negative_scores):
    '''
    Gaussian Binary Ranking. Learns to max P(Yui > 0) * I(Yui=1) + P(Yui < 0) * I(Yui=0). Y_ui~N(\mu_ui, 1)
    Returns the average likelihood and the NLL.
    '''
    positives_like = 1 - normal_cdf(positive_scores, torch.tensor(1))
    negatives_like = normal_cdf(negative_scores, torch.tensor(1))
    likelihoods = torch.cat((positives_like, negatives_like))
    return likelihoods.mean(), -likelihoods.log().sum()


def Pairwise(positive_scores, negative_scores):
    '''
    Gaussian Pairwise Ranking. P(i >_u j) = P(Y_ui - Y_uj > 0). Y_ui~N(\mu_ui, 1)
    Returns the average likelihood and the NLL.
    '''
    mean_diff = positive_scores - negative_scores
    likelihoods = 1 - normal_cdf(mean_diff, torch.tensor(2)) #var(Yui + Yuj) = 2
    return likelihoods.mean(), -likelihoods.log().sum()


# Base class for implicit recommenders
class Implicit(LightningModule):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, user_ids, item_ids=None):
        user_embeddings = self.get_user_embeddings(user_ids)
        item_embeddings = self.get_item_embeddings(item_ids)
        return self.interact(user_embeddings, item_embeddings)
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids = batch[:, 0], batch[:, 1]
            
        # Selecting
        user_embeddings = self.get_user_embeddings(user_ids)
        item_embeddings = self.get_item_embeddings(item_ids)
        if self.n_negatives is None or self.n_negatives == 1:
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
        
        # Loss
        self.log('loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        user, rated, targets = batch
        if hasattr(self, 'uncertain_transform'):
            rec = self.uncertain_rank(user, ignored_item_ids=rated[0], top_n=5)[0]
        else:
            rec = self.rank(user, ignored_item_ids=rated[0], top_n=5)[0]
        hits = torch.isin(rec, targets[0], assume_unique=True)
        n_hits = hits.cumsum(0)
        if n_hits[-1] > 0:
            precision = n_hits / torch.arange(1, 6, device=n_hits.device)
            self.log('val_MAP', torch.sum(precision * hits) / n_hits[-1], prog_bar=True)
        else:
            self.log('val_MAP', torch.tensor(0, dtype=torch.float32), prog_bar=True)


# Simple implicit matrix factorization (Dot product)
class MF(Implicit, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim=128, lr=1e-4, weight_decay=0, n_negatives=None, loss='BCE'):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.n_negatives = n_negatives
        self.weight_decay = weight_decay
    
        # Init embeddings
        self.user_embeddings = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.1)
        
        if loss == 'BCE':
            self.loss = BCE
        elif loss == 'Flat':
            self.loss = Flat
        elif loss == 'BPR': 
            self.loss = BPR
        elif loss == 'MSE':
            self.loss = MSE
        elif loss == 'Pointwise':
            self.loss = Pointwise
        elif loss == 'Pairwise':
            self.loss = Pairwise
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def get_user_embeddings(self, user_ids):
        return self.user_embeddings(user_ids)
    
    def get_item_embeddings(self, item_ids=None):
        if item_ids is not None:
            return self.item_embeddings(item_ids)
        else:
            return self.item_embeddings.weight
    
    def interact(self, user_embeddings, item_embeddings):
        return (user_embeddings * item_embeddings).sum(1)
    

# MLP-based latent factor model (From NeuMF)
class MLP(Implicit, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim=128, lr=1e-4, dropout=0, n_hidden=3, n_negatives=None, loss='BCE'):
        
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
        
        if loss == 'BCE':
            self.loss = BCE
        elif loss == 'Flat':
            self.loss = Flat
        elif loss == 'BPR': 
            self.loss = BPR
        elif loss == 'MSE':
            self.loss = MSE
        elif loss == 'Pointwise':
            self.loss = Pointwise
        elif loss == 'Pairwise':
            self.loss = Pairwise
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
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
        return self.out_layer(x).flatten()


'''       

def training_step_user_based(self, batch, batch_idx):

    device = self.user_embeddings.weight.device

    user_embeddings = torch.cat([self.get_user_embeddings(torch.tensor(u[0], device=device)).repeat(len(u[1])) for u in batch])
    neg_user_embeggings = torch.cat([self.get_user_embeddings(torch.tensor(u[0], device=device)).repeat(self.n_item * self.negative_ratio) for u in batch])

    item_embeddings = torch.cat([self.get_item_embeddings(torch.tensor(u[1], device=device)) for u in batch])
    n_samples = (len(batch) * self.n_item * self.negative_ratio,)
    neg_item_ids = torch.cat([torch.randint(0, self.n_item, n_samples, device=item_ids.device)])
    neg_item_embeddings = self.get_item_embeddings(neg_item_ids)

    print(user_embeddings.shape, neg_user_embeddings.shape, item_embeddings.shape, n_samples, neg_item_ids.shape, neg_item_embeddings.shape)

    # Data likelihood
    pos_scores = self.interact(user_embeddings, item_embeddings)
    neg_scores = self.interact(neg_user_embeddings, neg_item_embeddings)

    train_likelihood, loss = self.loss(pos_scores, neg_scores)
    self.log('train_likelihood', train_likelihood, prog_bar=True)

    # Loss
    self.log('loss', loss)
    return loss
'''