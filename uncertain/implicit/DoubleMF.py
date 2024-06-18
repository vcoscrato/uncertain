import math
import torch
from ..core import UncertainRecommender
from .base import Implicit, normal_cdf, normal_zero_one_cdf


class AUR_loss():
    
    def __init__(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma
    
    def __call__(self, positives, negatives):
        # Not to forget: S_{ui} = \log\sigma^2_{ui}
        pos_nll = ((1 - positives[0]) ** 2) / positives[1].exp() + self.beta * positives[1] + self.gamma * positives[1] ** 2
        neg_nll = ((0 - negatives[0]) ** 2) / negatives[1].exp() + self.beta * negatives[1] + self.gamma * negatives[1] ** 2
        nll = torch.cat((pos_nll, neg_nll))
        return torch.exp(-nll).mean(), nll.sum()

    
class Pointwise_loss():
    
    def __init__(self, gamma=0):
        self.gamma = gamma
        
    def __call__(self, positives, negatives):
        pos_prob = torch.clamp(1 - normal_zero_one_cdf(-positives[0]/positives[1].exp().sqrt()), min=0, max=1)
        neg_prob = torch.clamp(normal_zero_one_cdf(-negatives[0]/negatives[1].exp().sqrt()), min=0, max=1)
        pos_nll = -torch.log(pos_prob) + self.gamma * positives[1] ** 2
        neg_nll = -torch.log(neg_prob) + self.gamma * negatives[1] ** 2
        nll = torch.cat((pos_nll, neg_nll))
        return torch.exp(-nll).mean(), nll.sum()

    
class Pairwise_loss():
    
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, positives, negatives):
        mean_diff = positives[0] - negatives[0]
        var_sum = positives[1] + negatives[1]
        log_likes = 1 - normal_cdf(mean_diff, var_sum) + self.gamma * (positives[1].log() ** 2 + negatives[1].log() ** 2)
        return log_likes.mean(), -log_likes.log().sum()


class Logit_Normal_loss():

    def __init__(self, gamma=0):
        self.gamma = gamma

    def __call__(self, positives, negatives):
        pos_bern = torch.clamp(1 - normal_zero_one_cdf(-positives[0]/positives[1].exp().sqrt()), min=0, max=1)
        neg_bern = torch.clamp(normal_zero_one_cdf(-negatives[0]/negatives[1].exp().sqrt()), min=0, max=1)
        pos_normal = 0
        pos_nll = -torch.log(pos_prob) + self.gamma * positives[1] ** 2
        neg_nll = -torch.log(neg_prob) + self.gamma * negatives[1] ** 2
        


class JointDoubleMF(Implicit, UncertainRecommender):

    def __init__(self, n_user, n_item, 
                 embedding_dim=128, embedding_dim_var=128, 
                 weight_decay=0, lr=1e-4, n_negatives=None, 
                 loss='Pointwise', beta=1/2, gamma=0, ratio=1):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.embedding_dim_var = embedding_dim_var
        self.weight_decay = weight_decay
        self.lr = lr
        self.n_negatives = n_negatives
        self.beta = beta
        self.gamma = gamma
        self.ratio = ratio
        
        # Init embeddings
        self.user_mean = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_mean = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_mean.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_mean.weight, mean=0, std=0.01)
        
        self.user_var = torch.nn.Embedding(self.n_user, self.embedding_dim_var)
        self.item_var = torch.nn.Embedding(self.n_item, self.embedding_dim_var)
        torch.nn.init.normal_(self.user_var.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_var.weight, mean=0, std=0.01)
        
        if loss == 'Pointwise':
            self.loss = Pointwise_loss(gamma=self.gamma)
        elif loss == 'Pairwise':
            self.loss = Pairwise_loss(gamma=self.gamma)
        elif loss == 'AUR':
            self.loss = AUR_loss(beta=self.beta, gamma=self.gamma)
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        params = [{'params': self.user_mean.parameters(), 'weight_decay': self.weight_decay},
                  {'params': self.item_mean.parameters(), 'weight_decay': self.weight_decay},
                  {'params': self.user_var.parameters()},
                  {'params': self.item_var.parameters()}]
        return torch.optim.Adam(params, lr=self.lr)
    
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
        var = (user_embeddings[1] * item_embeddings[1]).sum(1)
        return mean, var
    
    def uncertain_transform(self, obj):
        mean, var = obj
        if self.ratio is not None:
            return self.ratio * mean + (1-self.ratio) * var.exp().sqrt()
        else:
            return 1 - 0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))


class SequentialDoubleMF(Implicit, UncertainRecommender):
    
    def __init__(self, baseline, embedding_dim_var=1024, 
                 weight_decay=0, lr=1e-4, n_negatives=None, 
                 loss='Pointwise', beta=1/2, gamma=0, ratio=1/2):
        
        super().__init__()
        self.baseline = baseline
        
        self.n_user = baseline.n_user
        self.n_item = baseline.n_item
        self.embedding_dim = baseline.embedding_dim
        self.embedding_dim_var = embedding_dim_var
        self.weight_decay = weight_decay
        self.lr = lr
        self.n_negatives = n_negatives
        self.beta = beta
        self.gamma = gamma
        self.ratio = ratio
            
        # Init embeddings
        self.user_var = torch.nn.Embedding(self.n_user, self.embedding_dim_var)
        self.item_var = torch.nn.Embedding(self.n_item, self.embedding_dim_var)
        torch.nn.init.constant_(self.user_var.weight, 1/math.sqrt(self.embedding_dim_var))
        torch.nn.init.constant_(self.item_var.weight, 1/math.sqrt(self.embedding_dim_var))
        
        if loss == 'Pointwise':
            self.loss = Pointwise_loss(gamma=self.gamma)
        elif loss == 'Pairwise':
            self.loss = Pairwise_loss(gamma=self.gamma)
        elif loss == 'AUR':
            self.loss = AUR_loss(beta=self.beta, gamma=self.gamma)
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        params = [{'params': self.user_var.parameters()}, {'params': self.item_var.parameters()}]
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
    
    def get_user_embeddings(self, user_ids):
        return self.baseline.get_user_embeddings(user_ids), self.user_var(user_ids)
    
    def get_item_embeddings(self, item_ids=None):
        if item_ids is not None:
            item_var = self.item_var(item_ids)
        else:
            item_var = self.item_var.weight
        return self.baseline.get_item_embeddings(item_ids), item_var
    
    def interact(self, user_embeddings, item_embeddings):
        mean = self.baseline.interact(user_embeddings[0], item_embeddings[0])
        var = torch.softplus((user_embeddings[1] * item_embeddings[1]).sum(1))
        return mean, var
    
    def uncertain_transform(self, obj):
        mean, var = obj
        if self.ratio is not None:
            return self.ratio * mean + (1-self.ratio) * var.sqrt()
        else:
            return 1 - 0.5 * (1 + torch.erf((0 - mean) * var.sqrt().reciprocal() / math.sqrt(2)))
