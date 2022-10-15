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
