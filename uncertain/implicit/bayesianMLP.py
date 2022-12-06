import torch
from .base import Implicit
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


@variational_estimator
class BayesianMLP(Implicit):

    def __init__(self, n_user, n_item, embedding_dim, lr, num_batches, n_negatives=4, n_hidden=3, 
                 sample_train=5, sample_eval=5, prior_pi=0.5, prior_sigma_1=1, prior_sigma_2=0.01):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.n_negatives = n_negatives
        self.sample_train = sample_train
        self.sample_eval = sample_eval
        self.num_batches = num_batches
        self.prior_pi = prior_pi
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        
        self.n_hidden = n_hidden
        self.ReLU = torch.nn.ReLU()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Init embeddings
        self.user_embeddings = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.01)
        
        # Init linear layers
        self.layers = torch.nn.ModuleList()
        size = self.embedding_dim*2
        while len(self.layers) < self.n_hidden:
            self.layers.append(self.init_layer(size, int(size/2)))
            size = int(size/2)
        self.out_layer = self.init_layer(size, 1)
        
        self.save_hyperparameters()
        
    def init_layer(self, in_size, out_size):
        return BayesianLinear(in_size, out_size, posterior_rho_init=-4.5, prior_pi=self.prior_pi, 
                              prior_sigma_1 = self.prior_sigma_1, prior_sigma_2 = self.prior_sigma_2)
        
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
        return self.out_layer(x).sigmoid().flatten()
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids = batch[:, 0], batch[:, 1]
        
        # Selecting
        user_embeddings = self.get_user_embeddings(user_ids)
        item_embeddings = self.get_item_embeddings(item_ids)
        neg_item_ids = torch.randint(0, self.n_item, (len(user_ids) * self.n_negatives,), device=item_ids.device)
        neg_item_embeddings = self.get_item_embeddings(neg_item_ids)
        n = len(item_ids) + len(neg_item_ids)
        
        # Data likelihood
        loss = 0
        prior_scaling = 2 ** (self.num_batches - batch_idx - 1) / (2 ** self.num_batches - 1)
        # prior_scaling = 1 / self.num_batches
        for _ in range(self.sample_train):
            loss -= self.interact(user_embeddings, item_embeddings).log().sum() / n
            loss -= (1 - self.interact(user_embeddings.repeat(self.n_negatives, 1), neg_item_embeddings)).log().sum() / n
            loss += prior_scaling * self.nn_kl_divergence()
        return loss / self.sample_train
    
    def forward(self, user_ids, item_ids=None):
        user_embeddings = self.get_user_embeddings(user_ids)
        item_embeddings = self.get_item_embeddings(item_ids)
        pred = torch.vstack([self.interact(user_embeddings, item_embeddings) for _ in range(self.sample_eval)])
        return pred.mean(0), pred.var(0)



@variational_estimator
class FullBayesianGER(UncertainImplicit):

    def __init__(self, n_user, n_item, embedding_dim, lr, num_batches, n_negatives=4, n_hidden=3, 
                 sample_train=5, sample_eval=5, prior_pi=0.5, prior_sigma_1=1, prior_sigma_2=0.01):
        
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.n_negatives = n_negatives
        self.sample_train = sample_train
        self.sample_eval = sample_eval
        self.num_batches = num_batches
        self.prior_pi = prior_pi
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        
        self.n_hidden = n_hidden
        self.ReLU = torch.nn.ReLU()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Init embeddings
        self.user_embeddings = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.01)
        
        # Init linear layers
        self.layers = torch.nn.ModuleList()
        size = self.embedding_dim*2
        while len(self.layers) < self.n_hidden:
            self.layers.append(self.init_layer(size, int(size/2)))
            size = int(size/2)
        self.out_layer = self.init_layer(size, 1)
        
        self.save_hyperparameters()
        
    def init_layer(self, in_size, out_size):
        return BayesianLinear(in_size, out_size, posterior_rho_init=-4.5, prior_pi=self.prior_pi, 
                              prior_sigma_1 = self.prior_sigma_1, prior_sigma_2 = self.prior_sigma_2)
        
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
        return self.out_layer(x).sigmoid().flatten()
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids = batch[:, 0], batch[:, 1]
        
        # Selecting
        user_embeddings = self.get_user_embeddings(user_ids)
        item_embeddings = self.get_item_embeddings(item_ids)
        neg_item_ids = torch.randint(0, self.n_item, (len(user_ids) * self.n_negatives,), device=item_ids.device)
        neg_item_embeddings = self.get_item_embeddings(neg_item_ids)
        n = len(item_ids) + len(neg_item_ids)
        
        # Data likelihood
        loss = 0
        prior_scaling = 2 ** (self.num_batches - batch_idx - 1) / (2 ** self.num_batches - 1)
        # prior_scaling = 1 / self.num_batches
        for _ in range(self.sample_train):
            loss -= self.interact(user_embeddings, item_embeddings).log().sum() / n
            loss -= (1 - self.interact(user_embeddings.repeat(self.n_negatives, 1), neg_item_embeddings)).log().sum() / n
            loss += prior_scaling * self.nn_kl_divergence()
        return loss / self.sample_train
    
    def forward(self, user_ids, item_ids=None):
        user_embeddings = self.get_user_embeddings(user_ids)
        item_embeddings = self.get_item_embeddings(item_ids)
        pred = torch.vstack([self.interact(user_embeddings, item_embeddings) for _ in range(self.sample_eval)])
        return pred.mean(0), pred.var(0)
