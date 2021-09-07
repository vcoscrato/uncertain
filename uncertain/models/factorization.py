import torch
import pytorch_lightning as pl
from uncertain.core import Recommender
from uncertain.layers import ZeroEmbedding, ScaledEmbedding
from uncertain.losses import mse_loss, gaussian_loss, max_prob_loss, cross_entropy_loss, bpr_loss


class FactorizationModel(pl.LightningModule, Recommender):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__()
        self.pass_args(interactions)

        self.user_embeddings = ScaledEmbedding(self.num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(self.num_items, embedding_dim)

        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def predict(self, user_id):
        item_ids = torch.arange(self.num_items)
        user_ids = torch.full_like(item_ids, user_id)
        return self(user_ids, item_ids)

    def get_user_similarity(self, users, candidates=None):
        users_var = self.user_embeddings(users)
        if candidates is None:
            candidates_var = self.user_embeddings.weight
        else:
            candidates_var = self.user_embeddings(candidates)
        return torch.cosine_similarity(users_var, candidates_var, dim=-1)

    def get_item_similarity(self, items, candidates=None):
        items_var = self.item_embeddings(items)
        if candidates is None:
            candidates_var = self.item_embeddings.weight
        else:
            candidates_var = self.item_embeddings(candidates)
        return torch.cosine_similarity(items_var, candidates_var, dim=-1)


class Explicit(object):

    def training_step(self, train_batch, batch_idx):
        users, items, ratings = train_batch
        output = self.forward(users, items)
        loss = self.loss_func(output, ratings)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        users, items, ratings = val_batch
        output = self.forward(users, items)
        loss = self.loss_func(output, ratings)
        self.log('val_loss', loss)


class Implicit(object):
    
    def get_negative_prediction(self, users):
        sampled_items = torch.randint(0, self.num_items, (len(users),), device=self.device)
        negative_prediction = self.forward(users, sampled_items)
        return negative_prediction

    def training_step(self, train_batch, batch_idx):
        users, items = train_batch
        output = self.forward(users, items)
        loss = self.loss_func(output, self.get_negative_prediction(users))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        users, items = val_batch
        output = self.forward(users, items)
        loss = self.loss_func(output, self.get_negative_prediction(users))
        self.log('val_loss', loss)


class FunkSVD(Explicit, FactorizationModel):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__(interactions, embedding_dim, lr, batch_size, weight_decay)
        self.loss_func = mse_loss

    @property
    def is_uncertain(self):
        return False

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        dot = (user_embedding * item_embedding).sum(1)
        return dot


class CPMF(Explicit, FactorizationModel):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__(interactions, embedding_dim, lr, batch_size, weight_decay)
        self.loss_func = gaussian_loss
        self.user_gammas = ScaledEmbedding(self.num_users, 1)
        self.item_gammas = ScaledEmbedding(self.num_items, 1)
        self.var_activation = torch.nn.Softplus()

    @property
    def is_uncertain(self):
        return True

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        dot = (user_embedding * item_embedding).sum(1)
        user_gamma = self.user_gammas(user_ids)
        item_gamma = self.item_gammas(item_ids)
        var = self.var_activation(user_gamma + item_gamma)
        return dot, var


class OrdRec(Explicit, FactorizationModel):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__(interactions, embedding_dim, lr, batch_size, weight_decay)
        self.loss_func = max_prob_loss
        self.user_betas = ZeroEmbedding(self.num_users, len(self.score_labels) - 1)

    @property
    def is_uncertain(self):
        return True

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        y = (user_embedding * item_embedding).sum(1).reshape(-1, 1)
        user_beta = self.user_betas(user_ids)
        user_beta[:, 1:] = torch.exp(user_beta[:, 1:])
        user_distribution = torch.div(1, 1 + torch.exp(y - user_beta.cumsum(1)))
        one = torch.ones((len(user_distribution), 1), device=user_beta.device)
        user_distribution = torch.cat((user_distribution, one), 1)
        user_distribution[:, 1:] -= user_distribution[:, :-1].clone()
        return user_distribution

    def predict(self, user_id, return_distribution=False):
        item_ids = torch.arange(self.num_items, device=self.device)
        user_ids = torch.full_like(item_ids, user_id)
        distribution = self.forward(user_ids, item_ids)
        if return_distribution:
            return distribution
        else:
            mean = (distribution * self.score_labels).sum(1)
            var = ((distribution * self.score_labels ** 2).sum(1) - mean ** 2).abs()
            return mean, var


class ImplicitMF(Implicit, FactorizationModel):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay, loss='bpr'):
        super().__init__(interactions, embedding_dim, lr, batch_size, weight_decay)
        if loss == 'bpr':
            self.loss_func = bpr_loss
        elif loss == 'cross_entropy':
            self.loss_func = cross_entropy_loss
        else:
            raise AttributeError('loss should be one of ["bpr", "cross_entropy"].')

    @property
    def is_uncertain(self):
        return False

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        dot = (user_embedding * item_embedding).sum(1)
        return torch.sigmoid(dot)


class GMF(Explicit, FactorizationModel):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__(interactions, embedding_dim, lr, batch_size, weight_decay)
        self.linear = torch.nn.Linear(embedding_dim, 1, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        self.loss_func = mse_loss

    @property
    def is_uncertain(self):
        return False

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        dot = user_embedding * item_embedding
        return self.linear(dot).flatten()


class GaussianGMF(Explicit, FactorizationModel):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__(interactions, embedding_dim, lr, batch_size, weight_decay)
        self.var_activation = torch.nn.Softplus()
        self.linear = torch.nn.Linear(embedding_dim, 2, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        self.loss_func = gaussian_loss

    @property
    def is_uncertain(self):
        return True

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        dot = self.linear(user_embedding * item_embedding)
        return dot[:, 0], self.var_activation(dot[:, 1])
