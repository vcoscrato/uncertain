import torch
import pytorch_lightning as pl
from uncertain.core import Recommender
from uncertain.layers import ZeroEmbedding, ScaledEmbedding


class ExplicitFactorizationModel(pl.LightningModule, Recommender):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__()
        self.pass_args(interactions)
        self.user_embeddings = ScaledEmbedding(self.num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(self.num_items, embedding_dim)
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

    @property
    def is_uncertain(self):
        return False

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        dot = (user_embedding * item_embedding).sum(1)
        return dot

    def loss_func(self, predicted_ratings, observed_ratings):
        return ((observed_ratings - predicted_ratings) ** 2).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def get_negative_prediction(self, users):
        sampled_items = torch.randint(0, self.num_items, (len(users),))
        negative_prediction = self.forward(users, sampled_items)
        return negative_prediction

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

    def predict(self, user_id):
        with torch.no_grad():
            item_ids = torch.arange(self.num_items, device=self.device)
            user_ids = torch.full_like(item_ids, user_id)
            return self.forward(user_ids, item_ids)

    def get_item_similarity(self, item_id, candidate_ids=None):
        with torch.no_grad():
            item_var = self.item_embeddings(torch.tensor(item_id, device=self.device))
            if candidate_ids is None:
                candidate_ids = torch.arange(self.num_items, device=self.device)
            candidates_var = self.item_embeddings(candidate_ids)
            return torch.cosine_similarity(item_var, candidates_var, dim=-1)


class ExplicitCPMF(pl.LightningModule, Recommender):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__()
        self.pass_args(interactions)
        self.user_embeddings = ScaledEmbedding(self.num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(self.num_items, embedding_dim)
        self.user_gammas = ScaledEmbedding(self.num_users, 1)
        self.item_gammas = ScaledEmbedding(self.num_items, 1)
        self.var_activation = torch.nn.Softplus()
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

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

    def loss_func(self, predicted_ratings, observed_ratings):
        mean, variance = predicted_ratings
        return (((observed_ratings - mean) ** 2) / variance).mean() + torch.log(variance).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def get_negative_prediction(self, users):
        sampled_items = torch.randint(0, self.num_items, (len(users),))
        negative_prediction = self.forward(users, sampled_items)
        return negative_prediction

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

    def predict(self, user_id):
        with torch.no_grad():
            item_ids = torch.arange(self.num_items, device=self.device)
            user_ids = torch.full_like(item_ids, user_id)
            return self.forward(user_ids, item_ids)

    def get_item_similarity(self, item_id, candidate_ids=None):
        with torch.no_grad():
            item_var = self.item_embeddings(torch.tensor(item_id, device=self.device))
            if candidate_ids is None:
                candidate_ids = torch.arange(self.num_items, device=self.device)
            candidates_var = self.item_embeddings(candidate_ids)
            return torch.cosine_similarity(item_var, candidates_var, dim=-1)


class OrdRec(pl.LightningModule, Recommender):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__()
        self.pass_args(interactions)
        self.user_embeddings = ScaledEmbedding(self.num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(self.num_items, embedding_dim)
        self.user_betas = ZeroEmbedding(self.num_users, len(self.score_labels) - 1)
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

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

    def loss_func(self, predicted_ratings, observed_ratings):
        return -predicted_ratings[range(len(-predicted_ratings)), observed_ratings].log().mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

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

    def predict(self, user_id, return_distribution=False):
        with torch.no_grad():
            item_ids = torch.arange(self.num_items, device=self.device)
            user_ids = torch.full_like(item_ids, user_id)
            distribution = self.forward(user_ids, item_ids)
            if return_distribution:
                return distribution
            else:
                mean = (distribution * self.score_labels).sum(1)
                var = ((distribution * self.score_labels ** 2).sum(1) - mean ** 2).abs()
                return mean, var

    def get_item_similarity(self, item_id, candidate_ids=None):
        with torch.no_grad():
            item_var = self.item_embeddings(torch.tensor(item_id, device=self.device))
            if candidate_ids is None:
                candidate_ids = torch.arange(self.num_items, device=self.device)
            candidates_var = self.item_embeddings(candidate_ids)
            return torch.cosine_similarity(item_var, candidates_var, dim=-1)


class ImplicitFactorizationModel(pl.LightningModule, Recommender):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__()
        self.pass_args(interactions)
        self.user_embeddings = ScaledEmbedding(self.num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(self.num_items, embedding_dim)
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

    @property
    def is_uncertain(self):
        return False

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        dot = (user_embedding * item_embedding).sum(1)
        return dot
    
    def loss_func(self, predicted_ratings, predicted_negative):
        positive = (1.0 - torch.sigmoid(predicted_ratings))
        negative = torch.sigmoid(predicted_negative)
        return torch.cat((positive, negative)).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def get_negative_prediction(self, users):
        sampled_items = torch.randint(0, self.num_items, (len(users),), device=self.device)
        negative_prediction = self.forward(users, sampled_items)
        return negative_prediction

    def training_step(self, train_batch, batch_idx):
        users, items = train_batch
        output = self.forward(users, items)
        loss = self.loss_func(output, predicted_negative=self.get_negative_prediction(users))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        users, items = val_batch
        output = self.forward(users, items)
        loss = self.loss_func(output, predicted_negative=self.get_negative_prediction(users))
        self.log('val_loss', loss)

    def predict(self, user_id):
        with torch.no_grad():
            item_ids = torch.arange(self.num_items, device=self.device)
            user_ids = torch.full_like(item_ids, user_id)
            return self.forward(user_ids, item_ids)

    def get_item_similarity(self, item_id, candidate_ids=None):
        with torch.no_grad():
            item_var = self.item_embeddings(torch.tensor(item_id, device=self.device))
            if candidate_ids is None:
                candidate_ids = torch.arange(self.num_items, device=self.device)
            candidates_var = self.item_embeddings(candidate_ids)
            return torch.cosine_similarity(item_var, candidates_var, dim=-1)


class ImplicitCPMF(pl.LightningModule, Recommender):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__()
        self.pass_args(interactions)
        self.user_embeddings = ScaledEmbedding(self.num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(self.num_items, embedding_dim)
        self.user_gammas = ScaledEmbedding(self.num_users, 1)
        self.item_gammas = ScaledEmbedding(self.num_items, 1)
        self.var_activation = torch.nn.Softplus()
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

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
    
    def loss_func(self, predicted_ratings, predicted_negative):
        mean, variance = predicted_ratings
        positive = (((1.0 - mean) ** 2) / variance).mean() + torch.log(variance).mean()
        mean, variance = predicted_negative
        negative = ((mean ** 2) / variance).mean() + torch.log(variance).mean()
        return (positive + negative) / 2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def get_negative_prediction(self, users):
        sampled_items = torch.randint(0, self.num_items, (len(users),), device=self.device)
        negative_prediction = self.forward(users, sampled_items)
        return negative_prediction

    def training_step(self, train_batch, batch_idx):
        users, items = train_batch
        output = self.forward(users, items)
        loss = self.loss_func(output, predicted_negative=self.get_negative_prediction(users))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        users, items = val_batch
        output = self.forward(users, items)
        loss = self.loss_func(output, predicted_negative=self.get_negative_prediction(users))
        self.log('val_loss', loss)

    def predict(self, user_id):
        with torch.no_grad():
            item_ids = torch.arange(self.num_items, device=self.device)
            user_ids = torch.full_like(item_ids, user_id)
            return self.forward(user_ids, item_ids)

    def get_item_similarity(self, item_id, candidate_ids=None):
        with torch.no_grad():
            item_var = self.item_embeddings(torch.tensor(item_id, device=self.device))
            if candidate_ids is None:
                candidate_ids = torch.arange(self.num_items, device=self.device)
            candidates_var = self.item_embeddings(candidate_ids)
            return torch.cosine_similarity(item_var, candidates_var, dim=-1)
