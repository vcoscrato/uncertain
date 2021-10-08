import torch
import numpy as np
from .core import FactorizationModel, VanillaRecommender, UncertainRecommender


def mse(predicted, observed):
    return ((observed - predicted) ** 2).mean()


def gaussian(predicted, observed):
    mean, variance = predicted
    return (((observed - mean) ** 2) / variance).mean() + torch.log(variance).mean()


def max_prob(predicted, observed):
    return - predicted[range(len(predicted)), observed.long()].log().mean()


class Explicit:

    def training_step(self, batch, batch_idx):
        output = self.forward(batch[:, 0].long(), batch[:, 1].long())
        loss = self.loss_func(output, batch[:, 2])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch[:, 0].long(), batch[:, 1].long())
        loss = self.loss_func(output, batch[:, 2])
        self.log('val_loss', loss, prog_bar=True)


class MF(Explicit, FactorizationModel, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, mse)


class CPMF(Explicit, FactorizationModel, UncertainRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay_MF, weight_decay_gammas):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay_MF, gaussian)
        self.init_gammas()
        self.weight_decay_MF = weight_decay_MF
        self.weight_decay_gammas = weight_decay_gammas

    def init_gammas(self):
        self.user_gammas = torch.nn.Embedding(self.n_user, 1)
        self.item_gammas = torch.nn.Embedding(self.n_item, 1)
        torch.nn.init.constant_(self.user_gammas.weight, 1)
        torch.nn.init.constant_(self.item_gammas.weight, 1)
        self.var_activation = torch.nn.Softplus()

    def configure_optimizers(self):
        params = [{'params': self.user_embeddings.parameters(), 'weight_decay': self.weight_decay_MF},
                  {'params': self.item_embeddings.parameters(), 'weight_decay': self.weight_decay_MF},
                  {'params': self.user_gammas.parameters(), 'weight_decay': self.weight_decay_gammas},
                  {'params': self.item_gammas.parameters(), 'weight_decay': self.weight_decay_gammas}]
        return torch.optim.Adam(params, lr=self.lr)

    def forward(self, user_ids, item_ids):
        mean = self.dot(user_ids, item_ids)
        user_gamma = self.user_gammas(user_ids)
        item_gamma = self.item_gammas(item_ids)
        var = self.var_activation(user_gamma + item_gamma).flatten()
        return mean, var

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            mean, var = self(user_ids, item_ids)
            return mean.numpy(), var.numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            mean = (user_embedding * self.item_embeddings.weight).sum(1)
            user_gamma = self.user_gammas(user)
            var = self.var_activation(user_gamma + self.item_gammas.weight).flatten()
            return mean.numpy(), var.numpy()


class OrdRec(Explicit, FactorizationModel, UncertainRecommender):

    def __init__(self, n_user, n_item, score_labels, embedding_dim, lr, weight_decay_MF, weight_decay_step):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay_MF, max_prob)
        self.score_labels = score_labels
        self.init_step()
        self.weight_decay_MF = weight_decay_MF
        self.weight_decay_step = weight_decay_step

    def init_step(self):
        self.user_step = torch.nn.Embedding(self.n_user, len(self.score_labels) - 1)
        self.item_step = torch.nn.Embedding(self.n_item, len(self.score_labels) - 1)
        torch.nn.init.constant_(self.user_step.weight, 0)
        torch.nn.init.constant_(self.item_step.weight, 0)

    def configure_optimizers(self):
        params = [{'params': self.user_embeddings.parameters(), 'weight_decay': self.weight_decay_MF},
                  {'params': self.item_embeddings.parameters(), 'weight_decay': self.weight_decay_MF},
                  {'params': self.user_step.parameters(), 'weight_decay': self.weight_decay_step},
                  {'params': self.item_step.parameters(), 'weight_decay': self.weight_decay_step}]
        return torch.optim.Adam(params, lr=self.lr)

    def forward(self, user_ids, item_ids):
        y = self.dot(user_ids, item_ids).reshape(-1, 1)
        steps = torch.exp(self.user_step(user_ids) + self.item_step(item_ids)).cumsum(1)
        # steps[:, 0] = torch.log(steps[:, 0])
        distribution = torch.sigmoid(steps - y)
        one = torch.ones((len(distribution), 1), device=distribution.device)
        distribution = torch.cat((distribution, one), 1)
        distribution[:, 1:] -= distribution[:, :-1].clone()
        return distribution

    def _summarize(self, distributions):
        mean = (distributions * self.score_labels).sum(1)
        var = (distributions * self.score_labels ** 2).sum(1) - mean ** 2
        return mean.numpy(), var.numpy()

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            distributions = self.forward(user_ids, item_ids)
            return self._summarize(distributions)

    def predict_user(self, user):
        with torch.no_grad():
            y = (self.user_embeddings(user) * self.item_embeddings.weight).sum(1).reshape(-1, 1)
            steps = torch.exp(self.user_step(user) + self.item_step.weight).cumsum(1)
            # steps[:, 0] = torch.log(steps[:, 0])
            distributions = torch.sigmoid(steps - y)
            one = torch.ones((len(distributions), 1), device=distributions.device)
            distributions = torch.cat((distributions, one), 1)
            distributions[:, 1:] -= distributions[:, :-1].clone()
            return self._summarize(distributions)
