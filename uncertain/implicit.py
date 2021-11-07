import math
import torch
import numpy as np
from scipy.stats import beta
from .core import FactorizationModel, VanillaRecommender, UncertainRecommender


def cross_entropy(positive, negative):
    return - torch.cat((positive.sigmoid(), 1 - negative.sigmoid())).mean()


def bpr(positive, negative):
    return - (positive - negative).sigmoid().log().mean()


def abpr(positive, negative):
    x = positive[0] - negative[0]
    rho = positive[1] + negative[1]
    return torch.sigmoid(x / rho).log().mean()


def gpr(positive, negative):
    x = positive[0] - negative[0]
    rho = positive[1] + negative[1]
    return (0.5 * (1 + torch.erf((x / torch.sqrt(2 * rho))))).log().mean()


class Implicit:

    def get_negative_prediction(self, batch):
        user_ids = torch.randint(0, self.n_user, (len(batch) * self.n_negative,), device=batch.device)
        item_ids = torch.randint(0, self.n_item, (len(batch) * self.n_negative,), device=batch.device)
        negative_prediction = self.forward(user_ids, item_ids)
        return negative_prediction

    def training_step(self, batch, batch_idx):
        output = self.forward(batch[:, 0], batch[:, 1])
        loss = self.loss_func(output, self.get_negative_prediction(batch[:, 0]))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch[:, 0], batch[:, 1])
        loss = self.loss_func(output, self.get_negative_prediction(batch[:, 0]))
        self.log('val_loss', loss, prog_bar=True)


class logMF(Implicit, FactorizationModel, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, n_negative):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, cross_entropy, n_negative)


class bprMF(Implicit, FactorizationModel, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, bpr, 1)


class CAMF(Implicit, FactorizationModel, UncertainRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, n_negative):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, None, n_negative)
        self.linear = torch.nn.Linear(embedding_dim, 2, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=1, std=0.01)
        self.activation = torch.nn.Softplus()
        # self.quantile = 0.5
        # self.penalty = penalty

    def configure_optimizers(self):
        params = [{'params': self.user_embeddings.parameters(), 'weight_decay': self.weight_decay},
                  {'params': self.item_embeddings.parameters(), 'weight_decay': self.weight_decay},
                  {'params': self.linear.parameters(), 'weight_decay': 0}]
        return torch.optim.Adam(params, lr=self.lr)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        dot = user_embedding * item_embedding
        alpha_beta = self.activation(self.linear(dot))
        return alpha_beta

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            alpha_beta = self(user_ids, item_ids) # .numpy()
            return self.expectation(alpha_beta).numpy(), self.variance(alpha_beta).numpy()
            # return beta.ppf(self.quantile, a=dists[:, 0], b=dists[:, 1])

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            dot = user_embedding * self.item_embeddings.weight
            alpha_beta = self.activation(self.linear(dot))
            return self.expectation(alpha_beta).numpy(), self.variance(alpha_beta).numpy()

    @staticmethod
    def expectation(alpha_beta):
        return torch.divide(alpha_beta[:, 0], alpha_beta.sum(1)).flatten()

    @staticmethod
    def variance(alpha_beta):
        sum_params = alpha_beta.sum(1)
        return torch.divide(alpha_beta.prod(1), (sum_params**2 * (sum_params + 1))).flatten()

    def loss_func(self, positive_params, negative_params, train=True):
        prob_positive = self.expectation(positive_params)
        prob_negative = self.expectation(negative_params)
        loss = torch.cat((prob_positive, 1 - prob_negative))

        '''
        if train:
            penalty_pos = (positive_params[:, 0] - positive_params[:, 1]).square().sum()
            penalty_neg = (negative_params[:, 0] - negative_params[:, 1]).square().sum()
            penalty = self.penalty * (penalty_pos + penalty_neg)
            return - loss.log().mean() + penalty
        else:
            return loss.mean()
        '''
        return - loss.log().mean()


class GMF(Implicit, FactorizationModel, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, n_negative):
        super().__init__(n_user, n_item, embedding_dim, lr, 0, cross_entropy, n_negative)
        self.init_net()

    def init_net(self):
        self.linear = torch.nn.Linear(self.embedding_dim, 1, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        return self.linear(user_embeddings * item_embeddings)

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids).numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            return self.linear(user_embedding * self.item_embeddings.weight).numpy()


class MLP(Implicit, FactorizationModel, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, n_negative):
        super().__init__(n_user, n_item, embedding_dim, lr, 0, cross_entropy, n_negative)
        self.init_net()

    def init_net(self):
        self.linear0 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=False)
        self.linear1 = torch.nn.Linear(self.embedding_dim, int(self.embedding_dim/2), bias=False)
        self.linear2 = torch.nn.Linear(int(self.embedding_dim/2), 1, bias=False)
        torch.nn.init.normal_(self.linear0.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=0.01)

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        input_embeddings = torch.cat((user_embeddings, item_embeddings), dim=1)
        return self.linear2(self.linear1(self.linear0(input_embeddings)))

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids).numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user).expand(self.n_item, self.embedding_dim)
            input_embedding = torch.cat((user_embedding, self.item_embeddings.weight), dim=1)
            return self.linear2(self.linear1(self.linear0(input_embedding))).numpy()






class TwoWayMF(UncertainRecommender):

    def __init__(self, relevance, embedding_dim, lr, weight_decay, loss=abpr):
        super().__init__()
        self.relevance = relevance
        self.n_user = self.relevance.n_user
        self.n_item = self.relevance.n_item
        self.embedding_dim = embedding_dim
        self.user_embeddings = torch.nn.Embedding(self.n_user, embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, embedding_dim)
        torch.nn.init.constant_(self.user_embeddings.weight, 1)
        torch.nn.init.constant_(self.item_embeddings.weight, 1)
        self.var_activation = torch.nn.Softplus()
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_func = loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([self.user_embeddings.weight, self.item_embeddings.weight],
                                     lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        return self.relevance.predict(user_ids, item_ids), self.var_activation((user_embedding * item_embedding).sum(1))








class BaseMF(FactorizationModel):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay)

    def dot(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        return (user_embedding * item_embedding).sum(1)


class blabla(BaseMF):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, compose, init_value):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay)
        self.user_gammas = torch.nn.Embedding(self.n_user, 1)
        self.item_gammas = torch.nn.Embedding(self.n_item, 1)
        torch.nn.init.constant_(self.user_gammas.weight, init_value)
        torch.nn.init.constant_(self.item_gammas.weight, init_value)
        self.unc_activation = torch.nn.Softplus()
        self.compose = compose

    def unc(self, user_ids, item_ids):
        user_gamma = self.user_gammas(user_ids)
        item_gamma = self.item_gammas(item_ids)
        return self.unc_activation(self.compose(user_gamma, item_gamma)).flatten()

