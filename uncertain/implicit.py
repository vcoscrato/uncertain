import torch
import pytorch_lightning as pl
from uncertain.core import Recommender
from scipy.stats import beta


def cross_entropy(positive, negative):
    return - torch.cat((positive.sigmoid(), 1 - negative.sigmoid())).log().mean()


def bpr(positive, negative):
    return - (positive - negative).sigmoid().log().mean()


class FactorizationModel(pl.LightningModule, Recommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss, n_negative=1):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.init_embeddings()
        if loss == 'bpr':
            self.loss_func = bpr
            self.n_negative = 1
        elif loss == 'cross_entropy':
            self.loss_func = cross_entropy
            self.n_negative = n_negative
        else:
            self.loss_func = loss

    def init_embeddings(self):
        self.user_embeddings = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.01)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        return (user_embedding * item_embedding).sum(1)

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

    def test_step(self, batch, batch_idx):
        # TODO
        return 0

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids)

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            return (user_embedding * self.item_embeddings.weight).sum(1)

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



class UncertainMF():

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, compose, init_value, loss=ABPR()):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, compose, init_value)
        self.loss_func = loss

    def forward(self, user_ids, item_ids):
        return self.dot(user_ids, item_ids), self.unc(user_ids, item_ids)


class ImplicitGMF(BaseGMF):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss=BPR()):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay)
        self.loss_func = loss

    def forward(self, user_ids, item_ids):
        return self.linear(self.dot(user_ids, item_ids)).flatten()


class CAMF(FactorizationModel):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, n_negative, penalty):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay)
        self.linear = torch.nn.Linear(embedding_dim, 2, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=1, std=0.01)
        self.activation = torch.nn.Softplus()
        self.quantile = 0.5
        self.n_negative = n_negative
        self.penalty = penalty

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        dot = user_embedding * item_embedding
        alpha_beta = self.activation(self.linear(dot))
        return alpha_beta

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            dists = self(user_ids, item_ids).numpy()
        return beta.ppf(self.quantile, a=dists[:, 0], b=dists[:, 1])

    @staticmethod
    def expectation(alpha_beta):
        return torch.divide(alpha_beta[:, 0], alpha_beta.sum(1)).flatten()

    def get_negative_prediction(self, batch):
        user_ids = torch.randint(0, self.n_user, (len(batch) * self.n_negative,), device=batch.device)
        item_ids = torch.randint(0, self.n_item, (len(batch) * self.n_negative,), device=batch.device)
        negative_prediction = self.forward(user_ids, item_ids)
        return negative_prediction

    def loss_func(self, positive_params, negative_params, train=True):
        prob_positive = self.expectation(positive_params)
        prob_negative = self.expectation(negative_params)
        loss = torch.cat((prob_positive, 1 - prob_negative))
        if train:
            penalty_pos = (positive_params[:, 0] - positive_params[:, 1]).square().sum()
            penalty_neg = (negative_params[:, 0] - negative_params[:, 1]).square().sum()
            penalty = self.penalty * (penalty_pos + penalty_neg)
            return - loss.log().mean() + penalty
        else:
            return loss.mean()

    def training_step(self, batch, batch_idx):
        positive_params = self.forward(batch[:, 0], batch[:, 1])
        negative_params = self.get_negative_prediction(batch)
        loss = self.loss_func(positive_params, negative_params)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        positive_params = self.forward(batch[:, 0], batch[:, 1])
        negative_params = self.get_negative_prediction(batch)
        loss = self.loss_func(positive_params, negative_params, train=False)
        self.log('accuracy', loss, prog_bar=True)
        self.log('val_loss', -loss)


class TwoWayMF(pl.LightningModule, Recommender):

    def __init__(self, relevance, embedding_dim, lr, weight_decay, loss=ABPR()):
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


class (BaseMF):

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

