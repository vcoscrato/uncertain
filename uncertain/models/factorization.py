import torch
import pytorch_lightning as pl
from uncertain.core import Recommender
from uncertain.layers import ZeroEmbedding, ScaledEmbedding
from uncertain.losses import mse_loss, gaussian_loss, max_prob_loss, cross_entropy_loss, bpr_loss
from uncertain.metrics import rmse_score, rpi_score, classification, correlation, quantile_score, get_hits, ndcg


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
        with torch.no_grad():
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

    def test_recommendations(self, test_interactions, train_interactions, max_k=10, relevance_threshold=None):
        out = {}
        precision = []
        recall = []
        ndcg_ = []
        rri = []
        precision_denom = torch.arange(1, max_k + 1, dtype=torch.float64)
        ndcg_denom = torch.log2(precision_denom + 1)
        for user in range(test_interactions.num_users):
            targets = test_interactions.get_rated_items(user, threshold=relevance_threshold)
            if not len(targets):
                continue
            rec = self.recommend(user, train_interactions.get_rated_items(user))
            hits = get_hits(rec, targets)
            num_hit = hits.cumsum(0)
            precision.append(num_hit / precision_denom)
            recall.append(num_hit / len(targets))
            ndcg_.append(ndcg(hits, ndcg_denom))
            if self.is_uncertain and hits.sum().item() > 0:
                with torch.no_grad():
                    rri_ = torch.empty(max_k - 1)
                    for i in range(2, max_k + 1):
                        unc = rec.uncertainties[:i]
                        rri_[i - 2] = (unc.mean() - unc[hits[:i]]).mean() / unc.std()
                    rri.append(rri_)
        out['Precision'] = torch.vstack(precision).mean(0)
        out['Recall'] = torch.vstack(recall).mean(0)
        out['NDCG'] = torch.vstack(ndcg_).mean(0)
        if len(rri) > 0:
            rri = torch.vstack(rri)
            out['RRI'] = rri.nansum(0) / (~rri.isnan()).float().sum(0)
        return out


class Explicit(object):

    def training_step(self, batch, batch_idx):
        users, items, ratings = batch
        output = self.forward(users, items)
        loss = self.loss_func(output, ratings)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        users, items, ratings = batch
        output = self.forward(users, items)
        loss = self.loss_func(output, ratings)
        self.log('val_loss', loss)

    def test_ratings(self, test_interactions):
        with torch.no_grad():
            out = {}
            predictions = self.forward(test_interactions.users, test_interactions.items)
            out['loss'] = self.loss_func(predictions, test_interactions.scores).item()
            if not self.is_uncertain:
                out['RMSE'] = rmse_score(predictions, test_interactions.scores)
            else:
                out['RMSE'] = rmse_score(predictions[0], test_interactions.scores)
                errors = torch.abs(test_interactions.scores - predictions[0])
                out['RPI'] = rpi_score(errors, predictions[1])
                out['Classification'] = classification(errors, predictions[1])
                out['Correlation'] = correlation(errors, predictions[1])
                out['Quantile RMSE'] = quantile_score(errors, predictions[1])
            return out


class Implicit(object):

    def get_negative_prediction(self, users):
        sampled_items = torch.randint(0, self.num_items, (len(users),))
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
        var = self.var_activation(user_gamma + item_gamma).flatten()
        return dot, var


class OrdRec(Explicit, FactorizationModel):

    def __init__(self, interactions, embedding_dim, lr, batch_size, weight_decay):
        super().__init__(interactions, embedding_dim, lr, batch_size, weight_decay)
        self.user_betas = ZeroEmbedding(self.num_users, len(self.score_labels) - 1)
        self.loss_func = max_prob_loss

    @property
    def is_uncertain(self):
        return True

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        y = (user_embedding * item_embedding).sum(1).reshape(-1, 1)
        user_beta = self.user_betas(user_ids)
        user_beta[:, 1:] = torch.exp(user_beta[:, 1:])
        distribution = torch.div(1, 1 + torch.exp(y - user_beta.cumsum(1)))
        one = torch.ones((len(distribution), 1), device=distribution.device)
        distribution = torch.cat((distribution, one), 1)
        distribution[:, 1:] -= distribution[:, :-1].clone()
        return distribution

    def _summarize(self, distributions):
        labels = self.score_labels.to(distributions.device)
        mean = (distributions * labels).sum(1)
        var = ((distributions * labels ** 2).sum(1) - mean ** 2).abs()
        return mean, var

    def predict(self, user_id):
        with torch.no_grad():
            item_ids = torch.arange(self.num_items)
            user_ids = torch.full_like(item_ids, user_id)
            distributions = self.forward(user_ids, item_ids)
            return self._summarize(distributions)

    def test_ratings(self, test_interactions):
        with torch.no_grad():
            out = {}
            distributions = self.forward(test_interactions.users, test_interactions.items)
            ord_scores = torch.unique(test_interactions.scores, return_inverse=True)[1]
            out['loss'] = self.loss_func(distributions, ord_scores).item()
            predictions = self._summarize(distributions)
            out['RMSE'] = rmse_score(predictions[0], test_interactions.scores)
            errors = torch.abs(test_interactions.scores - predictions[0])
            out['RPI'] = rpi_score(errors, predictions[1])
            out['Classification'] = classification(errors, predictions[1])
            out['Correlation'] = correlation(errors, predictions[1])
            out['Quantile RMSE'] = quantile_score(errors, predictions[1])
            return out


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
