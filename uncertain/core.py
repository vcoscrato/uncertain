import torch
from pandas import DataFrame
from numpy import column_stack
from pytorch_lightning import LightningModule


class FactorizationModel(LightningModule):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss_func=None, n_negative=None):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.user_embeddings = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.01)
        if loss_func is not None:
            self.loss_func = loss_func
        if n_negative is not None:
            self.n_negative = n_negative

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        return optimizer

    def dot(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        return (user_embeddings * item_embeddings).sum(1)

    def forward(self, user_ids, item_ids):
        return self.dot(user_ids, item_ids)

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids).numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            return (user_embedding * self.item_embeddings.weight).sum(1).numpy()


class VanillaRecommender:

    def recommend(self, user, remove_items=None, n=10):
        out = DataFrame(self.predict_user(torch.tensor(user)), columns=['scores'])
        if remove_items is not None:
            out.loc[remove_items, 'scores'] = -float('inf')
        out = out.sort_values(by='scores', ascending=False)[:n]
        return out


class UncertainRecommender:

    def recommend(self, user, remove_items=None, n=10):
        out = DataFrame(column_stack(self.predict_user(torch.tensor(user))))
        out.columns = ['scores', 'uncertainties']
        if remove_items is not None:
            out.loc[remove_items, 'scores'] = -float('inf')
        out = out.sort_values(by='scores', ascending=False)[:n]
        return out
