import torch
from pandas import DataFrame
from numpy import column_stack
from pytorch_lightning import LightningModule


class BiasModel(LightningModule):

    def __init__(self, n_user, n_item, lr):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.lr = lr
        self.user_bias = torch.nn.Embedding(self.n_user, 1)
        self.item_bias = torch.nn.Embedding(self.n_item, 1)
        torch.nn.init.normal_(self.user_bias.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_bias.weight, mean=0, std=0.01)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, user_ids, item_ids):
        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)
        return (user_bias + item_bias).flatten()

    @staticmethod
    def loss_func(predicted, observed):
        return ((observed - predicted) ** 2).sum()

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids).numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_bias = self.user_bias(user)
            return (user_bias + self.item_bias.weight).flatten().numpy()


class FactorizationModel(LightningModule):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, **kwargs):
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
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
    
    def predict(self, user_ids, item_ids=None):
        '''
        Interface from forward to output numpy arrays.
        '''
        with torch.no_grad():
            user_ids = torch.tensor(user_ids)
            if item_ids is not None:
                item_ids = torch.tensor(item_ids)
            return self(user_ids, item_ids).numpy()
    
    def rank(self, user_id, item_ids=None, ignored_item_ids=None, top_n=None):
        '''
        Ranking is performing directly on pytorch tensors.
        '''
        assert item_ids is None or ignored_item_ids is None, 'Passing both item_ids and ignored_item_ids is not supported.'

        with torch.no_grad():
            preds = self(user_id, item_ids)
            
        if top_n is None:
            top_n = self.n_item

        if ignored_item_ids is not None:
            preds[ignored_item_ids] = -float('inf')
            cut = min(len(preds) - len(ignored_item_ids), top_n)
        else:
            cut = top_n

        scores, items = torch.sort(preds, descending=True)
        return items[:cut], scores[:cut]

    def recommend(self, user, remove_items=None, n=10):
        '''
        Outputs beautiful DataFrame
        '''
        with torch.no_grad():
            out = DataFrame(self(torch.tensor(user)), columns=['scores'])
            if remove_items is not None:
                out.loc[remove_items, 'scores'] = -float('inf')
            out = out.sort_values(by='scores', ascending=False)[:n]
            return out


class UncertainRecommender:
    
    def predict(self, user_ids, item_ids=None):
        '''
        Interface from forward to output numpy arrays.
        '''
        with torch.no_grad():
            user_ids = torch.tensor(user_ids)
            if item_ids is not None:
                item_ids = torch.tensor(item_ids)
            preds, uncertainties = self(user_ids, item_ids)
            return preds.numpy(), uncertainties.numpy()
        
    def uncertain_predict(self, user_ids, item_ids=None, **kwargs):
        '''
        Interface from uncertain_forward to output numpy arrays.
        '''
        with torch.no_grad():
            user_ids = torch.tensor(user_ids)
            if item_ids is not None:
                item_ids = torch.tensor(item_ids)
            obj = self.uncertain_transform(user_ids, item_ids)
            preds = self.uncertain_transform(obj, **kwargs)
            return preds.numpy()
    
    def rank(self, user_id, item_ids=None, ignored_item_ids=None, top_n=None):
        '''
        Ranking is performing directly on pytorch tensors.
        '''
        assert item_ids is None or ignored_item_ids is None, 'Passing both item_ids and ignored_item_ids is not supported.'

        with torch.no_grad():
            preds, uncertainties = self(user_id, item_ids)
        
        if top_n is None:
            top_n = self.n_item
        
        if ignored_item_ids is not None:
            preds[ignored_item_ids] = -float('inf')
            cut = min(len(preds) - len(ignored_item_ids), top_n)
        else:
            cut = top_n

        scores, items = torch.sort(preds, descending=True)
        uncertainties = uncertainties[items]

        return items[:cut], scores[:cut], uncertainties[:cut]
    
    def uncertain_rank(self, user_id, item_ids=None, ignored_item_ids=None, top_n=None, **kwargs):
        '''
        Ranking is performing directly on pytorch tensors.
        '''
        assert item_ids is None or ignored_item_ids is None, 'Passing both item_ids and ignored_item_ids is not supported.'

        with torch.no_grad():
            obj = self(user_id, item_ids)
            preds = self.uncertain_transform(obj, **kwargs)
            
        if top_n is None:
            top_n = self.n_item

        if ignored_item_ids is not None:
            preds[ignored_item_ids] = -float('inf')
            cut = min(len(preds) - len(ignored_item_ids), top_n)
        else:
            cut = top_n

        scores, items = torch.sort(preds, descending=True)
        return items[:cut], scores[:cut]

    def recommend(self, user, remove_items=None, n=10):
        '''
        Outputs beautiful DataFrame
        '''
        with torch.no_grad():
            out = DataFrame(column_stack(self(torch.tensor(user))))
            out.columns = ['scores', 'uncertainties']
            if remove_items is not None:
                out.loc[remove_items, 'scores'] = -float('inf')
            out = out.sort_values(by='scores', ascending=False)[:n]
            return out

    def uncertain_recommend(self, user, remove_items=None, n=10, **kwargs):
        '''
        Outputs beautiful DataFrame
        '''        
        with torch.no_grad():
            obj = self(torch.tensor(user))
            preds = self.uncertain_transform(obj, **kwargs)
            out = DataFrame(preds, columns=['scores'])
            if remove_items is not None:
                out.loc[remove_items, 'scores'] = -float('inf')
            out = out.sort_values(by='scores', ascending=False)[:n]
            return out
