import torch
from uncertain.models.explicit.BaseRecommender import BaseRecommender
from uncertain.utils import gpu
from uncertain.representations import CPMFNet
from uncertain.losses import gaussian_loss


class CPMF(BaseRecommender):

    def __init__(self,
                 embedding_dim,
                 batch_size,
                 l2_base,
                 l2_var,
                 learning_rate,
                 use_cuda,
                 path,
                 sparse=False):

        super(CPMF, self).__init__(batch_size,
                                   learning_rate,
                                   use_cuda,
                                   path)

        self._desc = 'CPMF'
        self._embedding_dim = embedding_dim
        self._l2_base = l2_base
        self._l2_var = l2_var
        self._sparse = sparse

    def initialize(self, interactions):
        
        self.train_loss = []
        self.test_loss = []

        (self._num_users,
         self._num_items,
         self._num_ratings) = (interactions.num_users,
                               interactions.num_items,
                               len(interactions.ratings))

        self._net = gpu(CPMFNet(self._num_users,
                                self._num_items,
                                self._embedding_dim,
                                self._sparse),
                        self._use_cuda)

        self._optimizer = torch.optim.SGD(
            [{'params': self._net.user_embeddings.parameters(), 'weight_decay': self._l2_base, 'lr': self._lr},
             {'params': self._net.item_embeddings.parameters(), 'weight_decay': self._l2_base, 'lr': self._lr},
             {'params': self._net.user_gammas.parameters(), 'weight_decay': self._l2_var, 'lr': self._lr},
             {'params': self._net.item_gammas.parameters(), 'weight_decay': self._l2_var, 'lr': self._lr}])

        self._loss_func = gaussian_loss

    def predict(self, user_ids, item_ids=None):

        out = self._predict(user_ids, item_ids)

        return out[0], out[1]
