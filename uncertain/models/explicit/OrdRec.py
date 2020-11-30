import torch
from uncertain.models.explicit.BaseRecommender import BaseRecommender
from uncertain.utils import gpu
from uncertain.representations import OrdRecNet
from uncertain.losses import max_prob_loss


class OrdRec(BaseRecommender):

    def __init__(self,
                 ratings_labels,
                 embedding_dim,
                 batch_size,
                 l2_base,
                 l2_step,
                 learning_rate,
                 use_cuda,
                 path,
                 sparse=False):

        super(OrdRec, self).__init__(batch_size,
                                     learning_rate,
                                     use_cuda,
                                     path)

        self._desc = 'OrdRec'
        self._rating_labels = ratings_labels
        self._embedding_dim = embedding_dim
        self._l2_base = l2_base
        self._l2_step = l2_step
        self._sparse = sparse

    def _initialize(self, interactions):
        
        self.train_loss = []
        self.test_loss = []

        (self._num_users,
         self._num_items,
         self._num_ratings) = (interactions.num_users,
                               interactions.num_items,
                               len(interactions.ratings))

        self._net = gpu(OrdRecNet(self._num_users,
                                  self._num_items,
                                  len(self._rating_labels),
                                  self._embedding_dim,
                                  self._sparse),
                        self._use_cuda)

        self._optimizer = torch.optim.SGD(
            [{'params': self._net.user_embeddings.parameters(), 'weight_decay': self._l2_base, 'lr': self._lr},
             {'params': self._net.item_embeddings.parameters(), 'weight_decay': self._l2_base, 'lr': self._lr},
             {'params': self._net.user_betas.parameters(), 'weight_decay': self._l2_step, 'lr': self._lr}])

        self._loss_func = max_prob_loss

    def predict(self, user_ids, item_ids=None, return_distribution=False):

        distribution = self._predict(user_ids, item_ids)

        if return_distribution:
            return distribution

        # Most probable rating
        #mean = self._rating_labels[(out.argmax(1))]
        #confidence = out.max(1)[0]

        # Average ranking
        mean = (distribution * self._rating_labels).sum(1)
        var = ((distribution * self._rating_labels**2).sum(1) - mean**2).abs()

        return mean, var