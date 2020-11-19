import torch
from uncertain.models.BaseRecommender import BaseRecommender
from uncertain.utils import gpu, assert_no_grad
from uncertain.layers import ZeroEmbedding, ScaledEmbedding


def max_prob_loss(observed_ratings, predicted_ratings):
    """
    Maximum probability loss for ordinal data.

    Parameters
    ----------
    observed_ratings: tensor
        Tensor (n_obs) containing the observed ordinal rating labels.
    predicted_ratings: tensor
        Tensor (n_obs, n_rating_labels) containing the probabilities for each rating label.

        Returns
    -------
    loss: float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    return -predicted_ratings[range(len(-predicted_ratings)), observed_ratings].mean()


class OrdRecNet(torch.nn.Module):
    """
    OrdRec representation.

    Parameters
    ----------
    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(self, num_users, num_items, num_labels, embedding_dim, sparse=False):

        super(OrdRecNet, self).__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)

        self.user_betas = ZeroEmbedding(num_users, num_labels-1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------
        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------
        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids).squeeze()
        item_embedding = self.item_embeddings(item_ids).squeeze()

        y = (user_embedding * item_embedding).sum(1).reshape(-1, 1)

        user_beta = self.user_betas(user_ids)
        user_beta[:, 1:] = torch.exp(user_beta[:, 1:])
        user_distribution = torch.div(1, 1 + torch.exp(y - user_beta.cumsum(1)))

        one = torch.ones((len(user_distribution), 1), device=user_beta.device)
        user_distribution = torch.cat((user_distribution, one), 1)

        user_distribution[:, 1:] -= user_distribution[:, :-1].clone()

        return user_distribution


class OrdRec(BaseRecommender):

    def __init__(self,
                 ratings_labels,
                 embedding_dim,
                 n_iter,
                 batch_size,
                 learning_rate,
                 l2_base,
                 l2_step,
                 use_cuda):

        super(OrdRec, self).__init__(n_iter,
                                     learning_rate,
                                     batch_size,
                                     use_cuda)

        self._desc = 'OrdRec'
        self._rating_labels = ratings_labels
        self._embedding_dim = embedding_dim
        self._l2_base = l2_base
        self._l2_step = l2_step

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
                                  self._embedding_dim),
                        self._use_cuda)

        self._optimizer = torch.optim.Adam(
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