import torch
from uncertain.models.BaseRecommender import BaseRecommender
from uncertain.utils import gpu, assert_no_grad
from uncertain.layers import ZeroEmbedding, ScaledEmbedding


def gaussian_loss(observed_ratings, predicted_ratings):
    """
    Gaussian loss.

    Parameters
    ----------
    observed_ratings: tensor
        Tensor (n_obs) containing the observed ratings.
    predicted_ratings: tuple
        A tuple containing 2 tensors: the estimated averages (n_obs) and variances (n_obs).

        Returns
    -------
    loss: float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    mean, variance = predicted_ratings

    return (((observed_ratings - mean) ** 2) / variance).mean() + torch.log(variance).mean()


class CPMFNet(torch.nn.Module):
    """
    Confidence-Aware probabilistic matrix factorization representation.

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
    def __init__(self, num_users, num_items, embedding_dim, sparse=False):

        super(CPMFNet, self).__init__()

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)

        self.user_gammas = ScaledEmbedding(num_users, 1, sparse=sparse)
        self.item_gammas = ScaledEmbedding(num_items, 1, sparse=sparse)

        self.var_activation = torch.nn.Softplus()

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

        dot = (user_embedding * item_embedding).sum(1)

        user_gamma = self.user_gammas(user_ids).squeeze()
        item_gamma = self.item_gammas(item_ids).squeeze()

        var = self.var_activation(user_gamma + item_gamma)

        return dot, var


class CPMF(BaseRecommender):

    def __init__(self,
                 embedding_dim,
                 n_iter,
                 batch_size,
                 learning_rate,
                 sigma,
                 use_cuda):

        super(CPMF, self).__init__(n_iter,
                                   learning_rate,
                                   batch_size,
                                   1/sigma,
                                   use_cuda)

        self._embedding_dim = embedding_dim
        self._sigma = sigma

        self.train_loss = []
        self.test_loss = []

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items,
         self._num_ratings) = (interactions.num_users,
                               interactions.num_items,
                               len(interactions.ratings))

        self._net = gpu(CPMFNet(self._num_users,
                                self._num_items,
                                self._embedding_dim),
                        self._use_cuda)

        self._optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self._learning_rate,
            weight_decay=self._l2
            )

        self._loss_func = gaussian_loss

    def predict(self, user_ids, item_ids=None):

        mean, var = self._predict(user_ids, item_ids)

        return mean, var