from uncertain.models.BaseRecommender import BaseRecommender
from uncertain.utils import gpu, assert_no_grad
from uncertain.layers import ZeroEmbedding, ScaledEmbedding
from torch.optim import Adam
from torch.nn import Module


def regression_loss(observed_ratings, predicted_ratings):
    """
    Regression loss.

    Parameters
    ----------
    observed_ratings: tensor
        Tensor containing observed ratings.
    predicted_ratings: tensor
        Tensor containing rating predictions.

    Returns
    -------
    loss, float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    return ((observed_ratings - predicted_ratings) ** 2).mean()


class BiasNet(Module):
    """
    Bias representation.
    The score for a user-item pair is given by the
    sum of the item and the user biases.

    Parameters
    ----------
    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(self, num_users, num_items, sparse=False):

        super(BiasNet, self).__init__()

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

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

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return user_bias + item_bias


class FunkSVDNet(Module):
    """
    Bilinear factorization representation.
    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

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

        super(FunkSVDNet, self).__init__()

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)

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

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        return dot


class ExplicitFactorizationModel(BaseRecommender):

    def __init__(self,
                 embedding_dim,
                 n_iter,
                 learning_rate,
                 batch_size,
                 l2_penalty,
                 use_cuda):

        super(ExplicitFactorizationModel, self).__init__(n_iter,
                                                         learning_rate,
                                                         batch_size,
                                                         use_cuda)

        self._embedding_dim = embedding_dim
        self._l2 = l2_penalty

    def _initialize(self, interactions):
        
        self.train_loss = []
        self.test_loss = []

        (self._num_users,
         self._num_items,
         self._num_ratings) = (interactions.num_users,
                               interactions.num_items,
                               len(interactions.ratings))

        if self._embedding_dim == 0:
            self._net = gpu(BiasNet(self._num_users,
                                    self._num_items),
                            self._use_cuda)
            self._desc = 'basic-Linear recommender'

        else:
            self._net = gpu(FunkSVDNet(self._num_users,
                                       self._num_items,
                                       self._embedding_dim),
                            self._use_cuda)
            self._desc = 'basic-FunkSVD'

        self._optimizer = Adam(
            self._net.parameters(),
            lr=self._lr,
            weight_decay=self._l2
            )

        self._loss_func = regression_loss

    def predict(self, user_ids, item_ids=None):

        return self._predict(user_ids, item_ids)