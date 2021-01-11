import torch
from uncertain.models.explicit.BaseRecommender import BaseRecommender
from uncertain.utils import gpu
from uncertain.representations import BiasNet, FunkSVDNet
from uncertain.losses import regression_loss


class ExplicitFactorizationModel(BaseRecommender):
    """
    An Explicit feedback matrix factorization model.

    Parameters
    ----------
    embedding_dim: int
        The dimension of the latent factors. If 0, then
        a bias model is used.
    n_iter: int
        Number of iterations to run.
    batch_size: int
        Minibatch size.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float
        Initial learning rate.
    use_cuda: boolean
        Run the model on a GPU.
    sparse: boolean
        Use sparse gradients for embedding layers.
    """

    def __init__(self,
                 embedding_dim,
                 batch_size,
                 l2,
                 learning_rate,
                 use_cuda,
                 path,
                 sparse=False):

        super(ExplicitFactorizationModel, self).__init__(batch_size,
                                                         learning_rate,
                                                         use_cuda,
                                                         path)

        self._embedding_dim = embedding_dim
        self._l2 = l2
        self._sparse = sparse

    def initialize(self, interactions):
        
        self.train_loss = []
        self.test_loss = []

        (self._num_users,
         self._num_items,
         self._num_ratings) = (interactions.num_users,
                               interactions.num_items,
                               len(interactions))

        if self._embedding_dim == 0:
            self._net = gpu(BiasNet(self._num_users,
                                    self._num_items,
                                    self._sparse),
                            self._use_cuda)
            self._desc = 'basic-Linear recommender'

        else:
            self._net = gpu(FunkSVDNet(self._num_users,
                                       self._num_items,
                                       self._embedding_dim,
                                       self._sparse),
                            self._use_cuda)
            self._desc = 'basic-FunkSVD'

        self._optimizer = torch.optim.SGD(
            self._net.parameters(),
            lr=self._lr,
            weight_decay=self._l2
            )

        self._loss_func = regression_loss

    def predict(self, user_ids, item_ids=None):

        return self._predict(user_ids, item_ids)