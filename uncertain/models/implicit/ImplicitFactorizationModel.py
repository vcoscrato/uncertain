import torch
from uncertain.models.implicit.BaseRecommender import BaseRecommender
from uncertain.utils import gpu
from uncertain.representations import BiasNet, FunkSVDNet
from uncertain.losses import pointwise_loss


class ImplicitFactorizationModel(BaseRecommender):
    """
    An Implicit feedback matrix factorization model.

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
                 n_iter,
                 batch_size,
                 l2,
                 learning_rate,
                 use_cuda,
                 sparse):

        super(ImplicitFactorizationModel, self).__init__(n_iter,
                                                         batch_size,
                                                         learning_rate,
                                                         use_cuda)

        self._embedding_dim = embedding_dim
        self._l2 = l2_penalty
        self._sparse = sparse

    def _initialize(self, interactions):

        self.train_loss = []
        self.test_loss = []

        (self._num_users,
         self._num_items,
         self._num_ratings) = (interactions.num_users,
                               interactions.num_items,
                               len(interactions.ratings))

        if self._representation is not None:
            self._net = gpu(self._representation,
                            self._use_cuda)
        else:
            self._net = gpu(
                BilinearNet(self._num_users,
                            self._num_items,
                            self._embedding_dim,
                            sparse=self._sparse),
                self._use_cuda
            )

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer_func(self._net.parameters())

        if self._loss == 'pointwise':
            self._loss_func = pointwise_loss
        elif self._loss == 'bpr':
            self._loss_func = bpr_loss
        elif self._loss == 'hinge':
            self._loss_func = hinge_loss
        else:
            self._loss_func = adaptive_hinge_loss

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._num_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def fit(self, interactions, verbose=False):
        """
        Fit the model.
        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.
        Parameters
        ----------
        interactions: :class:`spotlight.interactions.Interactions`
            The input dataset.
        verbose: bool
            Output additional information about current epoch and loss.
        """

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        if not self._initialized:
            self._initialize(interactions)

        self._check_input(user_ids, item_ids)

        for epoch_num in range(self._n_iter):

            users, items = shuffle(user_ids,
                                   item_ids,
                                   random_state=self._random_state)

            user_ids_tensor = gpu(torch.from_numpy(users),
                                  self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items),
                                  self._use_cuda)

            epoch_loss = 0.0

            for (minibatch_num,
                 (batch_user,
                  batch_item)) in enumerate(minibatch(user_ids_tensor,
                                                      item_ids_tensor,
                                                      batch_size=self._batch_size)):

                positive_prediction = self._net(batch_user, batch_item)

                if self._loss == 'adaptive_hinge':
                    negative_prediction = self._get_multiple_negative_predictions(
                        batch_user, n=self._num_negative_samples)
                else:
                    negative_prediction = self._get_negative_prediction(batch_user)

                self._optimizer.zero_grad()

                loss = self._loss_func(positive_prediction, negative_prediction)
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'
                                 .format(epoch_loss))


    def predict(self, user_ids, item_ids=None):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.
        Parameters
        ----------
        user_ids: int or array
           If int, will predict the recommendation scores for this
           user for all items in item_ids. If an array, will predict
           scores for all (user, item) pairs defined by user_ids and
           item_ids.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        Returns
        -------
        predictions: np.array
            Predicted scores for all items in item_ids.
        """

        self._check_input(user_ids, item_ids, allow_items_none=True)
        self._net.train(False)

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self._num_items,
                                                  self._use_cuda)

        out = self._net(user_ids, item_ids)

        return cpu(out).detach().numpy().flatten()