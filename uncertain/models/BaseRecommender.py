import numpy as np
from torch import tensor, arange, from_numpy
from uncertain.utils import gpu, minibatch
from tqdm import trange


class BaseRecommender(object):

    def __init__(self,
                 n_iter,
                 learning_rate,
                 batch_size,
                 l2,
                 use_cuda,
                 verbose=True):

        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._verbose = verbose

        self._num_users = None
        self._num_items = None
        self._net = None
        self._optimizer = None
        self._loss_func = None

        self.train_loss = None
        self.test_loss = None

    def __repr__(self):

        if self._net is None:
            net_representation = '[uninitialised]'
        else:
            net_representation = repr(self._net)

        return ('<{}: {}>'.format(
            self.__class__.__name__,
            net_representation,
        ))

    @property
    def _initialized(self):
        return self._net is not None

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

    def _predict_process_ids(self, user_ids, item_ids):

        if item_ids is None:
            item_ids = arange(self._num_items)

        if np.isscalar(user_ids):
            user_ids = tensor(user_ids)

        if item_ids.size() != user_ids.size():
            user_ids = user_ids.expand(item_ids.size())

        user_var = gpu(user_ids, self._use_cuda)
        item_var = gpu(item_ids, self._use_cuda)

        return user_var.squeeze(), item_var.squeeze()

    def _one_epoch(self, interactions):

        epoch_loss = 0

        for (minibatch_num,
             (batch_user,
              batch_item,
              batch_ratings)) in enumerate(minibatch(interactions.user_ids,
                                                     interactions.item_ids,
                                                     interactions.ratings,
                                                     batch_size=self._batch_size)):

            predictions = self._net(batch_user, batch_item)
            loss = self._loss_func(batch_ratings, predictions)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            epoch_loss += loss.item()

        epoch_loss /= minibatch_num+1
        return epoch_loss

    def fit(self, train, test=None):
        """
        Fit the model.
        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------
        train: :class:`uncertain.interactions.Interactions`
            The input dataset. Must have ratings.
        test: :class:`uncertain.interactions.Interactions`
            Test dataset for iterative evaluation.
        """

        if not self._initialized:
            self._initialize(train)

        self._check_input(train.user_ids, train.item_ids)

        if self._verbose:
            t = trange(self._n_iter)
        else:
            t = range(self._n_iter)

        for epoch_num in t:

            self.train_loss.append(self._one_epoch(train))

            if test:
                self.test_loss.append(self._loss_func(test.ratings, self._predict(test.user_ids, test.item_ids)))
                if self._verbose:
                    t.set_postfix_str('Epoch {} loss - Train: {}, Test: {}'.format(epoch_num+1,
                                      self.train_loss[-1], self.test_loss[-1]))
            else:
                if self._verbose:
                    t.set_postfix_str('Epoch {} loss: {}'.format(epoch_num + 1, self.train_loss[-1]))

    def _predict(self, user_ids, item_ids=None):
        """
        Make predictions: given a user id, compute the net forward pass.

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
        user_ids, item_ids = self._predict_process_ids(user_ids, item_ids)

        out = self._net(user_ids, item_ids)

        if type(out) is not tuple:
            return out.detach()
        else:
            return out[0].detach(), out[1].detach()
