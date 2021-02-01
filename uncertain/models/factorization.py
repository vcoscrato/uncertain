import torch
import numpy as np
from uncertain.models.base import BaseRecommender
from uncertain.utils import gpu, minibatch
from uncertain.representations import BiasNet, FunkSVDNet, CPMFNet, OrdRecNet
from uncertain.losses import regression_loss, gaussian_loss, max_prob_loss


class LatentFactorRecommender(BaseRecommender):

    def __init__(self,
                 batch_size,
                 learning_rate,
                 use_cuda,
                 path,
                 verbose=True):

        super().__init__()

        self._lr = learning_rate
        self._batch_size = batch_size
        self._use_cuda = use_cuda
        self._path = path
        self._verbose = verbose

        self._net = None
        self._optimizer = None
        self._loss_func = None
        self.train_loss = None
        self.test_loss = None
        self.min_val_loss = None

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
    def is_uncertain(self):
        return 'basic' not in self._desc

    @property
    def _initialized(self):
        return self._net is not None

    def _predict_process_ids(self, user_ids, item_ids):

        if item_ids is None:
            item_ids = torch.arange(self.num_items)

        if np.isscalar(user_ids):
            user_ids = torch.tensor(user_ids)

        if item_ids.size() != user_ids.size():
            user_ids = user_ids.expand(item_ids.size())

        user_var = gpu(user_ids, self._use_cuda)
        item_var = gpu(item_ids, self._use_cuda)

        return user_var.squeeze(), item_var.squeeze()

    def _one_epoch(self, interactions):

        epoch_loss = 0

        loader = minibatch(interactions, batch_size=self._batch_size)

        for (minibatch_num,
             (batch_interactions,
              batch_ratings)) in enumerate(loader):
            predictions = self._net(batch_interactions[:, 0], batch_interactions[:, 1])
            loss = self._loss_func(batch_ratings, predictions)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            epoch_loss += loss.item()

        epoch_loss /= minibatch_num + 1
        return epoch_loss

    def fit(self, train, validation):
        """
        Fit the model.
        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call. Uses the validation set for early stopping.

        Parameters
        ----------
        train: :class:`uncertain.interactions.Interactions`
            The input dataset. Must have ratings.
        validation: :class:`uncertain.interactions.Interactions`
            Test dataset for iterative evaluation.
        """

        if not self._initialized:
            self.initialize(train)
            try:
                self._net.load_state_dict(torch.load(self._path))
                if self._verbose:
                    print('Previous training file encountered. Loading and resuming training.')
            except:
                if self._verbose:
                    print('No previous training file found. Starting training from scratch.')

        epoch = 1
        tol = False
        while True:

            train.shuffle()
            self._net.train()
            self.train_loss.append(self._one_epoch(train))

            validation_loader = minibatch(validation, batch_size=int(1e5))
            epoch_loss = 0
            with torch.no_grad():

                for (minibatch_num,
                     (batch_interactions,
                      batch_ratings)) in enumerate(validation_loader):
                    self._net.eval()
                    predictions = self._net(batch_interactions[:, 0], batch_interactions[:, 1])
                    epoch_loss += self._loss_func(batch_ratings, predictions).item()

                epoch_loss /= minibatch_num + 1

            out = 'Epoch {} loss - Train: {}, Validation: {}'.format(epoch, self.train_loss[-1], epoch_loss)

            if epoch_loss < self.min_val_loss:
                tol = False
                self.min_val_loss = epoch_loss
                out += ' - This is the lowest validation loss so far'
                epoch += 1
                torch.save(self._net.state_dict(), self._path)

            else:
                self._net.load_state_dict(torch.load(self._path))
                if tol:
                    out += ' - Validation loss did not improve. Ending training.'
                    if self._verbose:
                        print(out)
                    break
                else:
                    out += ' - Validation loss did not improve. Reducing learning rate.'
                    tol = True
                    for g in self._optimizer.param_groups:
                        g['lr'] /= 2

            if self._verbose:
                print(out)

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

        user_ids, item_ids = self._predict_process_ids(user_ids, item_ids)

        with torch.no_grad():
            out = self._net(user_ids, item_ids)

        return out


class FunkSVD(LatentFactorRecommender):
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

        super().__init__(batch_size, learning_rate, use_cuda, path)

        self._embedding_dim = embedding_dim
        self._l2 = l2
        self._sparse = sparse

    def initialize(self, interactions):
        
        self.train_loss = []
        self.test_loss = []
        self.min_val_loss = float('inf')

        (self.num_users,
         self.num_items,
         self.num_ratings) = (interactions.num_users,
                              interactions.num_items,
                              len(interactions))

        if self._embedding_dim == 0:
            self._net = gpu(BiasNet(self.num_users,
                                    self.num_items,
                                    self._sparse),
                            self._use_cuda)
            self._desc = 'basic-Linear recommender'

        else:
            self._net = gpu(FunkSVDNet(self.num_users,
                                       self.num_items,
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


class CPMF(LatentFactorRecommender):

    def __init__(self,
                 embedding_dim,
                 batch_size,
                 l2,
                 learning_rate,
                 use_cuda,
                 path,
                 sparse=False):
        super().__init__(batch_size, learning_rate, use_cuda, path)

        self._desc = 'CPMF'
        self._embedding_dim = embedding_dim
        self._l2 = l2
        self._sparse = sparse

    def initialize(self, interactions):

        self.train_loss = []
        self.test_loss = []
        self.min_val_loss = float('inf')

        (self.num_users,
         self.num_items,
         self.num_ratings) = (interactions.num_users,
                              interactions.num_items,
                              len(interactions))

        self._net = gpu(CPMFNet(self.num_users,
                                self.num_items,
                                self._embedding_dim,
                                self._sparse),
                        self._use_cuda)

        self._optimizer = torch.optim.SGD(
            self._net.parameters(),
            lr=self._lr,
            weight_decay=self._l2
        )

        self._loss_func = gaussian_loss

    def predict(self, user_ids, item_ids=None):
        out = self._predict(user_ids, item_ids)

        return out[0], out[1]


class OrdRec(LatentFactorRecommender):

    def __init__(self,
                 ratings_labels,
                 embedding_dim,
                 batch_size,
                 l2,
                 learning_rate,
                 use_cuda,
                 path,
                 sparse=False):
        super().__init__(batch_size, learning_rate, use_cuda, path)

        self._desc = 'OrdRec'
        self._rating_labels = ratings_labels
        self._embedding_dim = embedding_dim
        self._l2 = l2
        self._sparse = sparse

    def initialize(self, interactions):

        self.train_loss = []
        self.test_loss = []
        self.min_val_loss = float('inf')

        (self.num_users,
         self.num_items,
         self.num_ratings) = (interactions.num_users,
                              interactions.num_items,
                              len(interactions))

        self._net = gpu(OrdRecNet(self.num_users,
                                  self.num_items,
                                  len(self._rating_labels),
                                  self._embedding_dim,
                                  self._sparse),
                        self._use_cuda)

        self._optimizer = torch.optim.SGD(
            self._net.parameters(),
            lr=self._lr,
            weight_decay=self._l2
        )

        self._loss_func = max_prob_loss

    def predict(self, user_ids, item_ids=None, return_distribution=False):
        distribution = self._predict(user_ids, item_ids)

        if return_distribution:
            return distribution

        # Most probable rating
        # mean = self._rating_labels[(out.argmax(1))]
        # confidence = out.max(1)[0]

        # Average ranking
        mean = (distribution * self._rating_labels).sum(1)
        var = ((distribution * self._rating_labels ** 2).sum(1) - mean ** 2).abs()

        return mean, var
