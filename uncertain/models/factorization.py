import os
import torch
import numpy as np
from uncertain.models.base import Recommender
from uncertain.representations import BiasNet, FunkSVDNet, CPMFNet, OrdRecNet
from uncertain.losses import funk_svd_loss, cpmf_loss, max_prob_loss


class LatentFactorRecommender(Recommender):

    def __init__(self,
                 batch_size,
                 learning_rate,
                 tolerance,
                 path,
                 verbose,
                 max_epochs=float('inf')):

        self._batch_size = batch_size
        self._lr = learning_rate
        self._tolerance = tolerance
        self._path = path
        self._verbose = verbose
        self.max_epochs = max_epochs

        self.type = None
        self._net = None
        self._optimizer = None
        self.train_loss = None
        self.test_loss = None
        self.min_val_loss = None

        super().__init__()

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

    def initialize(self, interactions):

        self.type = interactions.type
        self.device = interactions.device

        self.train_loss = []
        self.test_loss = []
        self.min_val_loss = float('inf')

        (self.user_labels,
         self.item_labels,
         self.num_ratings) = (interactions.user_labels,
                              interactions.item_labels,
                              len(interactions))

        self._net = self._construct_net()

        self._optimizer = torch.optim.SGD(
            self._net.parameters(),
            lr=self._lr,
            weight_decay=self._l2
            )

    def load(self):
        if os.path.exists(self._path):
            self._net.load_state_dict(torch.load(self._path))
            if self._verbose:
                print('Previous training file encountered. Loading saved weights.')

    def _forward_pass(self, batch_interactions, batch_ratings):

        predictions = self._net(batch_interactions[:, 0], batch_interactions[:, 1])
        if self.type == 'Explicit':
            return self._loss_func(predictions, batch_ratings)
        else:
            predictions = self._net(batch_interactions[:, 0], batch_interactions[:, 1])
            negative_predictions = self._get_negative_prediction(batch_interactions[:, 0])
            return self._loss_func(predictions, predicted_negative=negative_predictions)

    def _get_negative_prediction(self, user_ids):

        negative_items = self.sample_items(len(item_ids))
        negative_prediction = self._net(user_ids, negative_var)

        return negative_prediction

    def _one_epoch(self, loader):

        epoch_loss = 0

        for (minibatch_num,
             (batch_interactions,
              batch_ratings)) in enumerate(loader):

            loss = self._forward_pass(batch_interactions, batch_ratings)
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
        train: :class:`uncertain.data_structures.Interactions`
            The input dataset. Must have ratings.
        validation: :class:`uncertain.data_structures.Interactions`
            Test dataset for iterative evaluation.
        """

        if not self._initialized:
            self.initialize(train)

        epoch = 1
        tol = 0
        while epoch < self.max_epochs:

            train.shuffle()
            train_loader = train.minibatch(batch_size=self._batch_size)
            self._net.train()
            self.train_loss.append(self._one_epoch(train_loader))

            validation_loader = validation.minibatch(batch_size=int(1e5))
            epoch_loss = 0
            with torch.no_grad():

                for (minibatch_num,
                     (batch_interactions,
                      batch_ratings)) in enumerate(validation_loader):

                    self._net.eval()
                    predictions = self._net(batch_interactions[:, 0], batch_interactions[:, 1])
                    if train.type == 'Implicit':
                        epoch_loss += self._loss_func(predictions).item()
                    else:
                        epoch_loss += self._loss_func(predictions, batch_ratings).item()

                epoch_loss /= minibatch_num + 1

            out = 'Epoch {} loss - Train: {}, Validation: {}'.format(epoch, self.train_loss[-1], epoch_loss)

            if epoch_loss < self.min_val_loss:
                tol = 0
                self.min_val_loss = epoch_loss
                out += ' - This is the lowest validation loss so far'
                epoch += 1
                torch.save(self._net.state_dict(), self._path)

            else:
                tol += 1
                self._net.load_state_dict(torch.load(self._path))
                if tol > self._tolerance:
                    out += ' - Validation loss did not improve. Ending training.'
                    if self._verbose:
                        print(out)
                    break
                else:
                    out += ' - Validation loss did not improve. Reducing learning rate.'
                    for g in self._optimizer.param_groups:
                        g['lr'] /= 2

            if self._verbose:
                print(out)

        if self._path == os.getcwd()+'tmp':
            os.remove(self._path)

    def get_item_similarity(self, item_id, candidate_ids=None):

        if hasattr(self._net, 'item_embeddings'):
            with torch.no_grad():
                item_var = self._net.item_embeddings(torch.tensor(item_id, device=self.device))
                if candidate_ids is None:
                    candidate_ids = torch.arange(self.num_items, device=self.device)
                candidates_var = self._net.item_embeddings(candidate_ids)
                return torch.cosine_similarity(item_var, candidates_var, dim=-1)
        else:
            raise Exception('Model has no item_embeddings.')


class Linear(LatentFactorRecommender):
    """
    An Explicit feedback bias model.

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
    sparse: boolean
        Use sparse gradients for embedding layers.
    """

    def __init__(self,
                 batch_size,
                 l2,
                 learning_rate,
                 tolerance,
                 path=os.getcwd() + 'tmp',
                 sparse=False,
                 verbose=True):

        self._l2 = l2
        self._sparse = sparse
        self._loss_func = funk_svd_loss

        super().__init__(batch_size, learning_rate, tolerance, path, verbose, max_epochs=100)

    @property
    def is_uncertain(self):
        return False

    def _construct_net(self):

        return BiasNet(self.num_users, self.num_items, self._sparse).to(self.device)

    def predict(self, user_ids, item_ids):
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
    sparse: boolean
        Use sparse gradients for embedding layers.
    """

    def __init__(self,
                 embedding_dim,
                 batch_size,
                 l2,
                 learning_rate,
                 tolerance,
                 path=os.getcwd()+'tmp',
                 sparse=False,
                 verbose=True):

        self._embedding_dim = embedding_dim
        self._l2 = l2
        self._sparse = sparse
        self._loss_func = funk_svd_loss

        super().__init__(batch_size, learning_rate, tolerance, path, verbose)

    @property
    def is_uncertain(self):
        return False

    def _construct_net(self):
        
        return FunkSVDNet(self.num_users, self.num_items, self._embedding_dim, self._sparse).to(self.device)

    def predict(self, user_ids, item_ids):
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

        with torch.no_grad():
            out = self._net(user_ids, item_ids)

        return out


class CPMF(LatentFactorRecommender):

    def __init__(self,
                 embedding_dim,
                 batch_size,
                 l2,
                 learning_rate,
                 tolerance,
                 path=os.getcwd()+'tmp',
                 sparse=False,
                 verbose=True):

        self._embedding_dim = embedding_dim
        self._l2 = l2
        self._sparse = sparse
        self._loss_func = cpmf_loss

        super().__init__(batch_size, learning_rate, tolerance, path, verbose)

    @property
    def is_uncertain(self):
        return True

    def _construct_net(self):

        return CPMFNet(self.num_users, self.num_items, self._embedding_dim, self._sparse).to(self.device)

    def predict(self, user_ids, item_ids):
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

        with torch.no_grad():
            out = self._net(user_ids, item_ids)

        return out[0], out[1]


class OrdRec(LatentFactorRecommender):

    def __init__(self,
                 ratings_labels,
                 embedding_dim,
                 batch_size,
                 l2,
                 learning_rate,
                 tolerance,
                 path=os.getcwd()+'tmp',
                 sparse=False,
                 verbose=True):

        self._rating_labels = ratings_labels
        self._embedding_dim = embedding_dim
        self._l2 = l2
        self._sparse = sparse
        self._loss_func = max_prob_loss

        super().__init__(batch_size, learning_rate, tolerance, path, verbose)

    @property
    def is_uncertain(self):
        return True

    def _construct_net(self):

        return OrdRecNet(self.num_users, self.num_items, len(self._rating_labels),
                         self._embedding_dim, self._sparse).to(self.device)

    def predict(self, user_ids, item_ids):
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

        with torch.no_grad():
            out = self._net(user_ids, item_ids)

        mean = (out * self._rating_labels).sum(1)
        var = ((out * self._rating_labels ** 2).sum(1) - mean ** 2).abs()

        return mean, var
