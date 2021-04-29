import os
import torch
import numpy as np
from uncertain.models.base import Recommender
from uncertain.representations import BiasNet, FunkSVDNet, CPMFNet, OrdRecNet
from uncertain.losses import funk_svd_loss, cpmf_loss, max_prob_loss


class LatentFactorRecommender(Recommender):

    def __init__(self,
                 embedding_dim,
                 batch_size,
                 initial_lr,
                 l2_penalty,
                 tolerance=1,
                 min_improvement=0,
                 sparse=False,
                 path=os.getcwd()+'tmp',
                 verbose=True,
                 max_epochs=float('inf')):

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.l2_penalty = l2_penalty
        self.tolerance = tolerance
        self.min_improvement = min_improvement
        self.sparse = sparse
        self.path = path
        self.verbose = verbose
        self.max_epochs = max_epochs

        super().__init__()

    def __repr__(self):

        if not hasattr(self, 'net'):
            net_representation = '[uninitialised]'
        else:
            net_representation = repr(self.net)

        return ('<{}: {}>'.format(
            self.__class__.__name__,
            net_representation,
        ))

    @property
    def _initialized(self):
        return hasattr(self, 'net')

    def initialize(self, interactions):

        for key, value in interactions.pass_args().items():
            setattr(self, key, value)

        self.net = self._construct_net()

        self._optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.initial_lr,
            weight_decay=self.l2_penalty
            )

    def load(self):
        if os.path.exists(self.path):
            self.net.load_state_dict(torch.load(self.path))
            if self.verbose:
                print('Previous training file encountered. Loading saved weights.')

    def _forward_pass(self, batch_users, batch_items, batch_ratings):

        predictions = self.net(batch_users, batch_items)
        if batch_ratings is None:
            predictions = self.net(batch_users, batch_items)
            negative_predictions = self._get_negative_prediction(batch_users)
            return self.loss_func(predictions, predicted_negative=negative_predictions)
        return self.loss_func(predictions, batch_ratings)

    def _get_negative_prediction(self, user_ids):

        negative_prediction = self.net(user_ids, self.sample_items(len(user_ids)))
        return negative_prediction

    def _one_epoch(self, loader):

        epoch_loss = 0

        for (minibatch_num,
             (batch_users, batch_items, batch_ratings)) in enumerate(loader):

            loss = self._forward_pass(batch_users, batch_items, batch_ratings)
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            epoch_loss += loss.item()

        epoch_loss /= minibatch_num + 1
        return epoch_loss

    def fit(self, train, validation=None):
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
        min_loss = float('inf')
        while epoch < self.max_epochs:

            train.shuffle(epoch)
            train_loader = train.minibatch(batch_size=self.batch_size)
            self.net.train()
            train_loss = self._one_epoch(train_loader)

            if validation is not None:
                validation_loader = validation.minibatch(batch_size=int(1e5))
                epoch_loss = 0
                with torch.no_grad():

                    for (minibatch_num,
                         (batch_users, batch_items, batch_ratings)) in enumerate(validation_loader):

                        self.net.eval()
                        epoch_loss += self._forward_pass(batch_users, batch_items, batch_ratings).item()

                    epoch_loss /= minibatch_num + 1

                out = 'Epoch {} loss - train: {}, validation: {}'.format(epoch, train_loss, epoch_loss)

            else:
                epoch_loss = train_loss
                out = 'Epoch {} train loss: {},'.format(epoch, epoch_loss)

            if epoch_loss < (min_loss * (1 - self.min_improvement)):
                tol = 0
                min_loss = epoch_loss
                out += ' - Loss decrease above threshold.'
                epoch += 1
                torch.save(self.net.state_dict(), self.path)

            else:
                tol += 1
                self.net.load_state_dict(torch.load(self.path))
                if tol > self.tolerance:
                    out += ' - Loss did not improve enough. Ending training.'
                    if self.verbose:
                        print(out)
                    break
                else:
                    out += ' - Loss did not improve enough. Reducing learning rate.'
                    for g in self._optimizer.param_groups:
                        g['lr'] /= 2

            if self.verbose:
                print(out)

        if self.path == os.getcwd()+'tmp':
            os.remove(self.path)

    def get_item_similarity(self, item_id, candidate_ids=None):

        if hasattr(self.net, 'item_embeddings'):
            with torch.no_grad():
                item_var = self.net.item_embeddings(torch.tensor(item_id, device=self.device))
                if candidate_ids is None:
                    candidate_ids = torch.arange(self.num_items, device=self.device)
                candidates_var = self.net.item_embeddings(candidate_ids)
                return torch.cosine_similarity(item_var, candidates_var, dim=-1)
        else:
            raise Exception('Model has no item_embeddings.')


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

    def __init__(self, **kwargs):

        self.loss_func = funk_svd_loss
        super().__init__(**kwargs)

    @property
    def is_uncertain(self):
        return False

    def _construct_net(self):

        if self.embedding_dim == 0:
            return BiasNet(self.num_users, self.num_items, self.sparse).to(self.device)
        return FunkSVDNet(self.num_users, self.num_items, self.embedding_dim, self.sparse).to(self.device)

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
            out = self.net(user_ids, item_ids)

        return out


class CPMF(LatentFactorRecommender):

    def __init__(self, **kwargs):

        self.loss_func = cpmf_loss
        super().__init__(**kwargs)

    @property
    def is_uncertain(self):
        return True

    def _construct_net(self):

        return CPMFNet(self.num_users, self.num_items, self.embedding_dim, self.sparse).to(self.device)

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
            out = self.net(user_ids, item_ids)

        return out[0], out[1]


class OrdRec(LatentFactorRecommender):

    def __init__(self, **kwargs):

        self.loss_func = max_prob_loss
        super().__init__(**kwargs)

    @property
    def is_uncertain(self):
        return True

    def _construct_net(self):

        return OrdRecNet(self.num_users, self.num_items, len(self.score_labels),
                         self.embedding_dim, self.sparse).to(self.device)

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
            out = self.net(user_ids, item_ids)

        mean = (out * self.score_labels).sum(1)
        var = ((out * self.score_labels ** 2).sum(1) - mean ** 2).abs()

        # aux = torch.vstack([mean] * len(self.score_labels)).T - self.score_labels
        # var = 1 / (1 + (out * torch.log2(1 - torch.abs(aux) / (len(self.score_labels) - 1))).sum(1))

        return mean, var
