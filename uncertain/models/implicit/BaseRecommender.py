import torch
import numpy as np
from uncertain.utils import gpu, minibatch, sample_items


class BaseRecommender(object):

    def __init__(self,
                 batch_size,
                 learning_rate,
                 use_cuda,
                 path,
                 verbose=True):

        self._lr = learning_rate
        self._batch_size = batch_size
        self._use_cuda = use_cuda
        self._path = path
        self._verbose = verbose
        self._random_state = np.random.RandomState()

        self._num_users = None
        self._num_items = None
        self._net = None
        self._optimizer = None
        self._loss_func = None

        self.train_loss = None
        self.min_val_loss = float('inf')

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
    def _is_uncertain(self):
        return 'basic' not in self._desc

    @property
    def _initialized(self):
        return self._net is not None

    def _predict_process_ids(self, user_ids, item_ids):

        if item_ids is None:
            item_ids = torch.arange(self._num_items)

        if np.isscalar(user_ids):
            user_ids = torch.tensor(user_ids)

        if item_ids.size() != user_ids.size():
            user_ids = user_ids.expand(item_ids.size())

        user_var = gpu(user_ids, self._use_cuda)
        item_var = gpu(item_ids, self._use_cuda)

        return user_var.squeeze(), item_var.squeeze()

    def _get_negative_prediction(self, user_ids):

        negative_items = sample_items(
            self._num_items,
            len(user_ids),
            random_state=self._random_state)
        negative_var = gpu(torch.from_numpy(negative_items), self._use_cuda)

        negative_prediction = self._net(user_ids, negative_var)

        return negative_prediction

    def _one_epoch(self, interactions):

        epoch_loss = 0

        loader = minibatch(interactions, batch_size=self._batch_size)

        for (minibatch_num,
             (batch_user,
              batch_item)) in enumerate(loader):

            positive_predictions = self._net(batch_user, batch_item)
            negative_prediction = self._get_negative_prediction(batch_user)
            loss = self._loss_func(positive_predictions, negative_prediction)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            epoch_loss += loss.item()

        epoch_loss /= minibatch_num+1
        return epoch_loss

    def fit(self, train, validation=None):
        """
        Fit the model.
        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------
        train: :class:`uncertain.interactions.Interactions`
            The input dataset. Must have ratings.
        validation: :class:`uncertain.interactions.Interactions`
            Test dataset for iterative evaluation.
        """

        if not self._initialized:
            self._initialize(train)

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
                     (batch_user,
                      batch_item)) in enumerate(validation_loader):

                    self._net.eval()
                    positive_predictions = self._net(batch_user, batch_item)
                    negative_prediction = self._get_negative_prediction(batch_user)
                    epoch_loss += self._loss_func(positive_predictions, negative_prediction).item()

                epoch_loss /= minibatch_num + 1

            out = 'Epoch {} loss - Train: {}, Test: {}'.format(epoch, self.train_loss[-1], epoch_loss)

            if epoch_loss < self.min_val_loss:
                tol = False
                self.min_val_loss = epoch_loss
                out += ' - This is the lowest validation loss so far'
                epoch += 1
                torch.save(self._net.state_dict(), self._path)

            else:
                checkpoint = torch.load(self._path)
                self._net.load_state_dict(checkpoint)
                if tol:
                    out += ' - Validation loss did not improve. Ending training.'
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

    def recommend(self, user_id, train=None, top=10):

        predictions = self.predict(user_id)

        if not self._is_uncertain:
            predictions = -predictions
            uncertainties = None
        else:
            uncertainties = predictions[1]
            predictions = -predictions[0]

        if train is not None:
            rated = train.item_ids[train.user_ids == user_id]
            predictions[rated] = float('inf')

        idx = predictions.argsort()
        predictions = idx[:top]
        if self._is_uncertain:
            uncertainties = uncertainties[idx][:top]

        return predictions, uncertainties
