"""
The :mod:`surprise.prediction_algorithms.algo_base` module defines the base
class :class:`AlgoBase` from which every single prediction algorithm has to
inherit.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from surprise import PredictionImpossible
from surprise import similarities as sims
from .metrics import kendallW
from collections import namedtuple


class ReliablePrediction(namedtuple('Prediction',
                                     ['uid', 'iid', 'r_ui', 'est', 'rel', 'details'])):
    """A named tuple for storing the results of a reliable prediction.
    It's wrapped in a class, but only for documentation and printing purposes.
    Args:
        uid: The (raw) user id. See :ref:`this note<raw_inner_note>`.
        iid: The (raw) item id. See :ref:`this note<raw_inner_note>`.
        r_ui(float): The true rating :math:`r_{ui}`.
        est(float): The estimated rating :math:`\\hat{r}_{ui}`.
        rel(float): The estimated rating reliability.
        details (dict): Stores additional details about the prediction that
            might be useful for later analysis.
    """

    __slots__ = ()  # for memory saving purpose.

    def __str__(self):
        s = 'user: {uid:<10} '.format(uid=self.uid)
        s += 'item: {iid:<10} '.format(iid=self.iid)
        if self.r_ui is not None:
            s += 'r_ui = {r_ui:1.2f}   '.format(r_ui=self.r_ui)
        else:
            s += 'r_ui = None   '
        s += 'est = {est:1.2f}   '.format(est=self.est)
        if self.rel is not None:
            s += 'rel = {rel:1.2f}   '.format(rel=self.rel)
        s += str(self.details)

        return s


class ReliableRanking(namedtuple('Ranking', ['uid', 'iid', 'avg_rank', 'rel', 'w'])):
    __slots__ = ()

    def __str__(self):
        s = 'user: {uid}  -  '.format(uid=self.uid)
        s += 'Kendall\'s W: {w:1.2f} \n'.format(w=self.w)
        for i in range(len(self.iid)):
            s += 'Top {j}: '.format(j=i + 1)
            s += '{iid:<10} '.format(iid=self.iid[i])
            s += 'avg_rank = {avg_rank:1.2f}     '.format(avg_rank=self.avg_rank[i])
            s += 'rel = {rel:1.2f} \n'.format(rel=self.rel[i])
        return s


class ReliableAlgoBase(object):
    """Abstract class where is defined the basic behavior of a prediction
    algorithm.

    Keyword Args:
        baseline_options(dict, optional): If the algorithm needs to compute a
            baseline estimate, the ``baseline_options`` parameter is used to
            configure how they are computed. See
            :ref:`baseline_estimates_configuration` for usage.
    """

    def __init__(self, **kwargs):

        self.bsl_options = kwargs.get('bsl_options', {})
        self.sim_options = kwargs.get('sim_options', {})
        if 'user_based' not in self.sim_options:
            self.sim_options['user_based'] = True

    def fit(self, trainset):
        """Train an algorithm on a given training set.

        This method is called by every derived class as the first basic step
        for training an algorithm. It basically just initializes some internal
        structures and set the self.trainset attribute.

        Args:
            trainset(:obj:`Trainset <surprise.Trainset>`) : A training
                set, as returned by the :meth:`folds
                <surprise.dataset.Dataset.folds>` method.

        Returns:
            self
        """

        self.trainset = trainset

        # (re) Initialise baselines
        self.bu = self.bi = None

        return self

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        """Compute the rating prediction for given user and item.

        The ``predict`` method converts raw ids to inner ids and then calls the
        ``estimate`` method which is defined in every derived class. If the
        prediction is impossible (e.g. because the user and/or the item is
        unkown), the prediction is set according to :meth:`default_prediction()
        <surprise.prediction_algorithms.algo_base.AlgoBase.default_prediction>`.

        Args:
            uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
            iid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`.
            r_ui(float): The true rating :math:`r_{ui}`. Optional, default is
                ``None``.
            clip(bool): Whether to clip the estimation into the rating scale.
                For example, if :math:`\\hat{r}_{ui}` is :math:`5.5` while the
                rating scale is :math:`[1, 5]`, then :math:`\\hat{r}_{ui}` is
                set to :math:`5`. Same goes if :math:`\\hat{r}_{ui} < 1`.
                Default is ``True``.
            verbose(bool): Whether to print details of the prediction.  Default
                is False.

        Returns:
            A :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` object
            containing:

            - The (raw) user id ``uid``.
            - The (raw) item id ``iid``.
            - The true rating ``r_ui`` (:math:`\\hat{r}_{ui}`).
            - The estimated rating (:math:`\\hat{r}_{ui}`).
            - The estimation reliability.
            - Some additional details about the prediction that might be useful
              for later analysis.
        """

        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:
            a = self.estimate(iuid, iiid)
            est, rel = a

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)
        
        if self.trainset:
            for i in self.trainset.ur[iuid]:
                if i[0] == iiid:
                    r_ui = i[1]

        pred = ReliablePrediction(uid, iid, r_ui, est, rel, details)

        if verbose:
            print(pred)

        return pred

    def default_prediction(self):
        '''Used when the ``PredictionImpossible`` exception is raised during a
        call to :meth:`predict()
        <surprise.prediction_algorithms.algo_base.AlgoBase.predict>`. By
        default, return the global mean of all ratings (can be overridden in
        child classes).

        Returns:
            (float): The mean of all ratings in the trainset.
        '''

        return self.trainset.global_mean

    def test(self, testset, build_correlation=False, verbose=False):
        """Test the algorithm on given testset, i.e. estimate all the ratings
        in the given testset.

        Args:
            testset: A test set, as returned by a :ref:`cross-validation
                itertor<use_cross_validation_iterators>` or by the
                :meth:`build_testset() <surprise.Trainset.build_testset>`
                method.
            verbose(bool): Whether to print details for each predictions.
                Default is False.

        Returns:
            A list of :class:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` objects
            that contains all the estimated ratings.
        """

        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]

        if build_correlation:
            uids = np.unique([i.uid for i in predictions])
            iids = np.unique([i.iid for i in predictions])
            nu = [len(self.trainset.ir[self.trainset.to_inner_uid(uid)]) for uid in uids]
            ni = np.empty((len(iids)))
            avgu = [np.mean([t[1] for t in self.trainset.ur[self.trainset.to_inner_uid(uid)]]) for uid in uids]
            avgi = np.empty((len(iids)))
            simu = self.compute_similarities().mean(axis=0)
            self.sim_options['user_based'] = False
            simi = self.compute_similarities().mean(axis=0)
            self.sim_options['user_based'] = True
            for i, iid in enumerate(iids):
                try:
                    ni[i] = len(self.trainset.ir[self.trainset.to_inner_iid(iid)])
                    avgi[i] = np.mean([t[1] for t in self.trainset.ir[self.trainset.to_inner_iid(iid)]])
                except:
                    print('a')
                    ni[i] = 0
                    avgi[i] = self.trainset.global_mean
            out = np.empty((len(predictions), 7))
            for i in range(len(predictions)):
                out[i] = [predictions[i].rel,
                          nu[np.where(predictions[i].uid == uids)[0][0]], ni[np.where(predictions[i].iid == iids)[0][0]],
                          avgu[np.where(predictions[i].uid == uids)[0][0]], avgi[np.where(predictions[i].iid == iids)[0][0]],
                          simu[self.trainset.to_inner_uid(predictions[i].uid)], simi[self.trainset.to_inner_iid(predictions[i].iid)]]
            names = ['r_user', 'r_item', 'avg_user', 'avg_item', 'sim_user', 'sim_item']
            corr = np.corrcoef(out, rowvar=False)[0][1:]
            corr_dict = {}
            for i in range(6):
                corr_dict[names[i]] = corr[i]
            return predictions, corr_dict
        return predictions

    def rank(self, uid, n=10, iid_list=None, remove_rated=True, W=False):
        """Build a recommendation rank for a given user.
        
        Args:
            uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
            n(int): The desired number of recommendations.
            iid_list(list): A list containing the (Raw) iids to be considered when
                ranking. If None, consider all items.
            remove_rated(bool): If True, items already rated by the user wont be 
                part of the ranking.
            W(bool): If True, return the Kendall's coefficient of concordance (W)
                among the base models.
                
        Returns:
            A list containing tuples: the (Raw) ids of the predicted item, its 
                estimated rating and reliability std. The list is in decreasing
                order of preference.
        """
        if iid_list:
            item_set = [self.trainset.to_inner_iid(i) for i in iid_list]
        else:
            item_set = self.trainset.all_items()
            
        if remove_rated:
            item_set = [i for i in item_set if i not in 
                        [i[0] for i in self.trainset.ur[0]]]
        
        iuid = self.trainset.to_inner_uid(uid)
        preds = [[iiid, self.estimate(iuid, iiid)] for iiid in item_set]
        est = [i[1][0] for i in preds]
        sort_idx = sorted(range(len(est)), key=est.__getitem__, reverse=True)
        rank = [[self.trainset.to_raw_iid(preds[i][0]), preds[i][1]] for i in 
                sort_idx[:n]]
        
        if not W:
            return rank
        else:
            rank_items = [r[0] for r in rank]
            ranks = [i.rank('1', iid_list=rank_items) for i in self.models]
            W = kendallW(ranks)
            
            return rank, W

    def compute_similarities(self):
        """Build the similarity matrix.
        The way the similarity matrix is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).
        This method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.
        Returns:
            The similarity matrix."""

        construction_func = {'cosine': sims.cosine,
                             'msd': sims.msd,
                             'pearson': sims.pearson,
                             'pearson_baseline': sims.pearson_baseline}

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        min_support = self.sim_options.get('min_support', 1)

        args = [n_x, yr, min_support]

        name = self.sim_options.get('name', 'msd').lower()
        if name == 'pearson_baseline':
            shrinkage = self.sim_options.get('shrinkage', 100)
            bu, bi = self.compute_baselines()
            if self.sim_options['user_based']:
                bx, by = bu, bi
            else:
                bx, by = bi, bu

            args += [self.trainset.global_mean, bx, by, shrinkage]

        try:
            sim = construction_func[name](*args)
            return sim
        except KeyError:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')
