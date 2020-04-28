"""
The :mod:`Reliable.metrics` module provides tools for computing accuracy
metrics on a set of predictions.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    rmse
    mse
    mae
    fcp
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import numpy as np
from six import iteritems
from scipy.stats import t
from tqdm import tqdm


def rmse(predictions, verbose=False):
    """Compute RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mse(predictions, verbose=False):
    """Compute MSE (Mean Squared Error).

    .. math::
        \\text{RMSE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse_ = np.mean([float((true_r - est)**2)
                    for (_, _, true_r, est, _, _) in predictions])

    if verbose:
        print('MSE: {0:1.4f}'.format(mse_))

    return mse_


def mae(predictions, verbose=False):
    """Compute MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_


def fcp(predictions, verbose=False):
    """Compute FCP (Fraction of Concordant Pairs).

    Computed as described in paper `Collaborative Filtering on Ordinal User
    Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
    and Sill, section 5.2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp))

    return fcp


def kendallW(ranks):
    items = len(ranks[0]) #items rated
    ensemble = len(ranks) #raters
    
    tokens = [i[0] for i in ranks[0]]
    tokenized_rankings = np.empty((items, ensemble))
    for i in range(items):
        for e in range(ensemble):
            tokenized_rankings[i, e] = np.argwhere(np.array([k[0] for k in ranks[e]]) == tokens[i])[0][0]+1
            
    ri = np.sum(tokenized_rankings, axis=1)
    S = np.var(ri)
    
    return 12*S/(ensemble**2*(items**3-items))


def misscalibration(n, predictions, verbose=False):
    err = [float(abs(true_r - est)) for (_, _, true_r, est, _, _) in predictions]
    sd = [u.rel for u in predictions]
    
    p = np.arange(0.51, 0.99, 0.01)
    t_p = [t.ppf(p, df=n-1) for p in p]
    conf = 2*p-1
    
    hot = np.empty((len(p), len(predictions)))
    for b in range(len(p)):
        for u in range(len(predictions)):
            hot[b, u] = err[u] < (t_p[b]*sd[u]/np.sqrt(n))
            
    out = np.mean(abs(hot.mean(axis=1) - conf))
    if verbose:
        print('Misscalibration:  {0:1.4f}'.format(out))
        
    return out, hot, err, sd, p, t_p, conf


def build_intervals(predictions, bins=20):
    predictions = [predictions[i] for i in np.argsort([p.rel for p in predictions])]
    n = len(predictions)
    w = np.zeros((bins))
    for b in range(bins):
        predictions_ = predictions[int((b/bins)*n):int(((b+1)/bins)*n)]
        p = 0
        while(p < 0.95):
            w[b] += 0.01
            inside = 0
            for p in predictions_:
                if abs(p.est - p.r_ui) < w[b]:
                    inside += 1
            p = inside/len(predictions_)
    return w


def build_quantiles(predictions):
    quantiles = np.linspace(start=0, stop=0.95, num=20, endpoint=True)
    out = []
    for q in quantiles:
        q_ = np.quantile([a.rel for a in predictions], q)
        predictions_ = [a for a in predictions if a.rel > q_]
        out.append(rmse(predictions_))
    return out


def min_max(array):
    min = array.min()
    max = array.max()
    transformed = (array - min) / (max - min)
    return transformed


def RPI(predictions):
    err = np.array([np.abs(pred.est - pred.r_ui) for pred in predictions])
    rel = np.array([pred.rel for pred in predictions])
    rel = min_max(rel) #Scales reliability to [0, 1]
    err_deviation = err - err.mean()
    rel_deviation = rel.mean() - rel
    MAE = err.mean()
    sigma_err = err_deviation.__abs__().mean()
    sigma_rel = rel_deviation.__abs__().mean()
    RPI = (err*err_deviation*rel_deviation).mean()/(sigma_err*sigma_rel*MAE)
    return RPI

def _cumstd(array):
    out = np.zeros(len(array))
    out[0] = 0
    for i in range(1, len(array)):
        out[i] = np.std(array[:i+1])
    return out

def precision_recall_RRI(data, test, model, max_K):
    users = np.unique([d[0] for d in data.raw_ratings])
    user_is_relevant_at_k = np.zeros((len(users), max_K))
    user_n_relevant_total = np.empty(len(users))
    avg_reliability_at_k = np.zeros(max_K)
    std_reliability_at_k = np.zeros(max_K)
    for idxu, u in enumerate(tqdm(users)):
        recommendation = model.recommend(u, max_K)
        relevant_items = [t[1] for t in test if t[0] == u and t[2] >= 4]
        user_n_relevant_total[idxu] = len(relevant_items)
        reliabilities = [r[2] for r in recommendation]
        avg_reliability_at_k += np.array(reliabilities).cumsum() / range(1, max_K + 1)
        std_reliability_at_k += _cumstd(reliabilities)
        for idxr, r in enumerate(recommendation):
            if user_n_relevant_total[idxu] != 0:
                if r[0] in relevant_items:
                    user_is_relevant_at_k[idxu, idxr] = r[2]
    avg_reliability_at_k /= len(users)
    std_reliability_at_k /= len(users)
    avg_precision_at_k = (user_is_relevant_at_k != 0).cumsum(axis=1).mean(axis=0)/np.arange(1, 21)
    avg_recall_at_k = np.nanmean((user_is_relevant_at_k!=0).cumsum(axis=1)/user_n_relevant_total[:, np.newaxis], axis=0)
    reliability_deviations = user_is_relevant_at_k - (user_is_relevant_at_k!=0)*avg_reliability_at_k[np.newaxis, :]
    norm_deviation_at_k = (reliability_deviations.cumsum(axis=1) / std_reliability_at_k[np.newaxis, :]).sum(axis=0)
    RRI_at_k = norm_deviation_at_k / (user_is_relevant_at_k != 0).cumsum(axis=1).sum(axis=0)
    return avg_precision_at_k, avg_recall_at_k, RRI_at_k, user_is_relevant_at_k, reliability_deviations