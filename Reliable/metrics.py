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


def build_intervals(predictions, bins=20, step=0.1, verbose=False):
    predictions = [predictions[i] for i in np.argsort([p.rel for p in predictions])]
    n = len(predictions)
    est = [p.est for p in predictions]
    rel = [p.rel for p in predictions]
    y = [p.true_r for p in predictions]
    w = np.zeros((20))
    for b in range(bins):
        idx = range(int((b/bins)*n), int((b+1/bins)*n))
        est_, rel_, y_ = est[idx], rel[idx], y[idx]
        p = 0
        while(p < 0.95):
            w[b] += 0.01
            inside = 0
            for est, rel, y in zip(est_, rel_, y_):
                if abs(est - y) < w[b]:
                    inside += 1
            p = inside/len(idx)
    return w


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
