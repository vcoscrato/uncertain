import numpy as np
import random
from Reliable.algobase import ReliableAlgoBase
from Reliable.data import build_data
from surprise import SVD, NMF
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from Reliable.metrics import rmse, RPI, build_intervals
from surprise.accuracy import rmse as srmse
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from tqdm import tqdm
from matplotlib import pyplot as plt

# Parameters
dataset = 'ml-100k'
path = 'Results/' + dataset + '/'
random_state = 0
SVD_params = {'n_factors': 10, 'n_epochs': 200, 'lr_all': .005,
              'reg_all': .1, 'init_std_dev': 1, 'random_state': random_state}
Linear_params = {'method': 'sgd', 'reg': 0, 'learning_rate': .005, 'epochs': 200}
NMF_params = {'n_factors': 10, 'n_epochs': 200, 'reg_pu': .1, 'reg_qi': .1,
              'init_low': 0, 'init_high': 1, 'random_state': 0}
cv_folds = 2
max_K = 20
n_negatives = 4

# Wrapper to generate predictions


if __name__ == '__main__':

    data, test = build_data(name=dataset, test_size=0.2, random_state=0)
    trainset = data.build_full_trainset()

    print('Fitting rating estimator...')
    R = SVD(**SVD_params).fit(trainset)
    R_preds = R.test(test)

    print('Building the cross-validated error matrix...')
    data_ = deepcopy(data)
    users = [a[0] for a in data.raw_ratings]
    np.random.seed(random_state)
    np.random.shuffle(data.raw_ratings)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=False)
    cv_error_matrix = []
    for train_index, test_index in splitter.split(data.raw_ratings, users):
        data_.raw_ratings = [data.raw_ratings[i] for i in train_index]
        test_ = [data.raw_ratings[i] for i in test_index]
        model = SVD(**SVD_params).fit(data_.build_full_trainset())
        for t in test_:
            pred = model.predict(t[0], t[1]).est
            cv_error_matrix.append((t[0], t[1], np.abs(t[2] - pred), t[3]))
    data_.raw_ratings = cv_error_matrix
    data_.reader.rating_scale = (0, np.inf)

    print('Fitting error estimators...')
    E_linear = BaselineOnly(bsl_options=Linear_params)
    E_linear.fit(data_.build_full_trainset())
    E_NMF = NMF(**NMF_params).fit(data_.build_full_trainset())
    
    print('Calculating the errors for the test set to evaluate error estimators...')
    test_error = [(t.uid, t.iid, np.abs(t.est - t.r_ui)) for t in R_preds]

    preds = {}
    print('Evaluating the linear error model: ', end='')
    wrapper_linear = Model_reliability(R, E_linear, trainset)
    preds['Linear'] = wrapper_linear.test(test)
    wrapper_linear.RMSE = round(rmse(preds['Linear']), 4)
    wrapper_linear.ERMSE = round(srmse(E_linear.test(test_error), verbose=False), 4)
    wrapper_linear.RPI = round(RPI(preds['Linear']), 4)
    print('RMSE = {}; ERMSE = {}; RPI = {}.'.format(wrapper_linear.RMSE, wrapper_linear.ERMSE, wrapper_linear.RPI))

    print('Evaluating the NMF error model: ', end='')
    wrapper_NMF = Model_reliability(R, E_NMF, trainset)
    preds['NMF'] = wrapper_NMF.test(test)
    wrapper_NMF.RMSE = round(rmse(preds['NMF']), 4)
    wrapper_NMF.ERMSE = round(srmse(E_NMF.test(test_error), verbose=False), 4)
    wrapper_NMF.RPI = round(RPI(preds['NMF']), 4)
    print('RMSE = {}; ERMSE = {}; RPI = {}.'.format(wrapper_NMF.RMSE, wrapper_NMF.ERMSE, wrapper_NMF.RPI))

    ##########################

    print('Comparing models through visual analysis')
    intervals = list(map(build_intervals, preds.values()))
    aes = ['g-', 'r-x']
    f, ax = plt.subplots(figsize=(10, 5))
    for id, key in enumerate(preds.keys()):
        ax.plot(range(1, 21), intervals[id], aes[id], label=key)
    ax.set_xlabel('Reliability bin', Fontsize=20, labelpad=10)
    ax.set_ylabel('Interval half width', Fontsize=20)
    ax.set_xticks(range(1, 21))
    plt.legend()
    f.tight_layout()
    f.savefig(path + 'model_based/interval_width.pdf')
    eval = {'Linear': [], 'NMF': []}
    preds_ = {}
    quantiles = np.linspace(start=0.95, stop=0, num=20, endpoint=True)
    for q in quantiles:
        q_dmf = np.quantile([a.rel for a in preds['Linear']], q)
        preds_['Linear'] = [a for a in preds['Linear'] if a.rel > q_dmf]
        q_emf = np.quantile([a.rel for a in preds['NMF']], q)
        preds_['NMF'] = [a for a in preds['NMF'] if a.rel > q_emf]
        eval['Linear'].append(rmse(preds_['Linear']))
        eval['NMF'].append(rmse(preds_['NMF']))
    f, ax = plt.subplots(figsize=(10, 5))
    for id, key in enumerate(eval.keys()):
        ax.plot(quantiles, eval[key], aes[id], label=key)
    ax.set_xticks(quantiles)
    ax.set_xlabel('Reliability quantile', Fontsize=20, labelpad=10)
    ax.set_ylabel('RMSE', Fontsize=20)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.legend()
    f.tight_layout()
    f.savefig(path+'model_based/quantiles.pdf')

    print('Comparing models through recommendation metrics')
    f, ax = plt.subplots(figsize=(10, 5))
    avg_precision_at_k, avg_recall_at_k, RRI_at_k, user_is_relevant_at_k, reliability_deviations = func(data, test, wrapper_linear)
    ax.plot(range(1, max_K+1), avg_precision_at_k, 'b-', label='Precision@K')
    ax.plot(range(1, max_K+1), avg_recall_at_k, 'y-+', label='Recall@K')
    ax.set_xticks(range(1, max_K+1))
    ax.set_xlabel('K', Fontsize=20, labelpad=10)
    ax.set_ylabel('Metric@K', Fontsize=20)
    ax.legend()
    f.tight_layout()
    f.savefig(path + 'model_based/precision_recall.pdf')
    a = user_is_relevant_at_k != 0
    '''
    k = 1
    a.cumsum(axis=1)[:, k].sum() / ((k+1) * 943)
    reliability_deviations[:, k][reliability_deviations[:, k] != 0]
    '''

    f, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(2, max_K+1), RRI_at_k[1:], 'g-', label='Linear')
    avg_precision_at_k, avg_recall_at_k, RRI_at_k, user_is_relevant_at_k, reliability_deviations = func(data, test, wrapper_NMF)
    ax.plot(range(2, max_K+1), RRI_at_k[1:], 'r-+', label='NMF')
    ax.set_xticks(range(2, max_K+1))
    ax.set_xlabel('K', Fontsize=20, labelpad=10)
    ax.set_ylabel('RRI@K', Fontsize=20)
    ax.legend()
    f.tight_layout()
    f.savefig(path + 'RRI_at_k.pdf')
    '''
    reliability_deviations[:, k][reliability_deviations[:, k] != 0]
    '''

