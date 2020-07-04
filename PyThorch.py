import pickle
import numpy as np
from copy import deepcopy
from pandas import DataFrame as df
from matplotlib import pyplot as plt

from spotlight.cross_validation import random_train_test_split as split
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.interactions import Interactions
from spotlight.evaluation import precision_recall_score, rmse_score

from Utils.utils import dataset_loader, BiasNet
from Utils.models import EnsembleRecommender, ResampleRecommender, ModelWrapper
from Utils.metrics import epi_score, eri_score, graphs_score, precision_recall_eri_score

# Parameters
dataset = '20M'
path = 'Results/' + dataset + '/'
random_state = 0
#MF_params = {'embedding_dim': 20, 'n_iter': 20, 'l2': 2e-4, 'learning_rate': 5e-4,
#             'use_cuda': True, 'batch_size': 256, 'random_state': 0}
MF_params = {'embedding_dim': 50, 'n_iter': 1, 'l2': 2e-4, 'learning_rate': 3e-2,
             'use_cuda': True, 'batch_size': 2048, 'random_state': 0}
n_models = 10
k = np.arange(1, 11)



'''
train, test = dataset_loader(dataset)
model = ExplicitFactorizationModel(**MF_params)
for i in range(200):
    model.fit(train)
    print(i, rmse_score(model, test))
    
train, test = dataset_loader(dataset)
model = ExplicitFactorizationModel(**MF_params)
for i in range(50):
    model.fit(train)
    print(i, rmse_score(model, test))

with open(path+'fitted/R.pkl', 'rb') as f:
    R = pickle. load(f)

with open(path+'fitted/ensemble.pkl', 'rb') as f:
    ensemble = pickle.load(f)

with open(path+'fitted/resample.pkl', 'rb') as f:
    resample = pickle.load(f)

with open(path+'fitted/double.pkl', 'rb') as f:
    double = pickle.load(f)

with open(path+'fitted/linear.pkl', 'rb') as f:
    linear = pickle.load(f)
'''

if __name__ == '__main__':

    print('Loading the dataset...', end=' ')
    train, test = dataset_loader(dataset)
    print('DONE!')

    print('Fitting rating estimator...', end=' ')
    base_model = ExplicitFactorizationModel(**MF_params)
    R = deepcopy(base_model)
    R.fit(train, verbose=True)
    R.rmse = rmse_score(R, test)
    p, r = precision_recall_score(R, test, train, k)
    R.precision = p.mean(axis=0)
    R.recall = r.mean(axis=0)
    with open(path+'fitted/R.pkl', 'wb') as f:
        pickle.dump(R, f, pickle.HIGHEST_PROTOCOL)
    print('DONE!')

    print('Fitting Ensemble Models...', end=' ')
    ensemble = EnsembleRecommender(R, n_models)
    ensemble.fit(train, test)
    p, r, eri = precision_recall_eri_score(ensemble, test, train, k)
    ensemble.precision = p.mean(axis=0)
    ensemble.recall = r.mean(axis=0)
    ensemble.eri = np.nanmean(eri, axis=0)
    ensemble.quantiles, ensemble.intervals = graphs_score(ensemble.predict(test.user_ids, test.item_ids), test.ratings)
    with open(path+'fitted/ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble, f, pickle.HIGHEST_PROTOCOL)
    print('DONE!')
    f, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].plot(range(1, n_models+1), ensemble.rmse, 'b-', label='Ensemble')
    ax[0].plot(range(1, n_models+1), [ensemble.rmse[0]] * n_models, 'b--', label='Baseline')
    ax[0].set_xlabel('Number of models', Fontsize=20, labelpad=10)
    ax[0].set_xticks(range(1, n_models+1))
    ax[0].set_xticklabels(range(1, n_models+1))
    ax[0].set_ylabel('RMSE', Fontsize=20)
    ax[0].legend()
    ax[1].plot(k, ensemble.precision, 'r-', label='Ensemble precision')
    ax[1].plot(k, R.precision,  'r--', label='Baseline precision')
    ax[1].plot(k, ensemble.recall, 'g-', label='Ensemble recall')
    ax[1].plot(k, R.recall,  'g--', label='Baseline recall')
    ax[1].set_xticks(k)
    ax[1].set_xticklabels(k)
    ax[1].set_xlabel('K', Fontsize=20, labelpad=10)
    ax[1].set_ylabel('Metric@K', Fontsize=20)
    ax[1].legend()
    f.tight_layout()
    f.savefig(path + 'ensemble vs baseline.pdf')

    print('Fitting Resample Models...', end=' ')
    resample = ResampleRecommender(R, n_models)
    resample.fit(train, test)
    resample.eri = np.nanmean(eri_score(resample, test, train, k), axis=0)
    resample.quantiles, resample.intervals = graphs_score(resample.predict(test.user_ids, test.item_ids), test.ratings)
    with open(path+'fitted/resample.pkl', 'wb') as f:
        pickle.dump(resample, f, pickle.HIGHEST_PROTOCOL)
    print('DONE!')

    print('Building the cross-validated error matrix...', end=' ')
    fold1, fold2 = split(train, random_state=np.random.RandomState(0), test_percentage=0.5)
    model_ = deepcopy(base_model)
    model_.fit(fold1)
    preds1 = model_.predict(fold2.user_ids, fold2.item_ids)
    model_ = deepcopy(base_model)
    model_.fit(fold2)
    preds2 = model_.predict(fold1.user_ids, fold1.item_ids)
    user_ids = np.hstack((fold2.user_ids, fold1.user_ids))
    item_ids = np.hstack((fold2.item_ids, fold1.item_ids))
    train_errors = np.hstack((np.abs(fold2.ratings - preds1), np.abs(fold1.ratings - preds2)))
    train_errors = Interactions(user_ids, item_ids, train_errors)
    train_errors, val_errors = split(train_errors, test_percentage=0.1)
    test_errors = deepcopy(test)
    test_errors.ratings = np.abs(R.predict(test.user_ids, test.item_ids) - test.ratings)
    print('DONE!')

    print('Fitting the MF error estimator...', end=' ')
    error_mf = deepcopy(base_model)
    error_mf.fit(train_errors)
    error_mf.rmse = rmse_score(error_mf, test_errors)
    double = ModelWrapper(rating_estimator=R, error_estimator=error_mf)
    preds = double.predict(test.user_ids, test.item_ids)
    double.epi = epi_score(preds, test.ratings)
    double.eri = np.nanmean(eri_score(double, test, train, k), axis=0)
    double.quantiles, double.intervals = graphs_score(preds, test.ratings)
    with open(path+'fitted/double.pkl', 'wb') as f:
        pickle.dump(double, f, pickle.HIGHEST_PROTOCOL)
    print('DONE!')

    print('Fitting the linear error estimator...', end=' ')
    error_linear = deepcopy(base_model)
    representation = BiasNet(train_errors.num_users, train_errors.num_items)
    error_linear._representation = representation
    error_linear.fit(train_errors)
    error_linear.rmse = rmse_score(error_linear, test_errors)
    linear = ModelWrapper(rating_estimator=R, error_estimator=error_linear)
    preds = linear.predict(test.user_ids, test.item_ids)
    linear.epi = epi_score(preds, test.ratings)
    linear.eri = np.nanmean(eri_score(linear, test, train, k), axis=0)
    linear.quantiles, linear.intervals = graphs_score(preds, test.ratings)
    with open(path+'fitted/linear.pkl', 'wb') as f:
        pickle.dump(linear, f, pickle.HIGHEST_PROTOCOL)
    print('DONE!')

    print('Comparing all the methods...')
    eval = df({'RMSE': [ensemble.rmse[-1]]+[R.rmse]*3, 'Error RMSE': [0]*2+[double.ermse, linear.ermse],
               'EPI': [ensemble.epi[-1], resample.epi[-1], double.epi, linear.epi]},
              index=['Ensemble', 'Resample', 'Double', 'Linear'])
    eval.round(4).to_csv(path + 'comparison.txt', index=True, header=True)

    f, ax = plt.subplots(nrows=3, figsize=(10, 10))
    ax[0].plot(np.arange(1, 21), ensemble.quantiles, 'b-', label='Ensemble')
    ax[0].plot(np.arange(1, 21), resample.quantiles, 'g-', label='Resample')
    ax[0].plot(np.arange(1, 21), double.quantiles, 'r-', label='Double')
    ax[0].plot(np.arange(1, 21), linear.quantiles, 'k-', label='Linear')
    ax[0].set_xticks(np.arange(1, 21))
    ax[0].set_xticklabels(np.round(np.linspace(start=0, stop=1, num=21, endpoint=True), 2))
    ax[0].set_xlabel('Error bin', Fontsize=20)
    ax[0].set_ylabel('RMSE', Fontsize=20)
    ax[0].legend()
    ax[1].plot(np.arange(1, 21), ensemble.intervals, 'b-', label='Ensemble')
    ax[1].plot(np.arange(1, 21), resample.intervals, 'g-', label='Resample')
    ax[1].plot(np.arange(1, 21), double.intervals, 'r-', label='Double')
    ax[1].plot(np.arange(1, 21), linear.intervals, 'k-', label='Linear')
    ax[1].set_xticks(np.arange(1, 21))
    ax[1].set_xticklabels(np.arange(1, 21))
    ax[1].set_xlabel('Error bin', Fontsize=20)
    ax[1].set_ylabel(r'$\epsilon$', Fontsize=20)
    ax[1].legend()
    ax[2].plot(k, ensemble.eri, 'b-', label='Ensemble')
    ax[2].plot(k, resample.eri, 'g-', label='Resample')
    ax[2].plot(k, double.eri, 'r-', label='Double')
    ax[2].plot(k, linear.eri, 'k-', label='Linear')
    ax[2].set_xticks(k)
    ax[2].set_xticklabels(k)
    ax[2].set_xlabel('K', Fontsize=20)
    ax[2].set_ylabel('ERI@K', Fontsize=20)
    ax[2].legend()
    f.tight_layout()
    f.savefig(path + 'comparison.pdf')


    '''
    eval['Heuristic'] = [R.RMSE, np.nan, models['Heuristic'].RPI]
    eval['EMF'] = [models['EMF'].val_metrics['RMSE'][-1], np.nan, models['EMF'].val_metrics['RPI'][-1]]
    eval['RMF'] = [models['RMF'].val_metrics['RMSE'], np.nan, models['RMF'].val_metrics['RPI'][-1]]
    eval['Linear'] = [models['Linear'].RMSE, models['Linear'].ERMSE, models['Linear'].RPI]
    eval['NMF'] = [models['NMF'].RMSE, models['NMF'].ERMSE, models['NMF'].RPI]
    eval = df(eval, index=['RMSE', 'ERMSE', 'RPI'])
    print('Full test data metrics:\n', eval.T, '\n')
    preds_ = {}
    preds_['Heuristic'] = [a for a in preds['Heuristic'] if 4 < a.est < 5]
    preds_['EMF'] = [a for a in preds['EMF'] if 4 < a.est < 5]
    preds_['RMF'] = [a for a in preds['RMF'] if 4 < a.est < 5]
    preds_['Linear'] = [a for a in preds['Linear'] if 4 < a.est < 5]
    preds_['NMF'] = [a for a in preds['NMF'] if 4 < a.est < 5]
    test_error_ = [test_error[i] for i in range(len(test_error)) if 4 < preds['Linear'][i].est < 5]
    eval['Heuristic'] = [rmse(preds_['Heuristic']), np.nan, RPI(preds_['Heuristic'])]
    eval['EMF'] = [rmse(preds_['EMF']), np.nan, RPI(preds_['EMF'])]
    eval['RMF'] = [rmse(preds_['RMF']), np.nan, RPI(preds_['RMF'])]
    eval['Linear'] = [rmse(preds_['Linear']), srmse(models['Linear'].E.test(test_error_), verbose=False), RPI(preds_['Linear'])]
    eval['NMF'] = [rmse(preds_['NMF']), srmse(models['NMF'].E.test(test_error_), verbose=False), RPI(preds_['NMF'])]
    eval = df(eval, index=['RMSE', 'ERMSE', 'RPI'])
    print('Metrics for ratings in [4, 5]: \n {}'.format(eval.T, '\n'))
    '''

    f, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k, ensemble.eri, 'b-', label='Ensemble')
    ax.plot(k, resample.eri, 'g-', label='Resample')
    ax.plot(k, double.eri, 'r-', label='Double')
    ax.plot(k, linear.eri, 'k-', label='Linear')
    ax.set_xticks(k)
    ax.set_xticklabels(k)
    ax.set_xlabel('K', Fontsize=20)
    ax.set_ylabel('URI@K', Fontsize=20)
    ax.legend()
    f.tight_layout()