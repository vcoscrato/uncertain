import pickle
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from spotlight.cross_validation import random_train_test_split as split
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.interactions import Interactions
from spotlight.evaluation import precision_recall_score, rmse_score

from Utils.utils import dataset_loader, IterativeLearner, BiasNet
from Utils.models import EnsembleRecommender, ResampleRecommender, ModelWrapper
from Utils.metrics import rpi, precision_recall_rri

# Parameters
dataset = '20M'
path = 'Results/' + dataset + '/'
random_state = 0
MF_params = {'embedding_dim': 20, 'n_iter': 1, 'l2': 0, 'learning_rate': 1e-3,
             'use_cuda': True, 'batch_size': 512, 'random_state': 0}
Learner_params = {'max_iterations': 100, 'tolerance': 0, 'random_state': 0}
n_models = 20
max_K = 20

if __name__ == '__main__':

    print('Loading the dataset...', end=' ')
    train, test = dataset_loader(dataset)
    print('DONE!')

    print('Fitting rating estimator...', end=' ')
    train_, val = split(train, random_state=np.random.RandomState(0), test_percentage=0.1)
    base_model = IterativeLearner(ExplicitFactorizationModel(**MF_params), **Learner_params)
    R = deepcopy(base_model).fit(train_, val)
    R.rmse = rmse_score(R, test)
    p, r = precision_recall_score(R, test, train, np.arange(1, max_K+1))
    R.precision = p.mean(axis=0)
    R.recall = r.mean(axis=0)
    with open(path+'fitted/R.pkl', 'wb') as f:
        pickle.dump(R, f, pickle.HIGHEST_PROTOCOL)
    print('DONE!')

    print('Fitting Ensemble Models...', end=' ')
    ensemble = EnsembleRecommender(R, n_models).fit(train_, val, test)
    p, r = precision_recall_score(ensemble, test, train, np.arange(1, max_K+1))
    ensemble.precision = p.mean(axis=0)
    ensemble.recall = r.mean(axis=0)
    #ensemble.rri = np.nanmean(rri, axis=0)
    with open(path+'fitted/ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble, f, pickle.HIGHEST_PROTOCOL)
    print('DONE!')
    f, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].plot(range(1, n_models+1), ensemble.rmse, 'b-', label='Ensemble')
    ax[0].plot(range(1, n_models+1), [ensemble.rmse[0]] * n_models, 'b--', label='Baseline')
    ax[0].set_xlabel('Number of models', Fontsize=20, labelpad=10)
    ax[0].set_ylabel('RMSE', Fontsize=20)
    ax[0].legend()
    ax[1].plot(range(1, max_K+1), ensemble.precision, 'r-', label='Ensemble precision')
    ax[1].plot(range(1, max_K+1), R.precision,  'r--', label='Baseline precision')
    ax[1].plot(range(1, max_K+1), ensemble.recall, 'g-', label='Ensemble recall')
    ax[1].plot(range(1, max_K+1), R.recall,  'g--', label='Baseline recall')
    ax[1].set_xlabel('K', Fontsize=20, labelpad=10)
    ax[1].set_ylabel('Metric@K', Fontsize=20)
    ax[1].legend()
    f.tight_layout()
    f.savefig(path + 'ensemble vs baseline.pdf')

    print('Fitting Resample Models...', end=' ')
    resample = ResampleRecommender(R, n_models).fit(train_, val, test)
    #_, _, rri = precision_recall_rri(resample, test, train, np.arange(1, max_K+1))
    resample.rri = np.nanmean(rri, axis=0)
    with open(path+'fitted/resample.pkl', 'wb') as f:
        pickle.dump(resample, f, pickle.HIGHEST_PROTOCOL)
    print('DONE!')

    print('Building the cross-validated error matrix...', end=' ')
    fold1, fold2 = split(train, random_state=np.random.RandomState(0), test_percentage=0.5)
    train_, val = split(fold1, random_state=np.random.RandomState(0), test_percentage=0.1)
    model_ = deepcopy(base_model).fit(train_, val)
    preds1 = model_.predict(fold2.user_ids, fold2.item_ids)
    train_, val = split(fold2, random_state=np.random.RandomState(0), test_percentage=0.1)
    model_ = deepcopy(base_model).fit(train_, val)
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
    error_mf = deepcopy(base_model).fit(train_errors, val_errors)
    double = ModelWrapper(rating_estimator=R, error_estimator=error_mf)
    double.error_rmse = rmse_score(error_mf, test_errors)
    double.rpi = rpi(double, test)
    #_, _, rri = precision_recall_rri(double, test, train, np.arange(1, max_K + 1))
    double.rri = np.nanmean(rri, axis=0)
    with open(path+'fitted/double.pkl', 'wb') as f:
        pickle.dump(double, f, pickle.HIGHEST_PROTOCOL)
    print('DONE!')

    print('Fitting the linear error estimator...', end=' ')
    representation = BiasNet(train_errors.num_users, train_errors.num_items)
    error_linear = IterativeLearner(ExplicitFactorizationModel(**MF_params, representation=representation),
                                    **Learner_params).fit(train_errors, val_errors)
    linear = ModelWrapper(rating_estimator=R, error_estimator=error_linear)
    linear.error_rmse = rmse_score(error_linear, test_errors)
    linear.rpi = rpi(linear, test)
    #_, _, rri = precision_recall_rri(linear, test, train, np.arange(1, max_K + 1))
    linear.rri = np.nanmean(rri, axis=0)
    with open(path+'fitted/linear.pkl', 'wb') as f:
        pickle.dump(linear, f, pickle.HIGHEST_PROTOCOL)
    print('DONE!')


