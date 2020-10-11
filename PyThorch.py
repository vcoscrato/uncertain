import pickle
import numpy as np
from copy import deepcopy
from os.path import exists

from spotlight.cross_validation import random_train_test_split as split
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.interactions import Interactions
from spotlight.evaluation import precision_recall_score, rmse_score

from Utils.utils import dataset_loader, set_params
from Utils.models import Empirical, Ensemble, Resample, BiasNet, ModelWrapper, CPMF, OrdRec

# Parameters
dataset = '100K'
path = 'Empirical study/' + dataset + '/'
MF_params, CPMF_params, OrdRec_params, multi_model_size, k = set_params(dataset)
n_replicas = 2

FunkSVD = ExplicitFactorizationModel(**MF_params)
base_dict = {'RMSE': np.zeros(n_replicas),
             'Precision': np.zeros((n_replicas, len(k))),
             'Recall': np.zeros((n_replicas, len(k)))}

uncertainty_dict = {'Correlation': np.zeros((n_replicas, 2)),
                    'EpsilonReliability': np.zeros((n_replicas, 20)),
                    'RMSEReliability': np.zeros((n_replicas, 20)),
                    'RPI': np.zeros(n_replicas),
                    'RRI': np.zeros((n_replicas, len(k)))}

results = {'baseline': deepcopy(base_dict),
           'user_support': deepcopy(uncertainty_dict),
           'item_support': deepcopy(uncertainty_dict),
           'item_variance': deepcopy(uncertainty_dict),
           'resample': deepcopy(uncertainty_dict),
           'double': deepcopy(uncertainty_dict),
           'linear': deepcopy(uncertainty_dict),
           'ensemble': {**deepcopy(base_dict), **deepcopy(uncertainty_dict)},
           'CPMF': {**deepcopy(base_dict), **deepcopy(uncertainty_dict)},
           'OrdRec': {**deepcopy(base_dict), **deepcopy(uncertainty_dict)}}

if __name__ == '__main__':

    for seed in range(n_replicas):

        print('Replica {}:'.format(seed+1))
        train, test = dataset_loader(dataset, seed=0)

        print('Fitting baseline estimator...', end=' ')
        R = deepcopy(FunkSVD)
        R.fit(train, verbose=False)
        results['baseline']['RMSE'][seed] = rmse_score(R, test)
        print('Fitted with RMSE = {}.'.format(results['baseline']['RMSE'][seed]))
        p, r = precision_recall_score(R, test, train, k)
        results['baseline']['Precision'][seed] = p.mean(axis=0)
        results['baseline']['Recall'][seed] = r.mean(axis=0)

        print('Evaluating empirical measures...', end=' ')
        model = Empirical(R, type='user_support')
        model.fit(train)
        model.evaluate(test, train, k)
        results['user_support']['Correlation'][seed] = [model.correlation[0][0], model.correlation[1][0]]
        results['user_support']['EpsilonReliability'][seed] = model.intervals
        results['user_support']['RMSEReliability'][seed] = model.quantiles
        results['user_support']['RPI'][seed] = model.rpi
        results['user_support']['RRI'][seed] = model.rri
        model = Empirical(R, type='item_support')
        model.fit(train)
        model.evaluate(test, train, k)
        results['item_support']['Correlation'][seed] = [model.correlation[0][0], model.correlation[1][0]]
        results['item_support']['EpsilonReliability'][seed] = model.intervals
        results['item_support']['RMSEReliability'][seed] = model.quantiles
        results['item_support']['RPI'][seed] = model.rpi
        results['item_support']['RRI'][seed] = model.rri
        model = Empirical(R, type='item_variance')
        model.fit(train)
        model.evaluate(test, train, k)
        results['item_variance']['Correlation'][seed] = [model.correlation[0][0], model.correlation[1][0]]
        results['item_variance']['EpsilonReliability'][seed] = model.intervals
        results['item_variance']['RMSEReliability'][seed] = model.quantiles
        results['item_variance']['RPI'][seed] = model.rpi
        results['item_variance']['RRI'][seed] = model.rri
        print('DONE!')

        model = Ensemble(R, multi_model_size)
        model.fit(train, test)
        model.evaluate(test, train, k)
        results['ensemble']['RMSE'][seed] = model.rmse
        results['ensemble']['Precision'][seed] = model.precision
        results['ensemble']['Recall'][seed] = model.recall
        results['ensemble']['Correlation'][seed] = [model.correlation[0][0], model.correlation[1][0]]
        results['ensemble']['EpsilonReliability'][seed] = model.intervals
        results['ensemble']['RMSEReliability'][seed] = model.quantiles
        results['ensemble']['RPI'][seed] = model.rpi
        results['ensemble']['RRI'][seed] = model.rri

        model = Resample(R, multi_model_size)
        model.fit(train, test)
        model.evaluate(test, train, k)
        results['resample']['Correlation'][seed] = [model.correlation[0][0], model.correlation[1][0]]
        results['resample']['EpsilonReliability'][seed] = model.intervals
        results['resample']['RMSEReliability'][seed] = model.quantiles
        results['resample']['RPI'][seed] = model.rpi
        results['resample']['RRI'][seed] = model.rri

        print('Building the cross-validated error data...', end=' ')
        fold1, fold2 = split(train, random_state=np.random.RandomState(0), test_percentage=0.5)
        model_cv = deepcopy(FunkSVD)
        model_cv.fit(fold1)
        predictions1 = model_cv.predict(fold2.user_ids, fold2.item_ids)
        model_cv = deepcopy(FunkSVD)
        model_cv.fit(fold2)
        predictions2 = model_cv.predict(fold1.user_ids, fold1.item_ids)
        user_ids = np.hstack((fold2.user_ids, fold1.user_ids))
        item_ids = np.hstack((fold2.item_ids, fold1.item_ids))
        train_errors = np.hstack((np.abs(fold2.ratings - predictions1), np.abs(fold1.ratings - predictions2)))
        train_errors = Interactions(user_ids, item_ids, train_errors)
        print('DONE!')

        print('Fitting the MF error estimator...', end=' ')
        model_error = deepcopy(FunkSVD)
        model_error.fit(train_errors)
        model = ModelWrapper(rating_estimator=R, error_estimator=model_error)
        model.evaluate(test, train, k)
        results['double']['Correlation'][seed] = [model.correlation[0][0], model.correlation[1][0]]
        results['double']['EpsilonReliability'][seed] = model.intervals
        results['double']['RMSEReliability'][seed] = model.quantiles
        results['double']['RPI'][seed] = model.rpi
        results['double']['RRI'][seed] = model.rri
        print('DONE!')

        print('Fitting the bias error estimator...', end=' ')
        model_error = deepcopy(FunkSVD)
        representation = BiasNet(train_errors.num_users, train_errors.num_items)
        model_error._representation = representation
        model_error.fit(train_errors)
        model = ModelWrapper(rating_estimator=R, error_estimator=model_error)
        model.evaluate(test, train, k)
        results['linear']['Correlation'][seed] = [model.correlation[0][0], model.correlation[1][0]]
        results['linear']['EpsilonReliability'][seed] = model.intervals
        results['linear']['RMSEReliability'][seed] = model.quantiles
        results['linear']['RPI'][seed] = model.rpi
        results['linear']['RRI'][seed] = model.rri
        print('DONE!')

        model = CPMF(**CPMF_params)
        model.fit(train, test, verbose=False)
        model.evaluate(test, train)
        results['CPMF']['RMSE'][seed] = model.rmse
        results['CPMF']['Precision'][seed] = model.precision
        results['CPMF']['Recall'][seed] = model.recall
        results['CPMF']['Correlation'][seed] = [model.correlation[0][0], model.correlation[1][0]]
        results['CPMF']['EpsilonReliability'][seed] = model.intervals
        results['CPMF']['RMSEReliability'][seed] = model.quantiles
        results['CPMF']['RPI'][seed] = model.rpi
        results['CPMF']['RRI'][seed] = model.rri

        model = OrdRec(**OrdRec_params)
        model.fit(train, test, verbose=False)
        model.evaluate(test, train)
        results['OrdRec']['RMSE'][seed] = model.rmse
        results['OrdRec']['Precision'][seed] = model.precision
        results['OrdRec']['Recall'][seed] = model.recall
        results['OrdRec']['Correlation'][seed] = [model.correlation[0][0], model.correlation[1][0]]
        results['OrdRec']['EpsilonReliability'][seed] = model.intervals
        results['OrdRec']['RMSEReliability'][seed] = model.quantiles
        results['OrdRec']['RPI'][seed] = model.rpi
        results['OrdRec']['RRI'][seed] = model.rri
