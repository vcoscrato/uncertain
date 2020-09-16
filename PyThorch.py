import pickle
import numpy as np
from copy import deepcopy
from os.path import exists

from spotlight.cross_validation import random_train_test_split as split
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.interactions import Interactions
from spotlight.evaluation import precision_recall_score, rmse_score

from Utils.utils import dataset_loader, BiasNet
from Utils.models import EmpiricalMeasures, EnsembleRecommender, ResampleRecommender, ModelWrapper, CPMF, KorenSill

# Parameters
dataset = '1M'
path = 'Results/' + dataset + '/'
random_state = 0
if dataset == '1M':
    MF_params = {'embedding_dim': 50, 'n_iter': 200, 'l2': 2e-5, 'learning_rate': 0.02,
                 'batch_size': int(1e6), 'use_cuda': True, 'random_state': 0}
    CPMF_params = {'embedding_dim': 50, 'n_iter': 200, 'sigma': 0.05, 'learning_rate': 0.02,
                   'batch_size': int(1e6), 'use_cuda': True}
    OrdRec_params = {'embedding_dim': 50, 'n_iter': 200, 'l2': 2e-6, 'learning_rate': 0.02,
                     'batch_size': int(1e6), 'use_cuda': True}
n_models = 10
k = np.arange(1, 11)

if __name__ == '__main__':

    print('Loading the dataset...', end=' ')
    train, test = dataset_loader(dataset)
    print('DONE!')

    base_model = ExplicitFactorizationModel(**MF_params)
    if exists(path+'fitted/R.pkl'):
        R = pickle.load(open(path+'fitted/R.pkl', 'rb'))
    else:
        print('Fitting rating estimator...', end=' ')
        R = deepcopy(base_model)
        R.fit(train, verbose=True)
        R.rmse = rmse_score(R, test)
        print(R.rmse)
        p, r = precision_recall_score(R, test, train, k)
        R.precision = p.mean(axis=0)
        R.recall = r.mean(axis=0)
        with open(path+'fitted/R.pkl', 'wb') as f:
            pickle.dump(R, f, pickle.HIGHEST_PROTOCOL)
        print('DONE!')

    if not exists(path+'fitted/empirical.pkl'):
        model = EmpiricalMeasures(R)
        model.fit(train)
        model.evaluate(test, train, k)
        with open(path+'fitted/empirical.pkl', 'wb') as f:
            pickle.dump(R, f, pickle.HIGHEST_PROTOCOL)
        print('DONE!')

    if not exists(path+'fitted/ensemble.pkl'):
        print('Fitting Ensemble Models...', end=' ')
        model = EnsembleRecommender(R, n_models)
        model.fit(train, test)
        model.evaluate(test, train, k)
        with open(path+'fitted/ensemble.pkl', 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        print('DONE!')

    if not exists(path + 'fitted/resample.pkl'):
        print('Fitting Resample Models...', end=' ')
        model = ResampleRecommender(R, n_models)
        model.fit(train, test)
        model.evaluate(test, train, k)
        with open(path+'fitted/resample.pkl', 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        print('DONE!')

    if not exists(path + 'fitted/double.pkl') or not exists(path+'fitted/linear.pkl'):
        print('Building the cross-validated error matrix...', end=' ')
        fold1, fold2 = split(train, random_state=np.random.RandomState(0), test_percentage=0.5)
        model_cv = deepcopy(base_model)
        model_cv.fit(fold1)
        predictions1 = model_cv.predict(fold2.user_ids, fold2.item_ids)
        model_cv = deepcopy(base_model)
        model_cv.fit(fold2)
        predictions2 = model_cv.predict(fold1.user_ids, fold1.item_ids)
        user_ids = np.hstack((fold2.user_ids, fold1.user_ids))
        item_ids = np.hstack((fold2.item_ids, fold1.item_ids))
        train_errors = np.hstack((np.abs(fold2.ratings - predictions1), np.abs(fold1.ratings - predictions2)))
        train_errors = Interactions(user_ids, item_ids, train_errors)
        train_errors, val_errors = split(train_errors, test_percentage=0.1)
        test_errors = deepcopy(test)
        test_errors.ratings = np.abs(R.predict(test.user_ids, test.item_ids) - test.ratings)
        print('DONE!')

        print('Fitting the MF error estimator...', end=' ')
        model_error = deepcopy(base_model)
        model_error.fit(train_errors)
        model_error.rmse = rmse_score(model_error, test_errors)
        model = ModelWrapper(rating_estimator=R, error_estimator=model_error)
        model.evaluate(test, train, k)
        with open(path+'fitted/double.pkl', 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        print('DONE!')

        print('Fitting the model error estimator...', end=' ')
        model_error = deepcopy(base_model)
        representation = BiasNet(train_errors.num_users, train_errors.num_items)
        model_error._representation = representation
        model_error.fit(train_errors)
        model_error.rmse = rmse_score(model_error, test_errors)
        model = ModelWrapper(rating_estimator=R, error_estimator=model_error)
        model.evaluate(test, train, k)
        with open(path+'fitted/linear.pkl', 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        print('DONE!')

    if not exists(path + 'fitted/CPMF.pkl'):
        print('Fitting CPMF...', end=' ')
        model = CPMF(**CPMF_params)
        model.fit(train, test, verbose=False)
        model.evaluate(test, train)
        with open(path+'fitted/CPMF.pkl', 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        print('DONE!')

    if not exists(path + 'fitted/KorenSill.pkl'):
        print('Fitting KorenSill...', end=' ')
        model = KorenSill(**OrdRec_params)
        model.fit(train, test, verbose=True)
        model.evaluate(test, train)
        with open(path+'fitted/KorenSill.pkl', 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        print('DONE!')
