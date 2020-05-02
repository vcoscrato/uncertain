import numpy as np
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from pandas import DataFrame as df
from sklearn.model_selection import StratifiedKFold, train_test_split

# Reliable is a package built on top of surprise to deal with recommendations with reliability level
from Reliable.models import EMF, RMF, Model_reliability, Heuristic_reliability
from Reliable.metrics import RPI, rmse, build_intervals, build_quantiles, precision, recall, RRI
from surprise.prediction_algorithms import BaselineOnly
from surprise.prediction_algorithms import SVD, NMF
from surprise.accuracy import rmse as srmse
from surprise import Dataset
from surprise.reader import Reader

# Parameters
dataset = 'ml-10m'
path = 'Results/' + dataset + '/'
random_state = 0
SVD_params = {'n_factors': 10, 'n_epochs': 200, 'lr_all': .005,
              'reg_all': .1, 'init_std_dev': 1, 'random_state': random_state}
u_ratio = 1
max_size = 20
Linear_params = {'method': 'sgd', 'reg': 0, 'learning_rate': .005, 'epochs': 200}
NMF_params = {'n_factors': 10, 'n_epochs': 200, 'reg_pu': .1, 'reg_qi': .1,
              'init_low': 0, 'init_high': 1, 'random_state': random_state}
cv_folds = 2
max_K = 20


def build_data(name, test_size, random_state=0):
    if name == 'ml-10m':
        read = Reader(sep='::')
        data = Dataset.load_from_file('data/ml-10m/ratings.dat', read)
    else:
        data = Dataset.load_builtin(name)
    users = [a[0] for a in data.raw_ratings]
    if name == 'jester':
        uid, ucount = np.unique(users, return_counts=True)
        relevant_users = list(uid[[a[0] for a in np.argwhere(ucount > 10)]])
        data.raw_ratings = [a for a in data.raw_ratings if a[0] in relevant_users]
        users = [a[0] for a in data.raw_ratings]
    train, test = train_test_split(data.raw_ratings, shuffle=True, random_state=random_state, test_size=test_size, stratify=users)
    data.raw_ratings = train
    test = [test[:3] for test in test]
    # Delete a few test items that are not in the training set
    train_items = np.unique([train[1] for train in train])
    for i in np.unique([test[1] for test in test]):
        if i not in train_items:
            for id in reversed(np.argwhere(np.array([test[1] for test in test]) == i)):
                del test[id[0]]
    return data, test


def metricsatk(recommendations):
    precisionatk = np.empty(max_K)
    recallatk = np.empty(max_K)
    RRIatk = np.empty(max_K)
    recommendations_ = []
    for idx in range(len(recommendations)):
        recommendations_.append(([], recommendations[idx][1]))
        print(recommendations[idx][1])
    for k in range(max_K):
        for idx in range(len(recommendations)):
            recommendations_[idx][0].append(recommendations[idx][0][k])
        precisionatk[k] = precision(recommendations_)
        recallatk[k] = recall(recommendations_)
        RRIatk[k] = RRI(recommendations_)
    return precisionatk, recallatk, RRIatk


if __name__ == '__main__':

    print('Loading the dataset...', end=' ')
    data, test = build_data(name=dataset, test_size=0.2, random_state=0)
    trainset = data.build_full_trainset()
    models = {}
    preds = {}
    recommendations = {}
    print('DONE!')

    print('Fitting rating estimator...', end=' ')
    R = SVD(**SVD_params).fit(trainset)
    R_preds = R.test(test)
    R.RMSE = srmse(R_preds, verbose=False)
    print('Fitted with RMSE: {}.'.format(R.RMSE))
    with open(path+'fitted/R.pkl', 'wb') as f:
        pickle.dump(R, f, pickle.HIGHEST_PROTOCOL)

    print('Fitting heuristic strategic...', end= '')
    models['Heuristic'] = Heuristic_reliability(R, u_ratio=u_ratio)
    preds['Heuristic'] = models['Heuristic'].test(test)
    #recommendations['Heuristic'] = models['Heuristic'].test_recommend(test, n=20, njobs=2)
    models['Heuristic'].RPI = RPI(preds['Heuristic'])
    print('Fitted with RPI = {}.'.format(models['Heuristic'].RPI))

    print('Fitting reffiting strategies...')
    models['EMF'] = EMF(initial_model=deepcopy(R), minimum_improvement=-1, max_size=max_size, verbose=True)
    models['EMF'].fit(test)
    preds['EMF'] = models['EMF'].test(test)
    #recommendations['EMF'] = models['EMF'].test_recommend(test, n=20, njobs=2)
    with open(path+'fitted/EMF.pkl', 'wb') as f:
        pickle.dump(models['EMF'], f, pickle.HIGHEST_PROTOCOL)
    models['RMF'] = RMF(initial_model=deepcopy(R), resample_size=0.8, minimum_improvement=-1, max_size=max_size, verbose=True)
    models['RMF'].fit(test)
    preds['RMF'] = models['RMF'].test(test)
    #recommendations['RMF'] = models['RMF'].test_recommend(test, n=20, njobs=2)
    with open(path+'fitted/RMF.pkl', 'wb') as f:
        pickle.dump(models['RMF'], f, pickle.HIGHEST_PROTOCOL)

    print('Evaliating optimization for the refitting methods...', end=' ')
    f, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, max_size+1), models['EMF'].val_metrics['RMSE'], 'b-', label='EMF')
    ax.plot(range(1, max_size+1), [models['RMF'].val_metrics['RMSE']]*max_size, 'b--', label='RMF')
    ax.set_xticks(range(1, max_size+1))
    ax.set_xlabel('Number of models', Fontsize=20, labelpad=10)
    ax.set_ylabel('RMSE', Fontsize=20, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.legend(loc='center left')
    ax_ = ax.twinx()
    ax_.plot(range(1, max_size+1), models['EMF'].val_metrics['RPI'], 'r-', label='EMF')
    ax_.plot(range(1, max_size+1), models['RMF'].val_metrics['RPI'], 'r--', label='RMF')
    ax_.set_ylabel('RPI', Fontsize=20, color='red')
    ax_.tick_params(axis='y', labelcolor='red')
    ax_.legend(loc='center right')
    f.tight_layout()
    f.savefig(path+'n_models.pdf')
    print('DONE!')

    print('Building the cross-validated error matrix...', end=' ')
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
    print('DONE!')

    print('Calculating the errors for the test set to evaluate error estimators...', end=' ')
    test_error = [(t.uid, t.iid, np.abs(t.est - t.r_ui)) for t in R_preds]
    print('DONE!')

    print('Fitting the error estimators...')
    E_linear = BaselineOnly(bsl_options=Linear_params, verbose=False)
    E_linear.fit(data_.build_full_trainset())
    models['Linear'] = Model_reliability(R, E_linear, trainset)
    preds['Linear'] = models['Linear'].test(test)
    #recommendations['Linear'] = models['Linear'].test_recommend(test, n=20, njobs=2)
    models['Linear'].RMSE = rmse(preds['Linear'])
    models['Linear'].ERMSE = srmse(E_linear.test(test_error), verbose=False)
    models['Linear'].RPI = RPI(preds['Linear'])
    print('Linear model fitted with RMSE = {}; ERMSE = {}; '
          'RPI = {}.'.format(models['Linear'].RMSE, models['Linear'].ERMSE, models['Linear'].RPI))
    with open(path+'fitted/Linear.pkl', 'wb') as f:
        pickle.dump(models['Linear'], f, pickle.HIGHEST_PROTOCOL)
    E_NMF = NMF(**NMF_params).fit(data_.build_full_trainset())
    models['NMF'] = Model_reliability(R, E_NMF, trainset)
    preds['NMF'] = models['NMF'].test(test)
    #recommendations['NMF'] = models['NMF'].test_recommend(test, n=20, njobs=2)
    models['NMF'].RMSE = rmse(preds['NMF'])
    models['NMF'].ERMSE = srmse(E_NMF.test(test_error), verbose=False)
    models['NMF'].RPI = RPI(preds['NMF'])
    print('NMF model fitted with RMSE = {}; ERMSE = {}; '
          'RPI = {}.'.format(models['NMF'].RMSE, models['NMF'].ERMSE, models['NMF'].RPI))
    with open(path+'fitted/NMF.pkl', 'wb') as f:
        pickle.dump(models['NMF'], f, pickle.HIGHEST_PROTOCOL)

    print('Comparing all the methods...')
    eval = {}
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

    # Interval width
    intervals = list(map(build_intervals, preds.values()))
    aes = ['r-', 'g-', 'b-', 'y-', 'm-']
    f, ax = plt.subplots(figsize=(10, 5))
    for idx, key in enumerate(preds.keys()):
        ax.plot(range(1, 21), intervals[idx], aes[idx], label=key)
    ax.set_xlabel('Reliability bin', Fontsize=20, labelpad=10)
    ax.set_ylabel('Interval half width', Fontsize=20)
    ax.set_xticks(range(1, 21))
    plt.legend()
    f.tight_layout()
    f.savefig(path + 'interval_width.pdf')

    # RMSE per reliability fold
    quantiles = list(map(build_quantiles, preds.values()))
    q = np.linspace(start=0.95, stop=0, num=20, endpoint=True)
    aes = ['r-', 'g-', 'b-', 'y-', 'm-']
    f, ax = plt.subplots(figsize=(10, 5))
    for idx, key in enumerate(preds.keys()):
        ax.plot(range(1, 21), quantiles[idx], aes[idx], label=key)
    ax.set_xlabel('Reliability bin', Fontsize=20, labelpad=10)
    ax.set_ylabel('RMSE', Fontsize=20)
    ax.set_xticks(range(1, 21))
    plt.legend()
    f.tight_layout()
    f.savefig(path + 'quantiles.pdf')

    '''
    print('Comparing models through recommendation metrics')
    recommendation_metrics = list(map(metricsatk, recommendations.values()))
    f, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, max_K + 1), recommendation_metrics[0][0], 'b-', label='Precision@K Base')
    ax.plot(range(1, max_K + 1), recommendation_metrics[0][1], 'y-+', label='Recall@K Base')
    ax.plot(range(1, max_K + 1), recommendation_metrics[1][0], 'r-', label='Precision@K Ensemble')
    ax.plot(range(1, max_K + 1), recommendation_metrics[1][1], 'g-+', label='Recall@K Ensemble')
    ax.set_xticks(range(1, max_K + 1))
    ax.set_xlabel('K', Fontsize=20, labelpad=10)
    ax.set_ylabel('Metric@K', Fontsize=20)
    ax.legend()
    f.tight_layout()
    f.savefig(path + 'precision_recall.pdf')

    f, ax = plt.subplots(figsize=(10, 5))
    aes = ['r-', 'g-', 'b-', 'y-', 'm-']
    for idx, key in enumerate(models.keys()):
        ax.plot(range(1, max_K + 1), recommendation_metrics[idx][2], aes[idx], label=key)
    ax.set_xticks(range(1, max_K + 1))
    ax.set_xlabel('K', Fontsize=20, labelpad=10)
    ax.set_ylabel('RRI@K', Fontsize=20)
    ax.legend()
    f.tight_layout()
    f.savefig(path + 'RRI_at_k.pdf')

    # User analysis
    bu = models['DMF'].SVDrel.bu
    ur = [[a[1] for a in b] for b in pretrain.trainset.ur.values()]
    nu = [len(ur[i]) for i in range(len(ur))]
    avgu = [np.mean(a) for a in ur]
    stdu = [np.std(a) for a in ur]
    simu = pretrain.compute_similarities().mean(axis=0)
    ue = [[a[1] for a in b] for b in models['DMF'].SVDrel.trainset.ur.values()]
    erru = [np.mean(a) for a in ue]
    names = ['User bias', '#Ratings by user', 'User average rating',
             'User rating deviation', 'User average similarity', 'User average error']
    eval = df(np.corrcoef((bu, nu, avgu, stdu, simu, erru)), index=names)
    eval.columns = names
    print(eval.to_string())

    # Item analysis
    bi = models['DMF'].SVDrel.bi
    ir = [[a[1] for a in b] for b in pretrain.trainset.ir.values()]
    ni = [len(ir[i]) for i in range(len(ir))]
    avgi = [np.mean(a) for a in ir]
    stdi = [np.std(a) for a in ir]
    pretrain.sim_options['user_based'] = False
    simi = pretrain.compute_similarities().mean(axis=0)
    ie = [[a[1] for a in b] for b in models['DMF'].SVDrel.trainset.ir.values()]
    erri = [np.mean(a) for a in ie]
    names = ['Item bias', '#Ratings to item', 'Item average rating',
             'Item rating deviation', 'Item average similarity', 'Item average error']
    eval = df(np.corrcoef((bi, ni, avgi, stdi, simi, erri)), index=names)
    eval.columns = names
    print(eval.to_string())
    '''