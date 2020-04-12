import numpy as np
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from pandas import DataFrame as df


#Reliable is a package built on top of surprise to deal with recommendations with reliability level
from Reliable import EMF, RMF, DMF
from Reliable.data import build_data
from Reliable.metrics import RPI, rmse, build_intervals
from surprise.model_selection.split import KFold
from surprise.model_selection.search import GridSearchCV
from surprise.prediction_algorithms import SVD

dataset = 'ml-1m'
path = 'Results/' + dataset + '/'
max_size = 20

'''
with open(path+'fitted/pretrain.pkl', 'rb') as f:
    pretrain = pickle.load(f)

models = {}
with open(path+'fitted/EMF.pkl', 'rb') as f:
    models['EMF'] = pickle.load(f)

with open(path+'fitted/RMF.pkl', 'rb') as f:
    models['RMF'] = pickle.load(f)

with open(path+'fitted/DMF.pkl', 'rb') as f:
    models['DMF'] = pickle.load(f)
'''

if __name__ == '__main__':

    #Load the dataset
    data, test = build_data(name=dataset, test_size=0.2, random_state=0)

    #Fit an initial SVD model to choose the best parameters
    param_grid = {'n_factors': [50, 100],
                  'n_epochs': [100],
                  'lr_all': [.0025, .005],
                  'reg_all': [.1],
                  'init_std_dev': [1],
                  'random_state': [0]}
    cv=KFold(n_splits=4, random_state=0, shuffle=False)
    model = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=cv, n_jobs=-2, joblib_verbose=True, refit=True)
    model.fit(data)
    print(model.best_params, model.best_score)
    pretrain = model.best_estimator['rmse']
    print(model.cv_results['mean_test_rmse'])
    with open(path+'fitted/pretrain.pkl', 'wb') as f:
        pickle.dump(pretrain, f, pickle.HIGHEST_PROTOCOL)

    models = {}
    models['EMF'] = EMF(initial_model=deepcopy(pretrain), minimum_improvement=-1, max_size=max_size, verbose=True)
    models['EMF'].fit(test)
    with open(path+'fitted/EMF.pkl', 'wb') as f:
        pickle.dump(models['EMF'], f, pickle.HIGHEST_PROTOCOL)
    models['RMF'] = RMF(initial_model=pretrain, resample_size=0.8, minimum_improvement=-1, max_size=max_size, verbose=True)
    models['RMF'].fit(test)
    with open(path+'fitted/RMF.pkl', 'wb') as f:
        pickle.dump(models['RMF'], f, pickle.HIGHEST_PROTOCOL)
    #param_rel = {'n_factors': [1], 'n_epochs': [200], 'lr_all': [.0005],
                 #'reg_all': [.1], 'init_std_dev': [.001], 'random_state': [0], 'biased': [False]}
    param_rel = {'method': 'sgd', 'reg': 0, 'learning_rate': .005, 'epochs': 200}
    models['DMF'] = DMF(initial_model=pretrain, param_rel=param_rel, type='absolute', cv_folds=2, random_state=0)
    models['DMF'].fit(data)
    test_ = models['DMF'].test(test)
    models['DMF'].val_metrics = {'RMSE': rmse(test_), 'RPI': RPI(test_)}
    print(models['DMF'].val_metrics)
    with open(path+'fitted/DMF.pkl', 'wb') as f:
        pickle.dump(models['DMF'], f, pickle.HIGHEST_PROTOCOL)

    # Optimization for the refitting methods
    f, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, max_size+1), models['EMF'].val_metrics['RMSE'], 'b-', label='EMF')
    ax.plot(range(1, max_size+1), [models['RMF'].val_metrics['RMSE']]*max_size, 'bx-', label='RMF')
    ax.set_xticks(range(1, max_size+1))
    ax.set_xlabel('Number of models', Fontsize=20)
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

    # Full RMSE and RPI
    pred = {}
    pred['DMF'] = models['DMF'].test(test)
    pred['EMF'] = models['EMF'].test(test)
    pred['RMF'] = models['RMF'].test(test)
    eval = {}
    eval['DMF'] = [rmse(pred['DMF']), RPI(pred['DMF'])]
    eval['EMF'] = [rmse(pred['EMF']), RPI(pred['EMF'])]
    eval['RMF'] = [rmse(pred['RMF']), RPI(pred['RMF'])]
    eval = df(eval, index=['RMSE', 'RPI'])
    print('Full test data metrics:\n', eval, '\n')

    # RMSE and RPI for ratings in [4, 5]
    pred_ = {}
    pred_['DMF'] = [a for a in pred['DMF'] if 4 < a.est < 5]
    pred_['EMF'] = [a for a in pred['EMF'] if 4 < a.est < 5]
    pred_['RMF'] = [a for a in pred['RMF'] if 4 < a.est < 5]
    eval['DMF'] = [rmse(pred_['DMF']), RPI(pred_['DMF'])]
    eval['EMF'] = [rmse(pred_['EMF']), RPI(pred_['EMF'])]
    eval['RMF'] = [rmse(pred_['RMF']), RPI(pred_['RMF'])]
    eval = df(eval, index=['RMSE', 'RPI'])
    print('Metrics for ratings in [4, 5]: \n {}'.format(eval, '\n'))

    # Interval width
    intervals = list(map(build_intervals, pred.values()))
    aes = ['g-', 'r-x', 'b--']
    f, ax = plt.subplots()
    for id, key in enumerate(pred.keys()):
        ax.plot(range(1, 21), intervals[id], aes[id], label=key)
    ax.set_xlabel('k', Fontsize=20)
    ax.set_ylabel('Interval half width', Fontsize=20)
    ax.set_xticks(range(1, 21))
    plt.legend()
    f.tight_layout()
    f.savefig(path + 'interval_width.pdf')

    # RMSE per reliability fold
    eval = {'DMF': [], 'EMF': [], 'RMF': []}
    quantiles = np.linspace(start=0.95, stop=0, num=20, endpoint=True)
    for q in quantiles:
        q_dmf = np.quantile([a.rel for a in pred['DMF']], q)
        pred_['DMF'] = [a for a in pred['DMF'] if a.rel > q_dmf]
        q_emf = np.quantile([a.rel for a in pred['EMF']], q)
        pred_['EMF'] = [a for a in pred['EMF'] if a.rel > q_emf]
        q_rmf = np.quantile([a.rel for a in pred['RMF']], q)
        pred_['RMF'] = [a for a in pred['RMF'] if a.rel > q_rmf]
        eval['DMF'].append(rmse(pred_['DMF']))
        eval['EMF'].append(rmse(pred_['EMF']))
        eval['RMF'].append(rmse(pred_['RMF']))
    f, ax = plt.subplots(figsize=(10, 5))
    ax.plot(quantiles, eval['DMF'], 'g-', label='DMF')
    ax.plot(quantiles, eval['EMF'], 'r-x', label='EMF')
    ax.plot(quantiles, eval['RMF'], 'b-+', label='RMF')
    ax.set_xticks(quantiles)
    ax.set_xlabel('Reliability quantile', Fontsize=20)
    ax.set_ylabel('RMSE', Fontsize=20, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.legend()
    f.tight_layout()
    f.savefig(path+'quantiles.pdf')

    # User analysis
    bu = models['DMF'].SVDrel.bu
    ur = [[a[1] for a in b] for b in pretrain.trainset.ur.values()]
    nu = [len(ur[i]) for i in range(len(ur))]
    avgu = [np.mean(a) for a in ur]
    simu = pretrain.compute_similarities().mean(axis=0)
    ue = [[a[1] for a in b] for b in models['DMF'].SVDrel.trainset.ur.values()]
    erru = [np.mean(a) for a in ue]
    names = ['User bias', '#Ratings by user', 'User average rating', 'User average similarity', 'User average error']
    eval = df(np.corrcoef((bu, nu, avgu, simu, erru)), index=names)
    eval.columns = names
    print(eval.to_string())

    # Item analysis
    bi = models['DMF'].SVDrel.bi
    ir = [[a[1] for a in b] for b in pretrain.trainset.ir.values()]
    ni = [len(ir[i]) for i in range(len(ir))]
    avgi = [np.mean(a) for a in ir]
    pretrain.sim_options['user_based'] = False
    simi = pretrain.compute_similarities().mean(axis=0)
    ie = [[a[1] for a in b] for b in models['DMF'].SVDrel.trainset.ir.values()]
    erri = [np.mean(a) for a in ie]
    names = ['Item bias', '#Ratings to item', 'Item average rating', 'Item average similarity', 'Item average error']
    eval = df(np.corrcoef((bi, ni, avgi, simi, erri)), index=names)
    eval.columns = names
    print(eval.to_string())