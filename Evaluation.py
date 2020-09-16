import numpy as np
from matplotlib import pyplot as plt
import pickle
from pandas import DataFrame as df
from Utils.utils import dataset_loader, load_models
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

dataset = '1M'
path = 'Results/' + dataset + '/'
train, test = dataset_loader(dataset)
k = np.arange(1, 11)
models = load_models(path)

# Error vs Measures
errors = {}
confidence = {}
for model in ['R', 'Ensemble', 'CPMF', 'OrdRec', 'Double', 'Linear']:
    preds = models[model].predict(test.user_ids, test.item_ids)[0]
    errors[model] = np.abs(preds[0] - test.ratings)
    confidence[model] = preds[1]

errors = np.abs(R.predict(test.user_ids, test.item_ids) - test.ratings)
f, ax = plt.subplots(ncols=3, figsize=(10, 5), sharey=True)
ax[0].plot(user_support, errors, 'o', markersize=2)
regline = sm.OLS(errors, sm.add_constant(user_support)).fit().fittedvalues
ax[0].plot(user_support[np.argsort(user_support)], regline[np.argsort(user_support)], 'r-')
ax[0].set_ylabel('Error', fontsize=20)
ax[0].set_xlabel('User support', fontsize=20)
ax[1].plot(item_support, errors, 'o', markersize=2)
regline = sm.OLS(errors, sm.add_constant(item_support)).fit().fittedvalues
ax[1].plot(item_support[np.argsort(item_support)], regline[np.argsort(item_support)], 'r-')
ax[1].set_xlabel('Item support', fontsize=20)
ax[2].plot(user_variance, errors, 'o', markersize=2)
regline = sm.OLS(errors, sm.add_constant(user_variance)).fit().fittedvalues
ax[2].plot(user_variance[np.argsort(user_variance)], regline[np.argsort(user_variance)], 'r-')
ax[2].set_xlabel('User variance', fontsize=20)
f.savefig(path + 'empirical.pdf')

# Ensemble vs Baseline comparison


f, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].plot(range(1, len(ensemble.models) + 1), ensemble.rmse, 'b-', label='Ensemble')
ax[0].plot(range(1, len(ensemble.models) + 1), [ensemble.rmse[0]] * len(ensemble.models), 'b--', label='Baseline')
ax[0].set_xlabel('Number of models', Fontsize=20, labelpad=10)
ax[0].set_xticks(range(1, len(ensemble.models) + 1))
ax[0].set_xticklabels(range(1, len(ensemble.models) + 1))
ax[0].set_ylabel('RMSE', Fontsize=20)
ax[0].legend()
ax[1].plot(k, ensemble.precision, 'r-', label='Ensemble precision')
ax[1].plot(k, R.precision, 'r--', label='Baseline precision')
ax[1].plot(k, ensemble.recall, 'g-', label='Ensemble recall')
ax[1].plot(k, R.recall, 'g--', label='Baseline recall')
ax[1].set_xticks(k)
ax[1].set_xticklabels(k)
ax[1].set_xlabel('K', Fontsize=20, labelpad=10)
ax[1].set_ylabel('Metric@K', Fontsize=20)
ax[1].legend()
f.tight_layout()
f.savefig(path + 'ensemble vs baseline.pdf')

#Error vs Multi-modeling measures


f, ax = plt.subplots(ncols=2, figsize=(10, 5), sharey=True)
preds = ensemble.predict(test.user_ids, test.item_ids)
errors = np.abs(preds[0] - test.ratings)
ax[0].plot(preds[1], errors, 'o', markersize=2)
regline = sm.OLS(errors, sm.add_constant(preds[1])).fit().fittedvalues
ax[0].plot(preds[1][np.argsort(preds[1])], regline[np.argsort(preds[1])], 'r-')
ax[0].set_ylabel('Error', fontsize=20)
ax[0].set_xlabel('Emsemble variance', fontsize=20)
preds = resample.predict(test.user_ids, test.item_ids)
errors = np.abs(preds[0] - test.ratings)
ax[1].plot(preds[1], errors, 'o', markersize=2)
regline = sm.OLS(errors, sm.add_constant(preds[1])).fit().fittedvalues
ax[1].plot(preds[1][np.argsort(preds[1])], regline[np.argsort(preds[1])], 'r-')
ax[1].set_xlabel('Resample variance', fontsize=20)
f.savefig(path + 'Multi-modeling.pdf')


f, ax = plt.subplots(ncols=2, figsize=(10, 5), sharey=True)
preds = cpmf.predict(test.user_ids, test.item_ids)
errors = np.abs(preds[0] - test.ratings)
ax[0].plot(preds[1], errors, 'o', markersize=2)
regline = sm.OLS(errors, sm.add_constant(preds[1])).fit().fittedvalues
ax[0].plot(preds[1][np.argsort(preds[1])], regline[np.argsort(preds[1])], 'r-')
ax[0].set_ylabel('Error', fontsize=20)
ax[0].set_xlabel('CPMF variance', fontsize=20)
preds = ks.predict(test.user_ids, test.item_ids)
errors = np.abs(preds[0] - test.ratings)
ax[1].plot(preds[1], errors, 'o', markersize=2)
regline = sm.OLS(errors, sm.add_constant(preds[1])).fit().fittedvalues
ax[1].plot(preds[1][np.argsort(preds[1])], regline[np.argsort(preds[1])], 'r-')
ax[1].set_xlabel('OrdRec variance', fontsize=20)
f.savefig(path + 'distributional.pdf')

f, ax = plt.subplots(figsize=(10, 5))
ax.plot(preds[1], preds[0], 'o', markersize=2)
ax.set_ylabel('Predicted')
ax.set_xlabel('Variance')




f, ax = plt.subplots(ncols=2, figsize=(10, 5), sharey=True)
preds = double.predict(test.user_ids, test.item_ids)
errors = np.abs(preds[0] - test.ratings)
ax[0].plot(preds[1], errors, 'o', markersize=1)
regline = sm.OLS(errors, sm.add_constant(preds[1])).fit().fittedvalues
ax[0].plot(preds[1][np.argsort(preds[1])], regline[np.argsort(preds[1])], 'r-')
ax[0].set_ylabel('Error', fontsize=20)
ax[0].set_xlabel('Double model predicted error', fontsize=20)
preds = linear.predict(test.user_ids, test.item_ids)
errors = np.abs(preds[0] - test.ratings)
ax[1].plot(preds[1], errors, 'o', markersize=1)
regline = sm.OLS(errors, sm.add_constant(preds[1])).fit().fittedvalues
ax[1].plot(preds[1][np.argsort(preds[1])], regline[np.argsort(preds[1])], 'r-')
ax[1].set_xlabel('Linear model predicted error', fontsize=20)
f.savefig(path + 'distributional.pdf')

print('Comparing all the methods...')
eval = df({'RMSE': [ensemble.rmse[-1]] + [R.rmse] * 3 + [cpmf.rmse, ks.rmse],
           'Error RMSE': [0] * 2 + [double.ermse, linear.ermse] + [0] * 2,
           'RPI': [-ensemble.rpi[-1], -resample.rpi[-1], -double.rpi, -linear.rpi, -cpmf.rpi, -ks.rpi],
           'Pearson correlation': [ensemble.correlation[0][0], resample.correlation[0][0], double.correlation[0][0],
                           linear.correlation[0][0], cpmf.correlation[0][0], ks.correlation[0][0]],
           'Spearman correlation': [ensemble.correlation[1][0], resample.correlation[1][0], double.correlation[1][0],
                           linear.correlation[1][0], cpmf.correlation[1][0], ks.correlation[1][0]]},
          index=['Ensemble', 'Resample', 'Double', 'Linear', 'CPMF', 'OrdRec'])
eval.round(4).to_csv(path + 'comparison.txt', index=True, header=True)

f, ax = plt.subplots(nrows=2, figsize=(10, 10))
ax[0].plot(np.arange(1, 21), ensemble.quantiles, 'b-', label='Ensemble')
ax[0].plot(np.arange(1, 21), resample.quantiles, 'g-', label='Resample')
ax[0].plot(np.arange(1, 21), double.quantiles, 'r-', label='Double')
ax[0].plot(np.arange(1, 21), linear.quantiles, 'k-', label='Linear')
ax[0].plot(np.arange(1, 21), cpmf.quantiles, 'c-', label='CPMF')
ax[0].plot(np.arange(1, 21), ks.quantiles, 'm-', label='OrdRec')
ax[0].set_xticks(np.arange(1, 21))
ax[0].set_xticklabels(np.round(np.linspace(start=0, stop=1, num=21, endpoint=True), 2))
ax[0].set_xlabel('Uncertainty quantile', Fontsize=20)
ax[0].set_ylabel('RMSE', Fontsize=20)
ax[0].legend()
ax[1].plot(np.arange(1, 21), ensemble.intervals, 'b-', label='Ensemble')
ax[1].plot(np.arange(1, 21), resample.intervals, 'g-', label='Resample')
ax[1].plot(np.arange(1, 21), double.intervals, 'r-', label='Double')
ax[1].plot(np.arange(1, 21), linear.intervals, 'k-', label='Linear')
ax[1].plot(np.arange(1, 21), cpmf.intervals, 'c-', label='CPMF')
ax[1].plot(np.arange(1, 21), ks.intervals, 'm-', label='OrdRec')
ax[1].set_xticks(np.arange(1, 21))
ax[1].set_xticklabels(np.arange(1, 21))
ax[1].set_xlabel('Uncertainty bin', Fontsize=20)
ax[1].set_ylabel(r'$\epsilon$', Fontsize=20)
ax[1].legend()
f.tight_layout()
f.savefig(path + 'comparison.pdf')

f, ax = plt.subplots(nrows=2, figsize=(5, 10))
ax[0].plot(np.arange(1, 11), R.precision, 'r-', label='Baseline')
ax[0].plot(np.arange(1, 11), ensemble.precision, 'b-', label='Ensemble')
ax[0].plot(np.arange(1, 11), cpmf.precision, 'g-', label='CPMF')
ax[0].plot(np.arange(1, 11), ks.precision, 'm-', label='OrdRec')
ax[0].set_xticks(np.arange(1, 11))
ax[0].set_xlabel('K', Fontsize=20)
ax[0].set_ylabel('Average precision at K', Fontsize=20)
ax[0].legend()
ax[1].plot(np.arange(1, 11), R.recall, 'r-', label='Baseline')
ax[1].plot(np.arange(1, 11), ensemble.recall, 'b-', label='Ensemble')
ax[1].plot(np.arange(1, 11), cpmf.recall, 'g-', label='CPMF')
ax[1].plot(np.arange(1, 11), ks.recall, 'm-', label='OrdRec')
ax[1].set_xticks(np.arange(1, 11))
ax[1].set_xticklabels(np.arange(1, 11))
ax[1].set_xlabel('K', Fontsize=20)
ax[1].set_ylabel('Average recall at K', Fontsize=20)
ax[1].legend()
f.tight_layout()
f.savefig(path + 'precision_recall.pdf')

f, ax = plt.subplots(figsize=(10, 5))
ax.plot(k, -ensemble.rri, 'b-', label='Ensemble')
ax.plot(k, -resample.rri, 'g-', label='Resample')
ax.plot(k, -double.rri, 'r-', label='Double')
ax.plot(k, -linear.rri, 'k-', label='Linear')
ax.plot(k, -cpmf.rri, 'c-', label='CPMF')
ax.plot(k, -ks.rri, 'm-', label='OrdRec')
ax.set_xticks(k)
ax.set_xticklabels(k)
ax.set_xlabel('K', Fontsize=20)
ax.set_ylabel('RRI@K', Fontsize=20)
ax.legend()
f.tight_layout()
f.savefig(path + 'comparison_rri.pdf')

from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from Utils.metrics import rpi_score
splitter = KFold(n_splits=2, shuffle=True, random_state=0)

features = {'user_support': user_support, 'item_support': item_support, 'user_variance': user_variance,
            'resample_std': resample.predict(test.user_ids, test.item_ids)[1],
            'double_MF': double.predict(test.user_ids, test.item_ids)[1],
            'linear': linear.predict(test.user_ids, test.item_ids)[1]}
error = np.abs(R.predict(test.user_ids, test.item_ids) - test.ratings)
targets = error > 1
results = df((np.zeros((8, 2))), index=list(features.keys()) + ['CPMF', 'OrdRec'], columns=['Likelihood', 'AUC'])

for train_index, test_index in splitter.split(errors):
    for f_id, f in enumerate(features):
        mod = LogisticRegression().fit(features[f][train_index].reshape(-1, 1), targets[train_index])
        preds = mod.predict_proba(features[f][test_index].reshape(-1, 1))
        results['Likelihood'][f] += np.log(preds[range(len(preds)), targets[test_index].astype(int)]).mean() / 2
        results['AUC'][f] += roc_auc_score(targets[test_index], preds[:, 1]) / 2

preds, confidence = cpmf.predict(test.user_ids, test.item_ids)
error = np.abs(preds - test.ratings)
targets = error > 1
for train_index, test_index in splitter.split(errors):
    mod = LogisticRegression().fit(confidence[train_index].reshape(-1, 1), targets[train_index])
    preds = mod.predict_proba(confidence[test_index].reshape(-1, 1))
    results['Likelihood']['CPMF'] += np.log(preds[range(len(preds)), targets[test_index].astype(int)]).mean() / 2
    results['AUC']['CPMF'] += roc_auc_score(targets[test_index], preds[:, 1]) / 2

preds, confidence = ks.predict(test.user_ids, test.item_ids)
error = np.abs(preds - test.ratings)
targets = error > 1
for train_index, test_index in splitter.split(errors):
    mod = LogisticRegression().fit(confidence[train_index].reshape(-1, 1), targets[train_index])
    preds = mod.predict_proba(confidence[test_index].reshape(-1, 1))
    results['Likelihood']['OrdRec'] += np.log(preds[range(len(preds)), targets[test_index].astype(int)]).mean() / 2
    results['AUC']['OrdRec'] += roc_auc_score(targets[test_index], preds[:, 1]) / 2





rmse = []
rpi = []

for train_index, test_index in splitter.split(x):
    # mod = LinearRegression(fit_intercept=False).fit(X=x_train, y=y_train)
    mod = RandomForestRegressor().fit(X=x[train_index], y=errors[train_index])
    fitted_errors = mod.predict(x[test_index])
    rmse.append(np.sqrt(mean_squared_error(fitted_errors, errors[test_index])))
    rpi.append(rpi_score((preds[0][test_index], fitted_errors), test.ratings[test_index]))
