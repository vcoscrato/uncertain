import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame as df
from Utils.utils import dataset_loader, load_models

dataset = '100K'
path = 'Empirical study/' + dataset + '/'
train, test = dataset_loader(dataset, seed=0)
k = np.arange(1, 11)
models = load_models(path)


f, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].plot(range(1, 11), models['Ensemble'].rmse, 'b-', label='Ensemble')
ax[0].plot(range(1, 11), [models['Ensemble'].rmse[0]] * 10, 'b--', label='Baseline')
ax[0].set_xlabel('Number of models', Fontsize=20, labelpad=10)
ax[0].set_xticks(range(1, len(models['Ensemble'].models) + 1))
ax[0].set_xticklabels(range(1, len(models['Ensemble'].models) + 1))
ax[0].set_ylabel('RMSE', Fontsize=20)
ax[0].legend()
ax[1].plot(k, models['Ensemble'].precision, 'r-', label='Ensemble precision')
ax[1].plot(k, models['Baseline'].precision, 'r--', label='Baseline precision')
ax[1].plot(k, models['Ensemble'].recall, 'g-', label='Ensemble recall')
ax[1].plot(k, models['Baseline'].recall, 'g--', label='Baseline recall')
ax[1].set_xticks(k)
ax[1].set_xticklabels(k)
ax[1].set_xlabel('K', Fontsize=20, labelpad=10)
ax[1].set_ylabel('Metric@K', Fontsize=20)
ax[1].legend()
f.tight_layout()
f.savefig(path + 'ensemble vs baseline.pdf')


models['Ensemble'].rmse = models['Ensemble'].rmse[-1]
keys = ['Baseline', 'Ensemble', 'CPMF', 'OrdRec']
rmse = [models[key].rmse for key in keys]
df(rmse, index=keys, columns=['RMSE']).to_csv(path + 'RMSE comparison.csv', float_format='%.4f')


color=iter(plt.cm.rainbow(np.linspace(0, 1, 4)))
f, ax = plt.subplots(nrows=2, figsize=(5, 10))
for key in keys:
    c = next(color)
    ax[0].plot(np.arange(1, 11), models[key].precision, '-', color=c, label=key)
    ax[1].plot(np.arange(1, 11), models[key].recall, '-', color=c, label=key)
ax[0].set_xticks(np.arange(1, 11))
ax[0].set_xlabel('K', Fontsize=20)
ax[0].set_ylabel('Average precision at K', Fontsize=20)
ax[0].legend(ncol=2)
ax[1].set_xticks(np.arange(1, 11))
ax[1].set_xticklabels(np.arange(1, 11))
ax[1].set_xlabel('K', Fontsize=20)
ax[1].set_ylabel('Average recall at K', Fontsize=20)
ax[1].legend(ncol=2)
f.tight_layout()
f.savefig(path + 'precision and recall comparison.pdf')


models['Ensemble'].rpi = models['Ensemble'].rpi[-1]; models['Resample'].rpi = models['Resample'].rpi[-1]
keys = list(models.keys())[1:]
out = df(np.zeros((3, 9)), index=['RPI', 'Pearson', 'Spearman'], columns=keys)
for key in keys:
    out[key] = (models[key].rpi, models[key].correlation[0][0], models[key].correlation[1][0])
out.T.to_csv(path + 'RPI comparison.csv', float_format='%.4f')


color=iter(plt.cm.rainbow(np.linspace(0, 1, 9)))
f, ax = plt.subplots(nrows=2, figsize=(10, 10))
for key in keys:
    c = next(color)
    ax[0].plot(np.arange(1, 21), models[key].quantiles, '-', color=c, label=key)
    ax[1].plot(np.arange(1, 21), models[key].intervals, '-', color=c, label=key)
ax[0].set_xticks(np.arange(1, 21))
ax[0].set_xticklabels(np.round(np.linspace(start=0.05, stop=1, num=21, endpoint=True), 2))
ax[0].set_xlabel('Uncertainty quantile', Fontsize=20)
ax[0].set_ylabel('RMSE', Fontsize=20)
ax[0].legend(ncol=2)
ax[1].set_xticks(np.arange(1, 21))
ax[1].set_xticklabels(np.arange(1, 21))
ax[1].set_xlabel('Uncertainty bin', Fontsize=20)
ax[1].set_ylabel(r'$\epsilon$', Fontsize=20)
ax[1].legend(ncol=2)
f.tight_layout()
f.savefig(path + 'Graphs comparison.pdf')


color=iter(plt.cm.rainbow(np.linspace(0, 1, 9)))
f, ax = plt.subplots(figsize=(10, 5))
for key in keys:
    c = next(color)
    ax.plot(np.arange(1, 11), models[key].rri, '-', color=c, label=key)
ax.set_xticks(k)
ax.set_xticklabels(k)
ax.set_xlabel('K', Fontsize=20)
ax.set_ylabel('RRI@K', Fontsize=20)
ax.legend(ncol=2)
f.tight_layout()
f.savefig(path + 'comparison_rri.pdf')


results = df((np.zeros((2, 9))), index=['Likelihood', 'AUC'], columns=keys)
for key in keys:
    preds = models[key].predict(test.user_ids, test.item_ids)
    error = np.abs(preds[0] - test.ratings)

results.T.to_csv(path + 'classification.csv', float_format='%.4f')