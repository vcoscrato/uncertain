import torch
import numpy as np
from torch import tensor
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib.colors import BASE_COLORS

path = 'Movielens/'

exec(open(path+'results.txt').read())
evaluation = {'Baseline': baseline,
              'User support': usup,
              'Item support': isup,
              'Item variance': ivar,
              'Resample': resample,
              'Ensemble': ensemble,
              'FunkSVD-CV': funksvdcv,
              'Bias-CV': biascv,
              'CPMF': cpmf,
              'OrdRec': ordrec}

#######################################################################################################################

keys = ['Baseline', 'Ensemble', 'CPMF', 'OrdRec']
rmse = [evaluation[key]['RMSE'].item() for key in keys]
print(DataFrame(rmse, index=keys, columns=['RMSE']))

keys = list(evaluation.keys())[1:]
out = DataFrame(np.zeros((3, 9)), index=['RPI', 'Pearson', 'Spearman'], columns=keys)
for key in keys:
    out[key] = (evaluation[key]['RPI'].item(),
                evaluation[key]['Correlation'][0].item(),
                evaluation[key]['Correlation'][1].item())
print(out.T)



out = DataFrame((np.zeros((2, 9))), index=['Likelihood', 'AUC'], columns=keys)
for key in keys:
    results[key] = evaluation[key]['Classification']
print(out.T)

#######################################################################################################################

colors = [c for c in list(BASE_COLORS)]
keys = ['Baseline', 'FunkSVD-CV', 'Bias-CV', 'Ensemble', 'Resample', 'CPMF', 'OrdRec']
colors = {keys[i]:colors[i] for i in range(len(keys))}

f, ax = plt.subplots(nrows=2, figsize=(5, 10))
keys = ['Baseline', 'Ensemble', 'CPMF', 'OrdRec']
for key in keys:
    ax[0].plot(np.arange(1, 11), evaluation[key]['Precision'].cpu().detach().numpy(),
               '-', color=colors[key], label=key, linewidth=3, alpha=0.6)
    ax[1].plot(np.arange(1, 11), evaluation[key]['Recall'].cpu().detach().numpy(),
               '-', color=colors[key], label=key, linewidth=3, alpha=0.6)
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
f.savefig(path+'Plots/precision_recall.pdf')

f, ax = plt.subplots(figsize=(10, 5))
keys = ['FunkSVD-CV', 'Bias-CV', 'Ensemble', 'Resample', 'CPMF', 'OrdRec']
for key in keys:
    ax.plot(np.arange(1, 21), evaluation[key]['Quantile RMSE'], '-', color=colors[key], label=key, linewidth=3, alpha=0.6)
ax.set_xticks(np.arange(1, 21))
ax.set_xticklabels(np.round(np.linspace(start=0.05, stop=1, num=20, endpoint=True), 2))
ax.set_xlabel('Uncertainty quantile', Fontsize=20)
ax.set_ylabel('RMSE', Fontsize=20)
ax.legend(ncol=2)
f.tight_layout()
f.savefig(path+'Plots/RMSE_uncertainty.pdf')

f, ax = plt.subplots(figsize=(10, 5))
keys = ['FunkSVD-CV', 'Bias-CV', 'Ensemble', 'Resample', 'CPMF', 'OrdRec']
for key in keys:
    ax.plot(np.arange(1, 21), evaluation[key]['Quantile MAP'], '-', color=colors[key], label=key, linewidth=3, alpha=0.6)
ax.set_xticks(np.arange(1, 21))
ax.set_xticklabels(np.round(np.linspace(start=0.05, stop=1, num=20, endpoint=True), 2))
ax.set_xlabel('Uncertainty quantile', Fontsize=20)
ax.set_ylabel('MAP@10', Fontsize=20)
ax.legend(ncol=2)
f.tight_layout()
f.savefig(path+'Plots/MAP_uncertainty.pdf')

f, ax = plt.subplots(figsize=(10, 5))
keys = ['FunkSVD-CV', 'Bias-CV', 'Ensemble', 'Resample', 'CPMF', 'OrdRec']
for key in keys:
    ax.plot(np.arange(2, 11), evaluation[key]['RRI'], '-', color=colors[key], label=key, linewidth=3, alpha=0.6)
ax.set_xlabel('K', Fontsize=20)
ax.set_ylabel('URI@K', Fontsize=20)
ax.legend(ncol=2)
f.tight_layout()
f.savefig(path+'Plots/URI.pdf')

#######################################################################################################################


def recom_split_plot(baseline, certain, uncertain, metric_name, save_path=None):

    bar_width = 0.4
    cuts = ['Permissive cut', 'Median cut', 'Restrictive cut']
    certain_position = np.arange(0, 3)
    uncertain_position = certain_position + bar_width

    f, ax = plt.subplots(figsize=(6, 5))
    ax.set_ylabel(metric_name, fontsize=20)
    ax.set_xticks(certain_position + bar_width / 2)
    ax.set_xticklabels(cuts, fontsize=12)
    for pos in uncertain_position[:-1] + 3/4 * bar_width:
        ax.axvline(pos, linestyle='dashed', color='k', alpha=0.5)

    ax.axhline(baseline, label='Baseline '+metric_name)
    ax.bar(certain_position, certain, label='Certain', width=bar_width, color='g', edgecolor='k', hatch='//')
    ax.bar(uncertain_position, uncertain, label='Uncertain', width=bar_width, color='r', edgecolor='k', hatch='\\\\')
    ax.legend()

    for i in range(3):
        ax.text(x=certain_position[i], y=certain[i]+baseline/100, s=certain[i], ha='center')
        ax.text(x=uncertain_position[i], y=uncertain[i]+baseline/100, s=uncertain[i], ha='center')

    f.tight_layout()
    f.show()


baseline = 0.067
certain = np.array((0.079, 0.084, 0.097))
uncertain = np.array((0.002, 0.011, 0.032))
recom_split_plot(baseline, certain, uncertain, 'Hit rate')

baseline = 0.383
certain = np.array((0.344, 0.341, 0.337))
uncertain = np.array((0.528, 0.495, 0.432))
recom_split_plot(baseline, certain, uncertain, 'Surprise')
