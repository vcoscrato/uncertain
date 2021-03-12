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