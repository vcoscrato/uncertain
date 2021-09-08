import torch
from uncertain.models import FunkSVD, CPMF, OrdRec, GMF, GaussianGMF
from uncertain.datasets.movielens import get_movielens_dataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import BASE_COLORS

ML = get_movielens_dataset(variant='1M')
train_val, test = ML.split(test_percentage=0.1, seed=0)
train, val = train_val.split(test_percentage=0.1, seed=0)

results = {}
algs = [FunkSVD, CPMF, GMF, GaussianGMF, OrdRec]
for alg in algs:
    if alg == OrdRec:
        train.score_labels, train.scores = torch.unique(train.scores, return_inverse=True)
        val.scores = torch.unique(val.scores, return_inverse=True)[1]
    model = alg(interactions=train, embedding_dim=50, lr=1e-3, batch_size=512, weight_decay=0)
    es = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
    trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=es)
    trainer.fit(model, train_dataloader=train.dataloader(512), val_dataloaders=val.dataloader(512))
    results[f'{model}'.split('(')[0]] = model.test_ratings(test)
    results[f'{model}'.split('(')[0]].update(model.test_recommendations(test, train_val, 10, 4))

results_df = pd.DataFrame.from_dict(results, orient='Index')
ratings = results_df[['loss', 'RMSE', 'RPI', 'Classification']]
print(ratings)

colors = [c for c in list(BASE_COLORS)]
keys = results_df.index.to_list()
colors = {keys[i]:colors[i] for i in range(len(keys))}
f, ax = plt.subplots(nrows=3, figsize=(5, 10), sharex=True)
for key in keys:
    ax[0].plot(torch.arange(1, 11), results_df['Precision'][key],
               '-', color=colors[key], label=key, linewidth=3, alpha=0.6)
    ax[1].plot(torch.arange(1, 11), results_df['Recall'][key],
               '-', color=colors[key], label=key, linewidth=3, alpha=0.6)
    ax[2].plot(torch.arange(1, 11), results_df['NDCG'][key],
               '-', color=colors[key], label=key, linewidth=3, alpha=0.6)
ax[0].set_xticks(torch.arange(1, 11))
ax[0].set_xlabel('K', fontsize=20)
ax[0].set_ylabel('Precision@K', fontsize=20)
ax[0].legend(ncol=2)
ax[1].set_xlabel('K', fontsize=20)
ax[1].set_ylabel('Recall@K', fontsize=20)
ax[1].legend(ncol=2)
ax[2].set_xlabel('K', fontsize=20)
ax[2].set_ylabel('NDCG@K', fontsize=20)
ax[2].legend(ncol=2)
f.tight_layout()

f, ax = plt.subplots(figsize=(10, 5))
keys = ['CPMF', 'GaussianGMF', 'OrdRec']
for key in keys:
    ax.plot(torch.arange(1, 21), results_df['Quantile RMSE'][key],
            '-', color=colors[key], label=key, linewidth=3, alpha=0.6)
ax.set_xticks(torch.arange(1, 21))
ax.set_xticklabels([round(elem, 2) for elem in torch.linspace(start=0.05, end=1, steps=20).tolist()])
ax.set_xlabel('Uncertainty quantile', fontsize=20)
ax.set_ylabel('RMSE', fontsize=20)
ax.legend(ncol=2)
f.tight_layout()

f, ax = plt.subplots(figsize=(10, 5))
for key in keys:
    ax.plot(torch.arange(2, 11), results_df['RRI'][key].detach(),
            '-', color=colors[key], label=key, linewidth=3, alpha=0.6)
ax.set_xlabel('K', fontsize=20)
ax.set_ylabel('RRI@K', fontsize=20)
ax.legend(ncol=2)
f.tight_layout()