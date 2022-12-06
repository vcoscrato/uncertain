import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.metrics.pairwise import cosine_distances

import os
import pickle
import optuna
import numpy as np
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping


import torch
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
from scipy.special import comb
from matplotlib import pyplot as plt



class Data(LightningDataModule):

    def __init__(self, data):
        super().__init__()
        self.batch_size = int(1e5)

        # Remove explicit ratings if available
        if hasattr(data, 'score'):
            data = data.drop('score', 1)

        # Make sure user and item ids are consecutive integers
        data.user = data.user.factorize()[0]
        data.item = data.item.factorize()[0]

        # Shapes
        self.n_user = data.user.nunique()
        self.n_item = data.item.nunique()
        
        # Shuffle
        data = data.sample(frac=1, random_state=0)

        # Split
        val_test = data.groupby('user').apply(lambda x: x.tail(int(0.4 * len(x)))).reset_index(level=0, drop=True)
        train = data.drop(index=val_test.index)
        val = val_test.groupby('user').apply(lambda x: x.tail(int(0.5 * len(x)))).reset_index(level=0, drop=True)
        test = val_test.drop(index=val.index)

        # Heuristic measures
        self.user_support = train.groupby('user').size().to_numpy()
        self.item_support = train.groupby('item').size()
        empty = np.where(~pd.Series(np.arange(self.n_item)).isin(self.item_support.index))[0]
        empty = pd.Series(np.full(len(empty), float('NaN')), index=empty)
        self.item_support = self.item_support.append(empty).sort_index().fillna(0).to_numpy()
        
        # Training arrays
        self.train = train.to_numpy()
        self.test = test.to_numpy()
        self.val = val.to_numpy()

        # Random samples
        rng = np.random.default_rng(0)
        self.rand = {'users': rng.integers(self.n_user, size=len(self.test)),
                     'items': rng.integers(self.n_item, size=len(self.test))}
        
        # Finish
        print(f'Data prepared: {self.n_user} users, {self.n_item} items.')
        print(f'{len(self.train)} train, {len(self.val)} validation and {len(self.test)} test interactions.')
        
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, drop_last=True, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, drop_last=False, shuffle=False, num_workers=6)


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


def train(model, data, path, name):
    prog_bar = LitProgressBar()
    es = EarlyStopping(monitor='val_likelihood', min_delta=0.0001, patience=3, verbose=False, mode='max')
    cp = ModelCheckpoint(monitor='val_likelihood', dirpath=path, filename=name+'-{epoch}-{val_likelihood}', mode='max', save_weights_only=True)
    trainer = Trainer(gpus=1, min_epochs=5, max_epochs=200, logger=False, callbacks=[prog_bar, es, cp], check_val_every_n_epoch=5)
    trainer.fit(model, datamodule=data)
    return es.best_score.item(), cp.best_model_path


def run_study(name, objective=None, n_trials=0):
    file = 'tunning/' + name + '.pkl'
    if os.path.exists(file):
        with open(file, 'rb') as f:
            study = pickle.load(f)
    else:
        study = optuna.create_study(direction='maximize')
    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials)
    with open(file, 'wb') as f:
        pickle.dump(study, f, protocol=4)
    return study


def test_vanilla(model, data, max_k, name):
    
    pred = model.predict(data.test[:, 0], data.test[:, 1])
    neg = model.predict(data.test[:, 0], data.rand['items'])
    
    is_concordant = pred - neg > 0
    metrics = {'FCP': is_concordant.sum().item() / len(data.test)}
    
    precision_denom = torch.arange(1, max_k+1)
    MAP = torch.zeros((data.n_user, max_k))
    Recall = torch.zeros((data.n_user, max_k))

    for idxu, user in enumerate(tqdm(range(data.n_user), desc=name+' - Recommending')):
        
        targets = torch.tensor(data.test[data.test[:, 0] == user, 1])
        rated_train = torch.tensor(data.train[:, 1][data.train[:, 0] == user])
        rated_val = torch.tensor(data.val[:, 1][data.val[:, 0] == user])
        rated = torch.cat([rated_train, rated_val])
        
        rec, _ = model.rank(torch.tensor(user), ignored_item_ids=rated, top_n=10)
        hits = torch.isin(rec, targets, assume_unique=True)
        n_hits = hits.cumsum(0)
        
        if n_hits[-1] > 0:
            precision = n_hits / precision_denom
            MAP[idxu] = torch.cumsum(precision * hits, 0) / torch.clamp(n_hits, min=1)
            Recall[idxu] = n_hits / len(targets)

    metrics['MAP'] = MAP.mean(0).numpy()
    metrics['Recall'] = Recall.mean(0).numpy()
    
    with open('results/' + name + '.pkl', 'wb') as f:
        pickle.dump(metrics, file=f)
    
    return metrics


def test_uncertain(model, data, max_k, name):
    
    pred = model.predict(data.test[:, 0], data.test[:, 1])
    neg = model.predict(data.test[:, 0], data.rand['items'])
    
    is_concordant = pred[0] - neg[0] > 0
    unc = pred[1] + neg[1]
    
    metrics = {'FCP': is_concordant.sum().item() / len(data.test),
               'CP unc': pred[1][is_concordant].mean(),
               'DP unc': pred[1][~is_concordant].mean()}
    metrics['PUR'] = metrics['CP unc'] / metrics['DP unc']
    
    rand_preds = model.predict(data.rand['users'], data.rand['items'])
    metrics['corr_usup'] = stats.spearmanr(rand_preds[1], data.user_support[data.rand['users']].flatten())[0]
    metrics['corr_isup'] = stats.spearmanr(rand_preds[1], data.item_support[data.rand['items']].flatten())[0]
    
    f, ax = plt.subplots(ncols=2)
    ax[0].hist(rand_preds[1], density=True)
    ax[0].set_xlabel('Uncertainty')
    ax[0].set_ylabel('Density')
    ax[1].plot(rand_preds[0], rand_preds[1], 'o')
    ax[1].set_xlabel('Relevance')
    ax[1].set_ylabel('Uncertainty')
    f.tight_layout()
    f.savefig(f'plots/{name}.pdf')
    
    precision_denom = torch.arange(1, max_k+1)
    MAP = torch.zeros((data.n_user, max_k))
    Recall = torch.zeros((data.n_user, max_k))
    avg_unc = torch.zeros(data.n_user)
    URI = torch.full((data.n_user,), float('nan'))
    
    alphas = np.linspace(-1, 1, 11)
    unc_MAP = torch.zeros((data.n_user, len(alphas)))
    metrics['norm_unc'] = [[], []]
    
    
    rec_pop = torch.tensor(np.argpartition(data.item_support, -5)[-5:])

    for idxu, user in enumerate(tqdm(range(data.n_user), desc=name+' - Recommending')):
        
        targets = torch.tensor(data.test[data.test[:, 0] == user, 1])
        rated_train = torch.tensor(data.train[:, 1][data.train[:, 0] == user])
        rated_val = torch.tensor(data.val[:, 1][data.val[:, 0] == user])
        rated = torch.cat([rated_train, rated_val])
        
        rec_, score_, unc_ = model.rank(torch.tensor(user), ignored_item_ids=rated, top_n=999999)
        rec, unc = rec_[:10], unc_[:10]
        
        hits = torch.isin(rec, targets, assume_unique=True)
        n_hits = hits.cumsum(0)
        avg_unc[idxu] = unc.mean()
        
        if n_hits[-1] > 0:
            precision = n_hits / precision_denom
            MAP[idxu] = torch.cumsum(precision * hits, 0) / torch.clamp(n_hits, min=1)
            Recall[idxu] = n_hits / len(targets)
            URI[idxu] = (avg_unc[idxu] - unc[hits].mean()) / unc.std()
            
        norm_unc = (unc - unc.mean()) / unc.std()
        metrics['norm_unc'][0] += norm_unc[hits].tolist()
        metrics['norm_unc'][1] += norm_unc[~hits].tolist()
            
        # Unc rank
        for idxa, alpha in enumerate(alphas):
            score_unc = score_ + alpha * unc_
            rec_unc = rec_[torch.flip(np.argsort(score_unc), [0])[:10]]
            hits = torch.isin(rec_unc, targets, assume_unique=True)
            
            if hits.sum() > 0:
                precision = n_hits / precision_denom
                unc_MAP[idxu, idxa] = torch.sum(precision * hits) / hits.sum()
        
    metrics['MAP'] = MAP.mean(0).numpy()
    metrics['Recall'] = Recall.mean(0).numpy()
    metrics['UAC'] = stats.spearmanr(MAP[:, -1], avg_unc)[0]
    metrics['URI'] = torch.nanmean(URI, 0).item()
    metrics['unc_MAP'] = unc_MAP.mean(0).numpy()
    
    # MAP per uncertainty quantile
    quantiles = np.quantile(avg_unc, np.linspace(0.1, 1, 10))[::-1]
    MAP_quantiles = np.zeros((data.n_user, len(quantiles)))
    for user in range(data.n_user):
        for idx, quantile in enumerate(quantiles):
            if avg_unc[user] <= quantile:
                MAP_quantiles[user, idx] = MAP[user, -1]
            else:
                MAP_quantiles[user, idx] = np.nan
    metrics['MAP-covered'] = np.nanmean(MAP_quantiles, 0)
    
    with open('results/' + name + '.pkl', 'wb') as f:
        pickle.dump(metrics, file=f)
    
    return metrics
