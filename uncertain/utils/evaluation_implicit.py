import torch
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.special import comb
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from uncertain.core import VanillaRecommender, UncertainRecommender
    

def get_AP(hits):
    n_hits = hits.cumsum(0)
    if n_hits[-1] > 0:
        precision = n_hits / np.arange(1, len(hits) + 1)
        return np.sum(precision * hits) / n_hits[-1]
    else:
        return 0
    

def val_MAP(model, data):
    AP = np.zeros(data.n_user)
    for user in tqdm(range(data.n_user)):
        targets = data.val[data.val[:, 0] == user, 1]
        rated = data.train[:, 1][data.train[:, 0] == user]
        rec =  model.recommend(user, remove_items=rated, n=5)
        AP[user] = get_AP(rec.index.isin(targets))
    return np.mean(AP)


def accuracy_metrics(size, max_k, is_uncertain):
    out = {'MAP': np.zeros((size, max_k)),
           'Recall': np.zeros((size, max_k))}
    if is_uncertain:
        out['URI'] = np.zeros((size, max_k))
    return out


def test(model, data, max_k, name):
    
    metrics = {}
    
    Rating_rec = accuracy_metrics(len(data.test_users), max_k, isinstance(model, UncertainRecommender))
    
    pred = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test[:, 1]).long())
    neg = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test_negative_items).long())
    
    if isinstance(model, UncertainRecommender):
        pred, unc = pred[0], pred[1]
        test_avg_unc, test_std_unc = unc.mean(), unc.std()
        total_hits, total_hits_unc = np.zeros(max_k), np.zeros(max_k)
        avg_unc = np.zeros((len(data.test_users), max_k))
        
        is_concordant = pred - neg[0] > 0
        metrics['FCP'] = is_concordant.sum().item() / len(data.test)
        metrics['Concordant uncertainty'] = unc[is_concordant].mean()
        metrics['Discordant uncertainty'] = unc[~is_concordant].mean()
        
        rand_preds = model.predict(data.rand['users'], data.rand['items'])
        metrics['corr_usup'] = np.corrcoef(rand_preds[1], data.user_support[data.rand['users']])[0, 1]
        metrics['corr_udiv'] = np.corrcoef(rand_preds[1], data.user_diversity[data.rand['users']])[0, 1]
        metrics['corr_isup'] = np.corrcoef(rand_preds[1], data.item_support[data.rand['items']])[0, 1]
        
        '''
        metrics['unc_plot'], ax = plt.subplots(ncols=2)
        ax[0].hist(rand_preds[1])
        ax[0].set_xlabel('Uncertainty')
        ax[0].set_ylabel('Density')
        ax[1].plot(rand_preds[0], rand_preds[1], 'o')
        ax[1].set_xlabel('Score')
        ax[1].set_ylabel('Uncertainty')
        metrics['unc_plot'].tight_layout()
        '''
        
        Cuts = {'Values': np.quantile(rand_preds[1], np.linspace(0.8, 0.2, 4)),
                'Given': np.zeros(5),
                'Hits': np.zeros(5),
                'Surprise': np.zeros(5)}
        
    else:
        is_concordant = pred - neg > 0
        metrics['FCP'] = is_concordant.sum().item() / len(data.test)
    
    if hasattr(model, 'uncertain_predict_user'):
        Uncertain_rec = accuracy_metrics(len(data.test_users), max_k, True)
        
    precision_denom = np.arange(1, max_k + 1)
        
    for idxu, user in enumerate(tqdm(data.test_users, desc='Recommending')):
        targets = data.test[data.test[:, 0] == user, 1]
        rated = data.train[:, 1][data.train[:, 0] == user]
        rec = model.recommend(user, remove_items=rated, n=data.n_item - len(rated))
        top_k = rec[:max_k]
        hits = top_k.index.isin(targets)
        n_hits = hits.cumsum(0)

        if n_hits[-1] > 0:
            precision = n_hits / precision_denom
            Rating_rec['MAP'][idxu] = np.cumsum(precision * hits) / np.maximum(1, n_hits)
            Rating_rec['Recall'][idxu] = n_hits / len(targets)

        if isinstance(model, UncertainRecommender):
            unc = rec.uncertainties.to_numpy()
            top_k_unc = unc[:max_k]
            avg_unc[idxu] = top_k_unc.cumsum(0) / precision_denom
            hits_unc = (top_k_unc * hits).cumsum(0)
            total_hits += n_hits
            total_hits_unc += hits_unc
            stds = np.array([unc[:k].std() for k in range(1, max_k + 1)])
            Rating_rec['URI'][idxu] = (avg_unc[idxu] - (hits_unc / n_hits)) / stds
            
            Cuts['Given'][0] += len(top_k_unc)
            Cuts['Hits'][0] += n_hits[-1]
            surprise = data.distances[top_k.index][:, rated].min(1).sum(0)
            Cuts['Surprise'][0] += surprise

            for idxc, cut in enumerate(Cuts['Values']):
                if np.sum(top_k_unc < cut) == len(top_k_unc):
                    Cuts['Given'][idxc + 1] += len(top_k_unc)
                    Cuts['Hits'][idxc + 1] += n_hits[-1]
                    Cuts['Surprise'][idxc + 1] = surprise
                else:
                    top_k_constrained = rec[unc < cut][:max_k]
                    top_k_unc = top_k_constrained.uncertainties.to_numpy()
                    Cuts['Given'][idxc + 1] += len(top_k_unc)
                    if len(top_k_unc > 0):
                        Cuts['Hits'][idxc + 1] += top_k_constrained.index.isin(targets).sum()
                        surprise = data.distances[top_k_constrained.index][:, rated].min(1).sum(0)
                        Cuts['Surprise'][idxc + 1] += surprise
                    
            if hasattr(model, 'uncertain_predict_user'):
                top_k_unc = model.uncertain_recommend(user, threshold=rand_preds[1].mean(), remove_items=rated, n=max_k)
                if np.sum(top_k_unc.index.to_numpy() == top_k.index.to_numpy()) == max_k:
                    Uncertain_rec['MAP'][idxu] = Rating_rec['MAP'][idxu]
                    Uncertain_rec['Recall'][idxu] = Rating_rec['Recall'][idxu]
                else:
                    hits = top_k_unc.index.isin(targets)
                    n_hits = hits.cumsum(0)
                    
                    if n_hits[-1] > 0:
                        precision = n_hits / precision_denom
                        Uncertain_rec['MAP'][idxu] = np.cumsum(precision * hits) / np.maximum(1, n_hits)
                        Uncertain_rec['Recall'][idxu] = n_hits / len(targets)
    
    metrics['Rating_rec'] = {'MAP': np.mean(Rating_rec['MAP'], 0),
                             'Recall': np.mean(Rating_rec['Recall'], 0)}
                    
    if isinstance(model, UncertainRecommender):
        metrics['Rating_rec']['URI_global'] = (test_avg_unc - (total_hits_unc / total_hits)) / test_std_unc
        metrics['Rating_rec']['Unc_MAP_corr'] = np.corrcoef(Rating_rec['MAP'][:, -1], avg_unc[:, -1])[0, 1]
        metrics['Rating_rec']['URI'] = np.nanmean(Rating_rec['URI'], 0)
        
        metrics['Cuts'] = Cuts
        
        if hasattr(model, 'uncertain_predict_user'):
            metrics['Uncertain_rec'] = {'MAP': np.mean(Uncertain_rec['MAP'], 0),
                                        'Recall': np.mean(Uncertain_rec['Recall'], 0)}

    if name is not None:
        with open('results/' + name + '.pkl', 'wb') as f:
            pickle.dump(metrics, file=f)

    return metrics
