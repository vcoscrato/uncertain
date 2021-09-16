import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from uncertain.metrics import rmse_score, rpi_score, classification, correlation, quantile_score, get_hits, ndcg

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def train_test(model, train, val, test, max_k=10):
    prog_bar = LitProgressBar()
    es = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
    trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=[prog_bar, es], checkpoint_callback=False)
    trainer.fit(model, train_dataloader=train.dataloader(512), val_dataloaders=val.dataloader(len(val)))
    results = {}
    if train.type != 'Implicit':
        results.update(test_ratings(model, test))
    results.update(test_recommendations(model, test, train, max_k))
    results.update(test_pairwise(model, test, train))
    return results


class LitProgressBar(ProgressBar):
    def init_validation_tqdm(self) -> tqdm:
        bar = tqdm(disable=True)
        return bar


def test_ratings(model, test):
    with torch.no_grad():
        out = {}
        predictions = model.forward(test.users, test.items)
        if hasattr(model, 'score_labels'):
            predictions = model._summarize(predictions)
        if not model.is_uncertain:
            out['RMSE'] = rmse_score(predictions, test.scores)
        else:
            out['RMSE'] = rmse_score(predictions[0], test.scores)
            errors = torch.abs(test.scores - predictions[0])
            out['RPI'] = rpi_score(errors, predictions[1])
            out['Classification'] = classification(errors, predictions[1])
            out['Correlation'] = correlation(errors, predictions[1])
            out['Quantile RMSE'] = quantile_score(errors, predictions[1])
        return out


def test_recommendations(model, test, train, max_k=10):
    out = {}
    precision = []
    recall = []
    ndcg_ = []
    rri = []
    precision_denom = torch.arange(1, max_k + 1, dtype=torch.float64)
    ndcg_denom = torch.log2(precision_denom + 1)
    for user in range(test.num_users):
        targets = test.get_rated_items(user)
        if not len(targets):
            continue
        rec = model.recommend(user, train.get_rated_items(user))
        hits = get_hits(rec, targets)
        num_hit = hits.cumsum(0)
        precision.append(num_hit / precision_denom)
        recall.append(num_hit / len(targets))
        ndcg_.append(ndcg(hits, ndcg_denom))
        if model.is_uncertain and hits.sum().item() > 0:
            with torch.no_grad():
                rri_ = torch.empty(max_k - 1)
                for i in range(2, max_k + 1):
                    unc = rec.uncertainties[:i]
                    rri_[i - 2] = (unc.mean() - unc[hits[:i]]).mean() / unc.std()
                rri.append(rri_)
    out['Precision'] = torch.vstack(precision).mean(0)
    out['Recall'] = torch.vstack(recall).mean(0)
    out['NDCG'] = torch.vstack(ndcg_).mean(0)
    if len(rri) > 0:
        rri = torch.vstack(rri)
        out['RRI'] = rri.nansum(0) / (~rri.isnan()).float().sum(0)
    return out

    
def test_pairwise(model, test, train):
    auc = []
    for user in range(test.num_users):
        rec = model.recommend(user, train.get_rated_items(user), test.num_items).items.tolist()
        targets = test.get_rated_items(user).tolist()
        n_targets = len(targets)
        if not n_targets:
            continue
        targets_pos = sorted([rec.index(i) for i in targets])
        n_negative = test.num_items - n_targets
        n_after_target = [n_negative - pos + i for i, pos in enumerate(targets_pos)]
        auc.append(sum(n_after_target) / (len(n_after_target) * n_negative))
    return {'AUC': np.mean(auc)}
