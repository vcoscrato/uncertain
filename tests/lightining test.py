import optuna
import pandas as pd
import torch
from uncertain.models import FunkSVD, CPMF, OrdRec, GMF, GaussianGMF
from uncertain.datasets.movielens import get_movielens_dataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# TODO: Evaluation (maybe test_step?)

ML = get_movielens_dataset(variant='100K')
train, test = ML.split(test_percentage=0.2, seed=0)
test, val = test.split(test_percentage=0.5, seed=0)

funk_svd = FunkSVD(interactions=train, embedding_dim=50, lr=1e-3, batch_size=512, weight_decay=0)
es = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=es)
trainer.fit(funk_svd, train_dataloader=train.dataloader(512), val_dataloaders=val.dataloader(512))

cpmf = CPMF(interactions=train, embedding_dim=50, lr=1e-3, batch_size=512, weight_decay=0)
es = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=es)
trainer.fit(cpmf, train_dataloader=train.dataloader(512), val_dataloaders=val.dataloader(512))

gmf = GMF(interactions=train, embedding_dim=50, lr=1e-3, batch_size=512, weight_decay=0)
es = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=es)
trainer.fit(gmf, train_dataloader=train.dataloader(512), val_dataloaders=val.dataloader(512))

gaussiangmf = GaussianGMF(interactions=train, embedding_dim=50, lr=1e-3, batch_size=512, weight_decay=0)
es = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=es)
trainer.fit(gaussiangmf, train_dataloader=train.dataloader(512), val_dataloaders=val.dataloader(512))

train.score_labels, train.scores = torch.unique(train.scores, return_inverse=True)
val.scores = torch.unique(val.scores, return_inverse=True)[1]
ordrec = OrdRec(interactions=train, embedding_dim=50, lr=1e-3, batch_size=512, weight_decay=0)
es = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=es)
trainer.fit(ordrec, train_dataloader=train.dataloader(512), val_dataloaders=val.dataloader(512))

'''
funk_svd.evaluation = {}
evaluate_ratings(funk_svd, test)


generalized_mf = GCPMF(interactions=train, embedding_dim=50, lr=1e-3, batch_size=512, weight_decay=0)
es = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=es)
trainer.fit(generalized_mf, train_dataloader=train.dataloader(512), val_dataloaders=val.dataloader(512))
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-4)
    linear_lr = trial.suggest_float('linear_lr', 1e-2, 1e-1)
    model = GMF(interactions=train, embedding_dim=50, lr=lr, linear_lr=linear_lr, batch_size=512, weight_decay=0)
    es = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
    trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=es)
    trainer.fit(model, train_dataloader=train.dataloader(512), val_dataloaders=val.dataloader(2**12))
    return trainer.callback_metrics["val_loss"].item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
trials_summary = sorted(study.trials, key=lambda x: -np.inf if x.value is None else x.value)
trials_summary = [dict(trial_number=trial.number, score=trial.value, **trial.params) for trial in trials_summary]
trials_summary = pd.DataFrame(trials_summary)
print(trials_summary)
'''



