import os
import pickle
import optuna
import numpy as np
from tqdm.notebook import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar
    

def train(model, data, path, name, patience=3, check_interval=5, early_stopping=True):
    prog_bar = LitProgressBar()
    if data.implicit:
        criteria = 'val_MAP'
    else:
        criteria = 'val_likelihood'
    es_nan = EarlyStopping(monitor='train_likelihood', check_finite=True, mode='max')
    
    if early_stopping:
        es = EarlyStopping(monitor=criteria, min_delta=0.0001, patience=patience, verbose=False, mode='max')
        cp = ModelCheckpoint(monitor=criteria, mode='max', dirpath=path, filename=name+'-{epoch}-{'+criteria+'}', save_weights_only=True)
        cp = ModelCheckpoint(monitor=criteria, mode='max', dirpath=path, filename=name+'-{epoch}-{'+criteria+'}', save_weights_only=True, every_n_epochs=2, save_last=True)
        trainer = Trainer(gpus=1, min_epochs=10, max_epochs=300, logger=False, callbacks=[prog_bar, es, cp, es_nan], check_val_every_n_epoch=check_interval)
        trainer.fit(model, datamodule=data)
    else:
        cp = ModelCheckpoint(monitor=criteria, mode='max', dirpath=path, filename=name+'-{epoch}-{'+criteria+'}', every_n_epochs=2, save_weights_only=True, save_last=True)
        trainer = Trainer(gpus=1, min_epochs=10, max_epochs=20, logger=False, callbacks=[prog_bar, cp, es_nan], check_val_every_n_epoch=2)
        trainer.fit(model, datamodule=data)

    return cp.best_model_score.item(), cp.best_model_path, es_nan.best_score.item()


def run_study(name, objective=None, n_trials=0, **kwargs):
    file = 'tunning/' + name + '.pkl'
    if os.path.exists(file):
        with open(file, 'rb') as f:
            study = pickle.load(f)
    else:
        study = optuna.create_study(direction='maximize', **kwargs)
    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials)
    with open(file, 'wb') as f:
        pickle.dump(study, f, protocol=4)
    return study


def load(model, study, top=0):
    sorted_runs = study.trials_dataframe().sort_values('value')[::-1]
    model = model.load_from_checkpoint(sorted_runs.user_attrs_filename.iloc[top])
    model.eval()
    return model, sorted_runs
