import os
import pickle
import optuna
import numpy as np
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


def train(model, data, path, name, patience=5):
    prog_bar = LitProgressBar()
    if data.implicit:
        criteria = 'val_MAP'
        check_interval = 5
    else:
        criteria = 'val_likelihood'
        check_interval = 5
    es = EarlyStopping(monitor=criteria, min_delta=0.0001, patience=patience, verbose=False, mode='max')
    es_nan = EarlyStopping(monitor='train_likelihood', check_finite=True, mode='max')
    cp = ModelCheckpoint(monitor=criteria, dirpath=path, filename=name+'-{epoch}-{'+criteria+'}', mode='max', save_weights_only=True)
    trainer = Trainer(gpus=1, min_epochs=1, max_epochs=300, logger=False, callbacks=[prog_bar, es, cp, es_nan], check_val_every_n_epoch=check_interval)
    trainer.fit(model, datamodule=data)
    return es.best_score.item(), cp.best_model_path, es_nan.best_score.item()


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
