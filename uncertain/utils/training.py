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


def train(model, data, path, name):
    prog_bar = LitProgressBar()
    es = EarlyStopping(monitor='val_MAP', min_delta=0.0001, patience=3, verbose=False, mode='max')
    cp = ModelCheckpoint(monitor='val_MAP', dirpath=path, filename=name+'-{epoch}-{val_MAP}', mode='max', save_weights_only=True)
    trainer = Trainer(gpus=1, min_epochs=5, max_epochs=200, logger=False, callbacks=[prog_bar, es, cp], check_val_every_n_epoch=3)
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


def load(name, model):
    df = run_study(name, n_trials=0).trials_dataframe()
    return model.load_from_checkpoint(df.user_attrs_filename[df.value.argmax()])
