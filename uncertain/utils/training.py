import numpy as np
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, EarlyStopping


class LitProgressBar(ProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


class Checkpoint(ModelCheckpoint):
    def __init__(self, filename):
        super().__init__(monitor='val_loss', filename = filename)


def train(model, data, path, name):
    prog_bar = LitProgressBar()
    es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=False, mode='min')
    cp = ModelCheckpoint(dirpath=path, filename=name+'-{epoch}-{val_loss:.4f}', monitor='val_loss')
    trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=[prog_bar, es, cp])
    trainer.fit(model, datamodule=data)


'''
precision = np.zeros(data.n_user)
    for user in range(data.n_user):
        targets = data.val[:, 1][data.val[:, 0] == user]
        rated = data.train[:, 1][data.train[:, 0] == user]
        rec = model.recommend(user, remove_items=rated, n=4)
        hits = rec.index[:4].isin(targets)
        precision[user] = hits.sum() / 4
    mean_precision = precision.mean()
    print('Model\'s validation metrics: Precision = {}, RMSE = {}'.format(mean_precision, es.best_score.sqrt().item()))
'''