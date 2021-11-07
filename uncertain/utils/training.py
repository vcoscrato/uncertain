import numpy as np
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class LitProgressBar(ProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


def train(model, data):
    prog_bar = LitProgressBar()
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=False, mode='min')
    trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=[prog_bar, es], checkpoint_callback=False)
    trainer.fit(model, datamodule=data)
    return es.best_score


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