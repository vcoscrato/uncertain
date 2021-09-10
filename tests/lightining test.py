import torch
from uncertain.models import ExplicitMF, CPMF, OrdRec, GMF, GaussianGMF
from uncertain.datasets.movielens import get_movielens_dataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import BASE_COLORS

ML = get_movielens_dataset(variant='10M')
train, val, test = ML.split(test_percentage=0.1, validation_percentage=0.1, seed=0)

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
