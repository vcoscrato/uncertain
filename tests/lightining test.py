import torch
from uncertain.models import ExplicitMF, CPMF, OrdRec, GMF, GaussianGMF, ImplicitGaussianGMF, ImplicitMF
from uncertain.datasets.movielens import get_movielens_dataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import BASE_COLORS

ML = get_movielens_dataset(variant='1M')
delattr(ML, 'scores')
train, val, test = ML.split(test_percentage=0.1, validation_percentage=0.1, seed=0)


def train_test(model):
    es = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
    trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=[es], checkpoint_callback=False)
    trainer.fit(model, train_dataloader=train.dataloader(512), val_dataloaders=val.dataloader(len(val)))
    return model.test_recommendations(test, train, 10)


model = ImplicitMF(interactions=train, embedding_dim=50, lr=1e-4, batch_size=512, weight_decay=0, loss='bpr')
print(train_test(model))

model = ImplicitGaussianGMF(interactions=train, embedding_dim=50, lr=1e-4, batch_size=512, weight_decay=0, loss='bpr')
print(train_test(model))

model = ImplicitGaussianGMF(interactions=train, embedding_dim=50, lr=1e-4, batch_size=512, weight_decay=0, loss='uncertain')
print(train_test(model))

'''
# The problem is that when uncertainty drops, the log pushes the loss down too much

def bpr_loss(x, log):
    a = 1 - torch.sigmoid(x)
    if log:
        return a.log()
    else:
        return a

import math
def uncertain_bpr_loss(x, unc, log):
    a = 0.5 * (1 - torch.erf((x / (unc) / math.sqrt(2))))
    if log:
        return torch.log(a)
    else:
        return a

x = torch.linspace(-5, 15, 10000)
f, ax = plt.subplots(ncols=2, sharex=True)
ax[0].plot(x, bpr_loss(x, False), label='bpr')
ax[1].plot(x, bpr_loss(x, True), label='bpr')
for unc in [1, 2, 5]:
    ax[0].plot(x, uncertain_bpr_loss(x, unc, False), label='uncertain={}'.format(unc))
    ax[1].plot(x, uncertain_bpr_loss(x, unc, True), label='uncertain={}'.format(unc))
ax[0].legend()
ax[1].legend()

x = torch.linspace(0, 0.0001, 10000)
plt.plot(x, torch.log(x))
'''