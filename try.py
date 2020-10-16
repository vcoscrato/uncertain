import torch
import numpy as np
from pandas import factorize
from Utils.utils import dataset_loader

from uncertain.models.OrdRec import OrdRec
from uncertain.models.CPMF import CPMF
from uncertain.models.ExplicitFactorizationModel import ExplicitFactorizationModel

train, test = dataset_loader('10M', 0)
train.gpu()
test.gpu()
model = ExplicitFactorizationModel(embedding_dim=50, n_iter=100, batch_size=100000,
                                   learning_rate=0.01, l2=1e-6, use_cuda=True)
model.fit(train)







factor = factorize(train.ratings, sort=True)
rating_labels = torch.from_numpy(factor[1].astype(np.float64))
train.ratings = torch.from_numpy(factor[0])
test.ratings = torch.from_numpy(factorize(test.ratings, sort=True)[0])


model = OrdRec(rating_labels, embedding_dim=50, n_iter=100, batch_size=100000, learning_rate=0.01, l2=1e-6, use_cuda=True)
model.fit(train)
model._predict(test.user_ids, test.item_ids)


