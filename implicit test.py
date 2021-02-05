import numpy as np
from uncertain.interactions import ImplicitInteractions
from uncertain.models.implicit import ImplicitFactorizationModel

coo_row = []
coo_col = []
coo_val = []

with open('/home/vcoscrato/Documents/Data/lastfm.dat', "r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        splitted = line.split('\t')
        user = int(splitted[0])
        artist = int(splitted[1])
        plays = int(splitted[2].strip())
        coo_row.append(user)
        coo_col.append(artist)
        coo_val.append(plays)

users = np.unique(coo_row, return_inverse=True)[1]
items = np.unique(coo_col, return_inverse=True)[1]
plays = np.array(coo_val)

data = ImplicitInteractions(users, items).gpu()
model = ImplicitFactorizationModel.ImplicitFactorizationModel(embedding_dim=10, batch_size=256, l2=0,
                                                              learning_rate=2, use_cuda=True,
                                                              path='Empirical study/lastfm.pth')
model.fit(data, data)
plays.sum()







import torch
import numpy as np
from pandas import factorize

from uncertain.models import FunkSVD, CPMF, OrdRec, MultiModelling
from uncertain.models import LinearUncertainty, CVUncertainty, PlugIn
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split

ML = get_movielens_dataset(variant='1M').cuda()
ML.ratings = torch.ones_like(ML.ratings)
train, test = random_train_test_split(ML, test_percentage=0.2, random_state=0)
test, val = random_train_test_split(test, test_percentage=0.5, random_state=0)

CPMF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 5,
               'batch_size': 512, 'path': 'Empirical study/cpmf.pth', 'use_cuda': True}
cpmf = CPMF(**CPMF_params)
cpmf.fit(train, val)
print(cpmf.evaluate(test, train))