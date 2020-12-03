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