import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from sklearn.model_selection import StratifiedKFold, train_test_split

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as mse

def min_max(array):
    min = array.min()
    max = array.max()
    transformed = (array - min) / (max - min)
    return transformed



def RRI_at_k(model, test_interactions, k=5):
    ranks = model.predict_rank(test_)
    ranks.data = np.less(ranks.data, k, ranks.data)
    retrieved = np.squeeze(test_interactions.getnnz(axis=1))
    hit = np.squeeze(np.array(ranks.sum(axis=1)))

def Reliability(model, test_interactions):
    ranks = model.predict_rank(test_interactions)
    sum_ranks = np.array(ranks.sum(axis=1)).flatten()
    retrieved = test_interactions.getnnz(axis=1)
    print(sum_ranks[retrieved == 0])
    user_reliability = (sum_ranks-(retrieved/2))/retrieved
    return user_reliability

ratings_list = [i.strip().split("::") for i in open('data/ml-1m/ratings.dat', 'r').readlines()]
ratings_list = [(int(r[0])-1, int(r[1])-1, int(r[2])) for r in ratings_list if int(r[2]) >= 4]
users = [r[0] for r in ratings_list]
u = np.unique(users, return_counts=True)
good_u = u[0][u[1] > 5]
ratings_list = [r for r in ratings_list if r[0] in good_u]
users = [r[0] for r in ratings_list]
train, test = train_test_split(ratings_list, shuffle=True, test_size=0.2, stratify=users, random_state=0)
train_users = [r[0] for r in train]
train_coo = build_coo_matrix(train)
test_coo = build_coo_matrix(test)


sss = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
user_reliability = np.zeros(train_coo.shape[0])
for train_index, test_index in sss.split(train_users, train_users):
    train_ = build_coo_matrix([train[i] for i in train_index])
    test_ = build_coo_matrix([train[i] for i in test_index])
    model = LightFM(loss='warp').fit(train_, epochs=50, num_threads=6)
    print("Test precision: %.3f" % precision_at_k(model, test_, k=5).mean())
    user_reliability += Reliability(model, test_)

user_reliability = min_max(1/user_reliability)



model.predict(user_ids=1, item_ids=np.arange(n_items))
print("Train precision: %.2f" % precision_at_k(model, data['train'], k=5).mean())