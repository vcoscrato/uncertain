import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from surprise import SVD
from surprise import Dataset
from surprise.accuracy import rmse

from Uncertain import SVDAverageEnsemble
from Uncertain.metrics import rmse as urmse
from Uncertain.metrics import build_intervals
from Uncertain.metrics import RPI

data = Dataset.load_builtin('ml-1m')
users = [a[0] for a in data.raw_ratings]
train, test = train_test_split(data.raw_ratings, shuffle=True, random_state=0, test_size=0.1, stratify=users)
data.raw_ratings = train
test = [test[:3] for test in test]
# Delete a few test itens that are not in the training set
train_itens = np.unique([train[1] for train in train])
for i in np.unique([test[1] for test in test]):
    if i not in train_itens:
        for id in reversed(np.argwhere(np.array([test[1] for test in test]) == i)):
            del test[id[0]]
train = data.build_full_trainset()

ensemble = 30
epochs = 2
factors = 2

models = []
RMSE = np.empty((ensemble))
preds = np.empty((ensemble, len(test)))
for i in range(ensemble):
    models.append(SVD(random_state=i, n_epochs=epochs, n_factors=factors).fit(data.build_full_trainset()))
    preds_ = models[i].test(test)
    RMSE[i] = rmse(preds_, verbose=False)
    preds[i] = [i.est for i in preds_]
print('Min RMSE =', round(np.min(RMSE), 4), '; Max RMSE =', round(np.max(RMSE), 4))
eRMSE = np.empty((ensemble))
eRPI = np.empty((ensemble-1))
for i in range(ensemble):
    model = SVDAverageEnsemble(n_epochs=epochs, n_factors=factors).fit(models=models[:i+1])
    upreds = model.test(test)
    eRMSE[i] = urmse(upreds, verbose=False)
    if i > 0:
        eRPI[i-1] = RPI(upreds)

f, ax = plt.subplots(figsize=(15,5))
ax.plot(range(1, ensemble+1), eRMSE, color='blue')
ax.set_xticks(range(1, ensemble+1))
ax.set_xlabel('Ensemble size', Fontsize=20)
ax.set_ylabel('RMSE', Fontsize=20, color='blue')
ax.tick_params(axis='y', labelcolor='blue')
ax_ = ax.twinx()
ax_.plot(range(2, ensemble+1), eRPI, color='red')
ax_.set_ylabel('RPI', Fontsize=20, color='red')
ax_.tick_params(axis='y', labelcolor='red')
f.tight_layout()
f.savefig('Results/Seed/Ensemble size.pdf')
print('Full ensemble RMSE:', eRMSE[-1])
print('Full ensemble RPI:', eRPI[-1])

uids = np.unique([i.uid for i in upreds])
iids = np.unique([i.iid for i in upreds])
nu = [len(train.ir[train.to_inner_uid(uid)]) for uid in uids]
ni = np.empty((len(iids)))
for i, iid in enumerate(iids):
    try:
        ni[i] = len(train.ir[train.to_inner_iid(iid)])
    except:
        ni[i] = 0
pred_n = np.empty((len(upreds), 3))
for i in range(len(pred_n)):
    pred_n[i] = [upreds[i].rel, nu[np.where(upreds[i].uid == uids)[0][0]], ni[np.where(upreds[i].iid == iids)[0][0]]]
print('Correlation reliability vs #ratings by user:', np.corrcoef(pred_n.T)[0, 1])
print('Correlation reliability vs #ratings to item:', np.corrcoef(pred_n.T)[0, 2])

avgu = [np.mean([t[1] for t in train.ur[train.to_inner_uid(uid)]]) for uid in uids]
avgi = np.empty((len(iids)))
for i, iid in enumerate(iids):
    try:
        avgi[i] = np.mean([t[1] for t in train.ir[train.to_inner_iid(iid)]])
    except:
        avgi[i] = train.global_mean
pred_n = np.empty((len(upreds), 3))
for i in range(len(pred_n)):
    pred_n[i] = [upreds[i].rel, avgu[np.where(upreds[i].uid == uids)[0][0]], avgi[np.where(upreds[i].iid == iids)[0][0]]]
print('Correlation reliability vs avg user rating:', np.corrcoef(pred_n.T)[0, 1])
print('Correlation reliability vs avg item rating:', np.corrcoef(pred_n.T)[0, 2])

simu = models[0].compute_similarities().mean(axis=0)
models[1].sim_options['user_based'] = False
simi = models[1].compute_similarities().mean(axis=0)
pred_n = np.empty((len(upreds), 3))
invalid = []
for i in range(len(pred_n)):
    try:
        pred_n[i] = [upreds[i].rel, simu[train.to_inner_uid(upreds[i].uid)], simi[train.to_inner_iid(upreds[i].iid)]]
    except:
        invalid.append(i)
print('Correlation reliability vs avg user similarity:', np.corrcoef(pred_n.T)[0, 1])
print('Correlation reliability vs avg item similarity:', np.corrcoef(pred_n.T)[0, 2])

err = [abs(u.est - u.r_ui) for u in upreds]
rel = [u.rel for u in upreds]
f, ax = plt.subplots()
sp = ax.scatter(err, rel, markersize=2)
ax.set_xlabel('Absolute prediction error', Fontsize=20)
ax.set_ylabel('Prediction reliability', Fontsize=20)
f.tight_layout()
f.savefig('Results/Seed/Rel vs Error.pdf')
print('Correlation error and reliability:', np.corrcoef(err, rel)[0, 1])

a = build_intervals(upreds)
f, ax = plt.subplots()
ax.plot(range(1, ensemble+1), a)
ax.set_xlabel('k', Fontsize=20)
ax.set_ylabel('Interval half width', Fontsize=20)
ax.set_xticks(range(1, ensemble+1))
f.tight_layout()
f.savefig('Results/Seed/Interval width.pdf')