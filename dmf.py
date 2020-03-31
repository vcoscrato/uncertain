import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from surprise import Dataset

from Uncertain import DoubleSVD
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

epochs = 20
factors = 100

model = DoubleSVD(n_epochs=epochs, n_factors=factors, cv_folds=10, verbose=False).fit(data)
upreds, corr = model.test(test, build_correlation=True)
print('Model RMSE:', urmse(upreds))
print('Model RPI:', RPI(upreds))
print('Correlation reliability vs #ratings by user:', corr['r_user'])
print('Correlation reliability vs #ratings to item:', corr['r_item'])
print('Correlation reliability vs avg user rating:', corr['avg_user'])
print('Correlation reliability vs avg item rating:', corr['avg_item'])
print('Correlation reliability vs avg user similarity:', corr['sim_user'])
print('Correlation reliability vs avg item similarity:', corr['sim_item'])

err = [abs(u.est - u.r_ui) for u in upreds]
rel = [u.rel for u in upreds]
f, ax = plt.subplots()
sp = ax.scatter(err, rel, s=2)
ax.set_xlabel('Absolute prediction error', Fontsize=20)
ax.set_ylabel('Prediction reliability', Fontsize=20)
f.tight_layout()
f.savefig('Results/dmf/Rel vs Error.pdf')
print('Correlation reliability vs error:', np.corrcoef(err, rel)[0, 1])

a = build_intervals(upreds)
f, ax = plt.subplots()
ax.plot(range(1, 21), a)
ax.set_xlabel('k', Fontsize=20)
ax.set_ylabel('Interval half width', Fontsize=20)
ax.set_xticks(range(1, 21))
f.tight_layout()
f.savefig('Results/dmf/Interval width.pdf')
