import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas import DataFrame as df

#Uses the basic SVD implementation from Surprise, the dataset handler and the RMSE calculator
from surprise import SVD
from surprise import Dataset
from surprise.accuracy import rmse

#Uncertain is a package built on top of surprise to deal with recommendations with reliability level
from Uncertain import SVDAverageEnsemble, SamplingAverageEnsemble, SamplingSVD, DoubleSVD
from Uncertain.metrics import rmse as urmse, RPI
from Uncertain.metrics import misscalibration
from Uncertain.metrics import build_intervals

#Load the dataset
data = Dataset.load_builtin('ml-1m')
users = [a[0] for a in data.raw_ratings]
train, test = train_test_split(data.raw_ratings, shuffle=True, random_state=0, test_size=0.2, stratify=users)
data.raw_ratings = train
test = [test[:3] for test in test]
# Delete a few test itens that are not in the training set
train_itens = np.unique([train[1] for train in train])
for i in np.unique([test[1] for test in test]):
    if i not in train_itens:
        for id in reversed(np.argwhere(np.array([test[1] for test in test]) == i)):
            del test[id[0]]

#Define the models parameters
n_epochs = 20
n_factors = 100

#Build the models
modelDoubleSVD = DoubleSVD(n_epochs=n_epochs, n_factors=n_factors,
                           cv_folds=4, random_state=0).fit(data)
modelSVDAverageEnsemble = SVDAverageEnsemble(n_epochs=n_epochs, n_factors=n_factors,
                                             n_models=20, initial_random_state=0).fit(data)
modelSamplingAverage = SamplingAverageEnsemble(n_epochs=n_epochs, n_factors=n_factors,
                                               n_models=20, resample_size=0.9,
                                               initial_random_state=0).fit(data)
modelSamplingSVD = SamplingSVD(n_epochs=n_epochs, n_factors=n_factors,
                               n_models=20, resample_size=0.9,
                               initial_random_state=0).fit(data)

pred = {}
pred['DoubleSVD'] = modelDoubleSVD.test(test)
pred['SVDAverageEnsemble'] = modelSVDAverageEnsemble.test(test)
pred['SamplingAverage'] = modelSamplingAverage.test(test)
pred['SamplingSVD'] = modelSamplingSVD.test(test)

eval = {}
eval['DoubleSVD'] = [urmse(pred['DoubleSVD']), RPI(pred['DoubleSVD'])]
eval['SVDAverageEnsemble'] = [urmse(pred['SVDAverageEnsemble']), RPI(pred['SVDAverageEnsemble'])]
eval['SamplingAverage'] = [urmse(pred['SamplingAverage']), RPI(pred['SamplingAverage'])]
eval['SamplingSVD'] = [urmse(pred['SamplingSVD']), RPI(pred['SamplingSVD'])]

eval = df(eval, index=['RMSE', 'RPI'])
print(eval)

intervals = list(map(build_intervals, pred.values()))
aes = ['g-', 'r-', 'b-', 'y-']
f, ax = plt.subplots()
for id, key in enumerate(pred.keys()):
    ax.plot(range(1, 21), intervals[id], aes[id], label=key)
ax.set_xlabel('k', Fontsize=20)
ax.set_ylabel('Interval half width', Fontsize=20)
ax.set_xticks(range(1, 21))
plt.legend()
f.tight_layout()
f.savefig('Results/Interval width.pdf')