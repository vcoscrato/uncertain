import numpy as np
from .algobase import ReliableAlgoBase
from copy import deepcopy
from .metrics import rmse, RPI
from surprise.accuracy import rmse as srmse


class Model_reliability(ReliableAlgoBase):
    def __init__(self, R, E, trainset):
        self.R = R
        self.E = E
        self.trainset = trainset

    def estimate(self, u, i):
        rating = self.R.estimate(u, i)
        u_ = self.E.trainset.to_inner_uid(self.R.trainset.to_raw_uid(u))
        i_ = self.E.trainset.to_inner_iid(self.R.trainset.to_raw_iid(i))
        reliability = 1 / self.E.estimate(u_, i_)
        return rating, reliability

class Heuristic_reliability(ReliableAlgoBase):
    def __init__(self, R, u_ratio):
        self.trainset = R.trainset
        self.R = R
        ur = [[a[1] for a in b] for b in R.trainset.ur.values()]
        self.nu = [len(ur[i]) for i in range(len(ur))]
        ir = [[a[1] for a in b] for b in R.trainset.ir.values()]
        self.ni = [len(ir[i]) for i in range(len(ir))]
        self.u_ratio = u_ratio

    def estimate(self, u, i):
        rating = self.R.estimate(u, i)
        reliability = self.u_ratio*self.nu[u] + self.ni[i]
        return rating, reliability


class EMF(ReliableAlgoBase):
    def __init__(self, initial_model, minimum_improvement, max_size, verbose=True):
        ReliableAlgoBase.__init__(self)
        self.initial_model = initial_model
        self.minimum_improvement = minimum_improvement
        self.max_size = max_size
        self.verbose = verbose
        self.trainset = None
        self.ensemble_size=1
        self.models = []
        self.val_metrics = {'RMSE': [], 'RPI': []}
    
    def fit(self, valset):
        self.trainset = self.initial_model.trainset
        self.models.append((self.initial_model.bu, self.initial_model.bi,
                            self.initial_model.pu, self.initial_model.qi))
        val = self.initial_model.test(valset)
        rmse0, rpi = srmse(val, verbose=False), 0
        self.val_metrics['RMSE'].append(rmse0)
        self.val_metrics['RPI'].append(rpi)
        if self.verbose:
            print('Initial model performance: RMSE = {}; RPI = {}.'.format(rmse0, rpi))
        rmse1 = 0
        while rmse1 < (rmse0*(1-self.minimum_improvement)) and self.ensemble_size < self.max_size:
            self.ensemble_size += 1
            self.initial_model.random_state += 1
            self.initial_model.sgd(self.trainset)
            self.models.append((self.initial_model.bu, self.initial_model.bi,
                                self.initial_model.pu, self.initial_model.qi))
            val = self.test(valset)
            self.val_metrics['RMSE'].append(rmse(val))
            self.val_metrics['RPI'].append(RPI(val))
            rmse1 = self.val_metrics['RMSE'][-1]
            rmse0 = self.val_metrics['RMSE'][-2]
            if self.verbose:
                print('{}-model ensemble performance: RMSE = {}; RPI = {}.'.format(self.ensemble_size,
                                                                                   rmse1, self.val_metrics['RPI'][-1]))
        if self.verbose:
            print('Training finished with ensemble size {}.'.format(self.ensemble_size))

    def estimate(self, u, i):
        ests = np.zeros((self.ensemble_size))
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        for e in range(self.ensemble_size):
                ests[e] += self.trainset.global_mean
                if known_user:
                    ests[e] += self.models[e][0][u]
                if known_item:
                    ests[e] += self.models[e][1][i]
                if known_user and known_item:
                    ests[e] += np.dot(self.models[e][3][i], self.models[e][2][u])
        est = ests.mean()
        rel = 1/ests.std()
        return est, rel


class RMF(ReliableAlgoBase):
    def __init__(self, resample_size, initial_model, minimum_improvement, max_size, verbose=True):
        ReliableAlgoBase.__init__(self)
        self.initial_model = initial_model
        self.minimum_improvement = minimum_improvement
        self.max_size = max_size
        self.resample_size=resample_size
        self.verbose = verbose
        self.trainset = None
        self.n_models = 0
        self.models = []
        self.val_metrics = {'RMSE': None, 'RPI': []}

    def fit(self, valset):
        self.trainset = self.initial_model.trainset
        og_ur = deepcopy(self.trainset.ur)
        og_ir = deepcopy(self.trainset.ir)
        og_par = deepcopy((self.initial_model.bu, self.initial_model.bi, self.initial_model.pu, self.initial_model.qi))
        val = self.initial_model.test(valset)
        rmse0, rpi0 = srmse(val, verbose=False), 0
        self.val_metrics['RMSE'] = rmse0
        self.val_metrics['RPI'].append(rpi0)
        if self.verbose:
            print('Initial model performance: RMSE = {}; RPI = {}.'.format(rmse0, rpi0))
        rpi1 = np.inf
        while (rpi1 >= (rpi0*self.minimum_improvement)) and self.n_models < self.max_size:
            self.n_models += 1
            for u in self.trainset.ur:
                ur = self.trainset.ur[u]
                np.random.seed(self.n_models)
                sample = np.random.binomial(1, self.resample_size, len(ur))
                del_i = [ur[j][0] for j in range(len(ur)) if sample[j] == False]
                for i in del_i:
                    del self.trainset.ur[u][[a[0] for a in self.trainset.ur[u]].index(i)]
                    del self.trainset.ir[i][[a[0] for a in self.trainset.ir[i]].index(u)]
            self.initial_model.sgd(self.trainset)
            self.models.append((self.initial_model.bu, self.initial_model.bi,
                                self.initial_model.pu, self.initial_model.qi))
            if self.n_models > 1:
                val = self.test(valset)
                self.val_metrics['RPI'].append(RPI(val))
                epi1 = self.val_metrics['RPI'][-1]
                epi0 = self.val_metrics['RPI'][-2]
                if self.verbose:
                    print('{}-model uncertainty performance: RPI = {}.'.format(self.n_models,
                                                                               self.val_metrics['RPI'][-1]))
            self.trainset.ur = deepcopy(og_ur)
            self.trainset.ir = deepcopy(og_ir)
        self.initial_model.bu, self.initial_model.bi, self.initial_model.pu, self.initial_model.qi = og_par
        if self.verbose:
            print('Training finished with {} models.'.format(self.n_models))

    def estimate(self, u, i):
        est = self.initial_model.estimate(u, i)
        ests = np.zeros((self.n_models))
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        for e in range(self.n_models):
                ests[e] += self.trainset.global_mean
                if known_user:
                    ests[e] += self.models[e][0][u]
                if known_item:
                    ests[e] += self.models[e][1][i]
                if known_user and known_item:
                    ests[e] += np.dot(self.models[e][3][i], self.models[e][2][u])
        rel = 1/ests.std()
        return est, rel
