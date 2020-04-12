import numpy as np
from .algobase import ReliableAlgoBase
from surprise import SVD
from surprise.model_selection.search import GridSearchCV
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from .metrics import rmse, RPI
from surprise.accuracy import rmse as srmse
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.baseline_only import BaselineOnly

class DMF(ReliableAlgoBase):
    def __init__(self, initial_model, type, param_rel, cv_folds, random_state=0, verbose=True):
        ReliableAlgoBase.__init__(self)
        if type == 'absolute':
            self._type = np.abs
        elif type == 'squared':
            self._type = np.square
        else:
            Exception('type must be one of "absolute" or "squared".')
        self.SVDest = initial_model
        self.trainset = self.SVDest.trainset
        self.param_rel = param_rel
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        self.cv_results = None

    def build_cv_preds(self, data, random_state=0):
        data_ = deepcopy(data)
        users = [a[0] for a in data.raw_ratings]
        np.random.seed(0)
        np.random.shuffle(data.raw_ratings)
        splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=False)
        cv_preds = []
        for train_index, test_index in splitter.split(data.raw_ratings, users):
            data_.raw_ratings = [data.raw_ratings[i] for i in train_index]
            test = [data.raw_ratings[i] for i in test_index]
            train = data_.build_full_trainset()
            model = SVD(n_factors=self.SVDest.n_factors, n_epochs=self.SVDest.n_epochs,
                        lr_all=self.SVDest.lr_bi, reg_all=self.SVDest.reg_bi,
                        init_std_dev=self.SVDest.init_std_dev, random_state=self.SVDest.random_state).fit(train)
            for t in test:
                pred = model.predict(t[0], t[1]).est
                cv_preds.append((t[0], t[1], self._type(t[2] - pred), t[3]))
        return cv_preds

    def fit(self, data):
        if self.verbose:
            print('Obtaining cross-validated errors')
        self.cv_preds = self.build_cv_preds(data, self.random_state)
        data_ = deepcopy(data)
        data_.raw_ratings = self.cv_preds
        data_.reader.rating_scale = (0, np.inf)
        if self.verbose:
            print('Fitting reliability estimator')
        #cv = GridSearchCV(SVD, param_grid=self.param_rel, measures=['rmse'], cv=self.cv_folds, n_jobs=-2, refit=True)
        self.SVDrel = BaselineOnly(bsl_options=self.param_rel)
        self.SVDrel.fit(data_.build_full_trainset())
        #if self.verbose:
            #print('Reliability estimators fitted with best parameters: {} and best score: {}.'.format(cv.best_params,
                                                                                                      #cv.best_score))
        #self.cv_results = cv.cv_results
        #self.SVDrel = cv.best_estimator['rmse']

    def estimate(self, u, i):
        est = self.SVDest.estimate(u, i)
        u_ = self.SVDrel.trainset.to_inner_uid(self.SVDest.trainset.to_raw_uid(u))
        i_ = self.SVDrel.trainset.to_inner_iid(self.SVDest.trainset.to_raw_iid(i))
        rel = 1/self.SVDrel.estimate(u_, i_)
        return est, rel


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
