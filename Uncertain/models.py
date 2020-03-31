import numpy as np
from .algobase import ReliableAlgoBase
from surprise import SVD
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold

class SVDAverageEnsemble(ReliableAlgoBase):
    def __init__(self, n_factors, n_epochs, n_models=20, initial_random_state=0, verbose=True):
        ReliableAlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.n_models = n_models
        self.initial_random_state = initial_random_state
        self.verbose = verbose
    
    def fit(self, data=None, models=None):
        if models is not None:
            self.models = models
            self.trainset = models[0].trainset
        else:
            self.trainset = data.build_full_trainset()
            self.ensemblize(self.trainset)
        return self
        
    def ensemblize(self, trainset):
        self.models = []
        for id, i in enumerate(range(self.initial_random_state, self.initial_random_state+self.n_models)):
            if self.verbose:
                print('Fitting: Model', id+1)
            self.models.append(SVD(n_epochs=self.n_epochs, n_factors=self.n_factors, random_state=i).fit(self.trainset))
            
    def estimate(self, u, i):
        preds = [model.estimate(u, i) for model in self.models]
        pred = np.mean(preds)
        rel = 1/np.std(preds)
        return pred, rel


class SamplingAverageEnsemble(ReliableAlgoBase):
    def __init__(self, n_factors, n_epochs, n_models, resample_size=0.9, initial_random_state=0, verbose=True):
        ReliableAlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.n_models = n_models
        self.resample_size=resample_size
        self.initial_random_state = initial_random_state
        self.verbose = verbose

    def fit(self, data, fixed_seed_models=True):
        self.trainset = data.build_full_trainset()
        og_ur = deepcopy(self.trainset.ur)
        og_ir = deepcopy(self.trainset.ir)
        self.models = []
        for ensemble in range(self.n_models):
            for u in self.trainset.ur:
                ur = self.trainset.ur[u]
                np.random.seed(self.initial_random_state + ensemble)
                sample = np.random.binomial(1, self.resample_size, len(ur))
                del_i = [ur[j][0] for j in range(len(ur)) if sample[j] == False]
                for i in del_i:
                    del self.trainset.ur[u][[a[0] for a in self.trainset.ur[u]].index(i)]
                    del self.trainset.ir[i][[a[0] for a in self.trainset.ir[i]].index(u)]

            algo = SVD(n_epochs=self.n_epochs, n_factors=self.n_factors, random_state=self.initial_random_state)
            if not fixed_seed_models:
                algo.random_state += ensemble
            if self.verbose:
                print('Fitting: Model', ensemble+1)
            self.models.append(algo.fit(self.trainset))
            self.trainset.ur = deepcopy(og_ur)
            self.trainset.ir = deepcopy(og_ir)
        return self

    def estimate(self, u, i):
        preds = [model.estimate(u, i) for model in self.models]
        pred = np.mean(preds)
        rel = 1/np.std(preds)
        return pred, rel


class SamplingSVD(ReliableAlgoBase):
    def __init__(self, n_factors, n_epochs, n_models, resample_size=0.9, initial_random_state=0, verbose=True):
        ReliableAlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.n_models = n_models
        self.resample_size=resample_size
        self.initial_random_state = initial_random_state
        self.verbose = verbose

    def fit(self, data, fixed_seed_models=True):
        self.trainset = data.build_full_trainset()
        self.model = SVD(n_epochs=self.n_epochs, n_factors=self.n_factors, random_state=self.initial_random_state).fit(self.trainset)
        og_ur = deepcopy(self.trainset.ur)
        og_ir = deepcopy(self.trainset.ir)
        self.models = []
        for ensemble in range(self.n_models):
            for u in self.trainset.ur:
                ur = self.trainset.ur[u]
                np.random.seed(self.initial_random_state + ensemble)
                sample = np.random.binomial(1, self.resample_size, len(ur))
                del_i = [ur[j][0] for j in range(len(ur)) if sample[j] == False]
                for i in del_i:
                    del self.trainset.ur[u][[a[0] for a in self.trainset.ur[u]].index(i)]
                    del self.trainset.ir[i][[a[0] for a in self.trainset.ir[i]].index(u)]

            algo = SVD(n_epochs=self.n_epochs, n_factors=self.n_factors, random_state=self.initial_random_state)
            if not fixed_seed_models:
                algo.random_state += ensemble
            if self.verbose:
                print('Fitting: Model', ensemble+1)
            self.models.append(algo.fit(self.trainset))
            self.trainset.ur = deepcopy(og_ur)
            self.trainset.ir = deepcopy(og_ir)
        return self

    def estimate(self, u, i):
        pred = self.model.estimate(u, i)
        preds = [model.estimate(u, i) for model in self.models]
        rel = 1/np.std(preds)
        return pred, rel

class DoubleSVD(ReliableAlgoBase):
    def __init__(self, n_factors, n_epochs, cv_folds=4, random_state=0, verbose=True):
        ReliableAlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

    def build_cv_preds(self, data, random_state=0):
        data_ = deepcopy(data)
        users = [a[0] for a in data.raw_ratings]
        np.random.seed(0)
        np.random.shuffle(data.raw_ratings)
        splitter = StratifiedKFold(n_splits=self.cv_folds)
        cv_preds = []
        for train_index, test_index in splitter.split(data.raw_ratings, users):
            data_.raw_ratings = [data.raw_ratings[i] for i in train_index]
            train = data.build_full_trainset()
            model = SVD(n_epochs=self.n_epochs, n_factors=self.n_factors, random_state=random_state).fit(train)
            for t in [data.raw_ratings[i] for i in test_index]:
                cv_preds.append((t[0], t[1], np.abs(t[2] - model.estimate(t[0], t[1])), t[3]))
        return cv_preds

    def fit(self, data):
        self.trainset = data.build_full_trainset()
        if self.verbose:
            print('Obtaining cross-validated errors')
        cv_preds = self.build_cv_preds(data, self.random_state)
        self.cv_preds = cv_preds
        if self.verbose:
            print('Fitting rating estimator')
        self.SVDest = SVD(n_epochs=self.n_epochs, n_factors=self.n_factors, random_state=self.random_state).fit(self.trainset)
        data_ = deepcopy(data)
        data_.raw_ratings = cv_preds
        train_rel = data_.build_full_trainset()
        train_rel.rating_scale = (0, np.inf)
        if self.verbose:
            print('Fitting reliability estimator')
        self.SVDrel = SVD(n_epochs=self.n_epochs, n_factors=int(self.n_factors/2), random_state=self.random_state).fit(train_rel)
        return self

    def estimate(self, u, i):
        est = self.SVDest.estimate(u, i)
        u_ = self.SVDrel.trainset.to_inner_uid(self.SVDest.trainset.to_raw_uid(u))
        i_ = self.SVDrel.trainset.to_inner_iid(self.SVDest.trainset.to_raw_iid(i))
        rel = 1/self.SVDrel.estimate(u_, i_)
        return est, rel