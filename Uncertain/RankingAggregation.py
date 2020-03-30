import numpy as np
from .metrics import kendallW

class RankingAggregation(object):
    
    def __init__(self, models, data):
        self.models = models
        self.data = data
        self.ensemble = len(models)
        self.n_items = models[0].item_norms.shape[0]
        
    def recommend(self, user,  N=10):
        
        '''
        Obtain the recommendation for the first model to get the number of unrated itens, 
        iid is an auxiliar indexer to align the columns of the recommendation matrix
        '''
        rec = [a[0] for a in self.models[0].recommend(user, self.data, self.n_items)]
        n_unrated_items = len(rec)
        iid = np.sort(rec)
        
        '''
        Initialize the recommendation matrix and the aligned recommendations 
        (in the second one the i-th row correspond to the rank given to the i-th user for every model)
        '''
        recommendations = np.empty((n_unrated_items, self.ensemble))
        align_recommendations = np.empty((n_unrated_items, self.ensemble))
        
        
        '''
        Assert values
        '''
        recommendations[:, 0] = rec
        align_recommendations[:, 0] = np.argsort(rec)
        for i in range(1, len(self.models)):
            recommendations[:, i] = [a[0] for a in self.models[i].recommend(user, self.data, self.n_items)]
            align_recommendations[:, i] = np.argsort(recommendations[:, i])
            
        '''
        Calculate the aggregated ranking
        '''
        avg = align_recommendations.mean(axis=1)+1
        sd = align_recommendations.std(axis=1)
        indexer = np.argsort(avg)[:N]
        iid = iid[indexer]
        avg = avg[indexer]
        sd = sd[indexer]
        
        '''
        Obtain kendall's w on TOP N
        '''
        avg_w = np.argsort(align_recommendations[indexer], axis=0).mean(axis=1)
        w = 12*np.square(avg_w*self.ensemble-(avg_w*self.ensemble).mean()).sum()/((N**3-N)*self.ensemble**2)
        return UncertainRanking(user, iid, avg, sd, w)