import numpy as np
from uncertain.metrics import rmse_score, precision_recall_score, rpi_score, precision_recall_rri_score, classification
from scipy.stats import spearmanr, pearsonr

def evaluate_base(model, test, train):

    p, r = precision_recall_score(self, test, train, np.arange(1, 11))

    return {'RMSE': rmse_score(model.predict(test.user_ids, test.item_ids), test.ratings),
            'Precision': p.mean(axis=0),
            'Recall': r.mean(axis=0)}


def evaluate_uncertainty(model, test, train):
    
    predictions = model.predict(test.user_ids, test.item_ids)
    error = np.abs(test.ratings - predictions[0])

    #quantiles, intervals = graphs_score(predictions, test.ratings)

    p, r, e = precision_recall_rri_score(model, test, train, np.arange(1, 11))
    
    return {'RMSE': rmse_score(predictions[0], test.ratings),
            'Precision': p.mean(axis=0),
            'Recall': r.mean(axis=0),
            'Correlation': (pearsonr(error, predictions[1]), spearmanr(error, predictions[1])),
            #'EpsilonReliability': np.zeros((n_replicas, 20)),
            #'RMSEReliability': np.zeros((n_replicas, 20)),
            'RPI': -rpi_score(predictions, test.ratings.cpu().detach().numpy()),
            'RRI': -np.nanmean(e, axis=0)}
            #'Classification': classification(predictions, error, test)}
