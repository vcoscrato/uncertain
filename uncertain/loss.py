import math
import torch


'''
Explicit feedback losses
'''
def mse(predicted, observed):
    return ((observed - predicted) ** 2).sum()


def gaussian(predicted, observed):
    mean, variance = predicted
    return (((observed - mean) ** 2) / variance).sum() + torch.log(variance).sum()


'''
Implicit feedback losses
'''
def normal_cdf(mean, var, value=0):
    '''
    Returns the P(N(mean, var) <= value)
    '''
    return 0.5 * (1 + torch.erf((value - mean) * var.sqrt().reciprocal() / math.sqrt(2)))


def normal_logpdf(mean, var, value):
    '''
    Returns the log P(N(mean, var) = value) (log Prob density function)
    '''
    return -((value - mean) ** 2) / (2 * var) - var.log() / 2 #Constant: -math.log(math.sqrt(2 * math.pi))


def BCE(positive_scores, negative_scores):
    '''
    Binary cross entropy loss.
    Returns the average likelihood and the NLL.
    '''
    probs = torch.cat((positive_scores.sigmoid(), 1 - negative_scores.sigmoid()))
    return probs.mean(), - probs.log().sum()


def BPR(positive_scores, negative_scores):
    '''
    Bayesian Personalized Ranking opt.
    Returns the average likelihood and the NLL.
    '''
    probs = torch.sigmoid(positive_scores - negative_scores)
    return probs.mean(), - probs.log().sum()



def GBR(positive_scores, negative_scores):
    '''
    Gaussian Binary Ranking. P(u likes i) = P(Y_ui > 0).
    Returns the average likelihood and the NLL.
    '''
    pos_mean, pos_var = positive_scores
    neg_mean, neg_var = negative_scores

    pos_probs = 1 - normal_cdf(pos_mean, pos_var)
    neg_probs = normal_cdf(pos_mean, pos_var)

    probs = torch.cat((pos_probs, 1 - neg_probs))
    return probs.mean(), - probs.log().sum()


def GPR(positive_scores, negative_scores):
    '''
    Gaussian Pairwise Ranking. P(i >_u j) = P(Y_ui - Y_uj > 0)
    Returns the average likelihood and the NLL.
    '''
    pos_mean, pos_var = positive_scores
    neg_mean, neg_var = negative_scores
    
    mean_diff = pos_mean - neg_mean
    var_sum = pos_var + neg_var

    probs = 1 - normal_cdf(mean_diff, var_sum)
    return probs.mean(), - probs.log().sum()


def AUR(positive_scores, negative_scores):
    '''
    Loss function from the Aleatoric Uncertainty Recommender models.
    In short, this is a mean squared error loss used on binary data.
    Returns the average likelihood and the NLL.
    '''
    pos_mean, pos_var = positive_scores
    neg_mean, neg_var = negative_scores
    
    pos_logpdf = normal_logpdf(pos_mean, pos_var, 1)
    neg_logpdf = normal_logpdf(pos_mean, pos_var, 0)
    
    log_likelihoods = torch.cat((pos_logpdf, neg_logpdf))
    return torch.exp(log_likelihoods).mean(), -log_likelihoods.sum()
