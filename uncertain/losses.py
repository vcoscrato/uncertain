import torch
import math

def mse(predicted_ratings, observed_ratings):
    return ((observed_ratings - predicted_ratings) ** 2).mean()

def gaussian(predicted_ratings, observed_ratings):
    mean, variance = predicted_ratings
    return (((observed_ratings - mean) ** 2) / variance).mean() + torch.log(variance).mean()

def max_prob(predicted_ratings, observed_ratings):
    return -predicted_ratings[range(len(-predicted_ratings)), observed_ratings].log().mean()

def cross_entropy(positive, negative):
    positive = torch.log(positive)
    negative = torch.log(1 - negative)
    return - torch.cat((positive, negative)).mean()

def bpr(positive, negative):
    return - torch.sigmoid(positive - negative).log().mean()

def adaptive_bpr(positive, negative):
    mean = positive[0] - negative[0]
    unc = positive[1] + negative[1]
    return - torch.sigmoid(mean / unc).log().mean()

def uncertain(positive, negative):
    mean = positive[0] - negative[0]
    unc = positive[1] + negative[1]
    return - (0.5 * (1 + torch.erf((mean / torch.sqrt(2*unc))))).log().mean()
