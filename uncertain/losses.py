import torch

def mse_loss(predicted_ratings, observed_ratings):
    return ((observed_ratings - predicted_ratings) ** 2).mean()

def gaussian_loss(predicted_ratings, observed_ratings):
    mean, variance = predicted_ratings
    return (((observed_ratings - mean) ** 2) / variance).mean() + torch.log(variance).mean()

def max_prob_loss(predicted_ratings, observed_ratings):
    return -predicted_ratings[range(len(-predicted_ratings)), observed_ratings].log().mean()

def cross_entropy_loss(positive, negative):
    positive = torch.log(positive)
    negative = torch.log(1 - negative)
    return - torch.cat((positive, negative)).mean()

def bpr_loss(positive, negative):
    return - (positive - negative).sigmoid().log().sum()

def uncertain_bpr_loss(positive, negative, weight=1):
    relevance_pos = positive[0]
    relevance_neg = negative[0]
    unc = positive[1] + negative[1]
    return torch.divide(relevance_pos - relevance_neg, unc).mean() - weight * torch.log(unc).mean()