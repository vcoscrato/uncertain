import torch
import math








class ProbabilisticLoss:

    def __init__(self, log_scale=True):
        self.log_scale = log_scale

    def get_prob(self, x, sigma):
        """Override this"""
        return None

    def __call__(self, *args):
        prob = self.get_prob(*args)
        if self.log_scale:
            prob = prob.log()
        return - prob.mean()


class MaxProb(ProbabilisticLoss):

    def get_prob(self,predicted_ratings, observed_ratings):
        return predicted_ratings[range(len(predicted_ratings)), observed_ratings]


class CrossEntropy(ProbabilisticLoss):

    def get_prob(self, positive, negative):
        positive = positive.sigmoid()
        negative = 1 - negative.sigmoid()
        return


class BPR(ProbabilisticLoss):

    def __init__(self, log_scale=True):
        super().__init__(log_scale)

    def get_prob(self, positive, negative):
        x = positive - negative
        return torch.sigmoid(x)


class ABPR(ProbabilisticLoss):

    def __init__(self, log_scale=True):
        super().__init__(log_scale)

    def get_prob(self, positive, negative):
        x = positive[0] - negative[0]
        rho = positive[1] + negative[1]
        return torch.sigmoid(x / rho)


class GPR(ProbabilisticLoss):

    def __init__(self, log_scale=True):
        super().__init__(log_scale)

    def get_prob(self, positive, negative):
        x = positive[0] - negative[0]
        rho = positive[1] + negative[1]
        return 0.5 * (1 + torch.erf((x / torch.sqrt(2*rho))))