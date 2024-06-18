from uncertain.core import VanillaRecommender, UncertainRecommender
from uncertain.implicit.base import Implicit

import numpy as np
import torch
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.optim as optim

from pyro.infer import SVI, TraceGraph_ELBO, TraceMeanField_ELBO
from tqdm.notebook import tqdm


def train(model, data, n_steps=10, val_every_n_epochs=None):

    pyro.enable_validation(True)
    if not hasattr(model, 'losses'):
        model.epoch_loss = []

    # lr_decay = 0.1 ** (1 / n_steps)
    # model.optimizer = optim.ClippedAdam({'lr': lr, 'betas': (0.95, 0.999), 'lrd': lr_decay})
    model.optimizer = optim.Adam({'lr': model.lr})
    svi = SVI(model.model, model.guide, model.optimizer, loss=TraceGraph_ELBO())

    if val_every_n_epochs is not None:
        non_improving = 0
        best_map = 0
    
    for epoch in (pbar := tqdm(range(n_steps), desc='Training progress')):
        epoch_loss, n_bat = 0, 0
        
        for databat in (pbar2 := tqdm(data.train_dataloader(), leave=False, desc='Epoch progress')):
            n_bat += 1
            n = len(databat)
            user = databat[:, 0]
            item = databat[:, 1]
            
            # Sample negatives
            neg_user = np.random.randint(low=0, high=model.n_user, size=n*model.n_negatives)
            neg_item = np.random.randint(low=0, high=model.n_item, size=n*model.n_negatives)
            users = np.concatenate((user, neg_user))
            items = np.concatenate((item, neg_item))
            Y = np.concatenate((np.ones_like(user), np.zeros_like(neg_user)))
            
            # Update
            epoch_loss += svi.step(users, items, Y)
                            
        model.epoch_loss.append(epoch_loss / n_bat)
        if val_every_n_epochs is None:
            pbar.set_postfix({'ELBO loss': model.epoch_loss[-1]})

        if val_every_n_epochs is not None:
            if epoch // val_every_n_epochs == epoch / val_every_n_epochs:
                # fix variational parameters (to run validation)
                model.fix_learned_embeddings()

            MAP_user = torch.zeros(data.n_user)

            for batch in (pbar3 := tqdm(data.val_dataloader(), leave=False, desc='Validation progress')):
                user, rated, targets = batch
                if hasattr(model, 'uncertain_transform'):
                    rec, _, unc = model.uncertain_rank(user, ignored_item_ids=rated[0], top_n=5)
                else:
                    rec = model.rank(user, ignored_item_ids=rated[0], top_n=5)
                hits = torch.isin(rec, targets[0], assume_unique=True)
                n_hits = hits.cumsum(0)
                if n_hits[-1] > 0:
                    precision = n_hits / torch.arange(1, 6, device=n_hits.device)
                    MAP_user[user.item()] = torch.sum(precision * hits) / n_hits[-1]

            epoch_map = MAP_user.mean().item()
            pbar.set_postfix({'Epoch MAP': epoch_map})
            if epoch_map > best_map:
                best_map = epoch_map
                non_improving = 0
            else:
                non_improving = non_improving + 1

            if non_improving > patience:
                print('Early stopping triggered in epoch {}'.format(epoch))
                return best_map


def visualize(model, data, trace=True):

    print('Graphical model:')
    display(pyro.render_model(model.model, model_args=(data.train[:, 0], 
                                                   data.train[:, 1], 
                                                   np.ones_like(data.train[:, 0])), 
                              render_params=True, render_distributions=True))

    print('Variational distribution graph:')
    display(pyro.render_model(model.guide, model_args=(data.train[:, 0], 
                                                       data.train[:, 1], 
                                                       np.ones_like(data.train[:, 0])), 
                              render_params=True, render_distributions=True))

    if trace:
        print('Model Trace:')
        trace = poutine.trace(model.model).get_trace(data.train[:, 0], 
                                                     data.train[:, 1], 
                                                     np.ones_like(data.train[:, 0]))
        print(trace.format_shapes())



# with pyro.plate('Relevance', len(user)):
#     mean = (p_u[user] * q_i[item]).sum(1)
#     var = self.softplus(gamma_u[user] * gamma_i[item])
#     r_ui = pyro.sample('r_ui', dist.Normal(mean, var.sqrt()))

# # Debug
# f, ax = plt.subplots(ncols=2)
# print('Variance:', var.detach().numpy().min(), var.detach().numpy().max(), var.detach().numpy().mean())
# ax[0].hist(var.detach().numpy(), bins=100, color='green', label='variance')
# ax[1].hist(r_ui.detach().numpy(), bins=100, color='green', label='relevance')

class HBR(Implicit, UncertainRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr=0.0001, n_negatives=10, tau_gamma=1e-2, tau_u=1e-2, tau_i=1e-2):

        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.n_negatives = n_negatives
        self.softplus = torch.nn.Softplus()
        self.sigmoid = lambda x: 1 / (1 + torch.exp(-x))
        self.sig_trans = dist.transforms.SigmoidTransform()
        
        self.predictive = False

        # Embeddings precision (alpha_u, alpha_i) in the paper
        # Higher values -> More precision -> Less variance -> More reguralization
        self.τ_u = torch.tensor(tau_u)
        self.τ_i = torch.tensor(tau_i)

        # Penalty factor (precision) for the relevance variances
        self.τ_γ = torch.tensor(tau_gamma)
    
    def model(self, user, item, Y=None):
        if Y is not None:
            Y = torch.tensor(Y).float()
        
        p_u = pyro.sample('p_u', dist.Normal(torch.zeros(self.n_user, self.embedding_dim), 1/self.τ_u.sqrt()).to_event(2))
        q_i = pyro.sample('q_i', dist.Normal(torch.zeros(self.n_item, self.embedding_dim), 1/self.τ_i.sqrt()).to_event(2))

        with pyro.plate('Relevance', len(user)):
            mean = (p_u[user] * q_i[item]).sum(1)
            γ_ui = pyro.sample('γ_ui', dist.Normal(0, 1/self.τ_γ.sqrt()))
            sqrt_var_ui = torch.sqrt(1 / torch.exp(γ_ui))
            prob_ui = pyro.sample('prob_ui', dist.TransformedDistribution(dist.Normal(mean, sqrt_var_ui), [self.sig_trans]))
            Y_ui = pyro.sample('Y_ui', dist.Bernoulli(prob_ui), obs=Y)

    def guide(self, user, item, Y=None):
        
        # Init the embeddings according to N(0, 1/d)
        µ_u = pyro.param('µ_u', torch.randn(self.n_user, self.embedding_dim) / self.embedding_dim)
        µ_i = pyro.param('µ_i', torch.randn(self.n_item, self.embedding_dim) / self.embedding_dim)

        # CPMF precision params
        γ_u = pyro.param('γ_u', torch.randn(self.n_user))
        γ_i = pyro.param('γ_i', torch.randn(self.n_item))

        p_u = pyro.sample('p_u', dist.Delta(µ_u).to_event(2))
        q_i = pyro.sample('q_i', dist.Delta(µ_i).to_event(2))

        if self.predictive:
            return None
        
        with pyro.plate('Relevance', len(user)):
            mean = (p_u[user] * q_i[item]).sum(1)
            γ_ui = pyro.sample('γ_ui', dist.Delta(γ_u[user] * γ_i[item]))
            sqrt_var_ui = torch.sqrt(1 / torch.exp(γ_ui))
            prob_ui = pyro.sample('prob_ui', dist.TransformedDistribution(dist.Normal(mean, sqrt_var_ui), [self.sig_trans]))

    def fix_learned_embeddings(self):
        self.user_embeddings = pyro.param('µ_u').detach()
        self.item_embeddings = pyro.param('µ_i').detach()
        self.user_var = pyro.param('γ_u').detach()
        self.item_var = pyro.param('γ_i').detach()

    def get_user_embeddings(self, user_ids):
        return self.user_embeddings[user_ids], self.user_var[user_ids]
    
    def get_item_embeddings(self, item_ids=None):
        if item_ids is not None:
            return self.item_embeddings[item_ids], self.item_var[item_ids]
        else:
            return self.item_embeddings, self.item_var

    def interact(self, user_embeddings, item_embeddings):
        # This interaction function returns the mean and variance for the gaussian hidden variable
        mu = (user_embeddings[0] * item_embeddings[0]).sum(1)
        var = 1 / torch.exp(user_embeddings[1] * item_embeddings[1])
        return mu, var

    def uncertain_transform(self, obj):
        mu, var = obj
        # Logistic normal approximated analytical form for the mean
        constant =  3 / torch.pi**2
        denom = torch.sqrt(1 + constant * var)
        mu_logistic_normal = self.sigmoid(mu / denom)
        return mu_logistic_normal
    
    def interact_return_logit_normal(self, user_embeddings, item_embeddings):
        mu = (user_embeddings[0] * item_embeddings[0]).sum(1)
        var = 1 / torch.exp(user_embeddings[1] * item_embeddings[1])
        
        # Logistic normal approximated analytical forms
        constant =  3 / torch.pi**2
        denom = torch.sqrt(1 + constant * var)
        mu_logistic_normal = self.sigmoid(mu / denom)
        var_logistic_normal = mu_logistic_normal * (1 - mu_logistic_normal) * (1 - 1 / denom)
        return mu_logistic_normal, var_logistic_normal


# MF_SVI and PMF_SVI are test implementations that are functional, but hasn't been used in my experiments.

class MF_SVI(Implicit, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim):

        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim

    # Here I do MLE estimation, for this reason guide is blank and the params have no priors
    def model(self, user, item, Y=None):
        # Obs noise (alpha in the papers)
        obs_noise = torch.tensor(2.)

        # Init the embeddings according to N(0, 1/d)
        p_u = pyro.param('p_u', torch.randn(self.n_user, self.embedding_dim) / self.embedding_dim)
        q_i = pyro.param('q_i', torch.randn(self.n_item, self.embedding_dim) / self.embedding_dim)

        with pyro.plate('Relevance', len(user)):
            mean = (p_u[user] * q_i[item]).sum(1)
            r_ui = pyro.sample('r_ui', dist.Normal(mean, obs_noise.sqrt()), obs=torch.tensor(Y).float())

    def guide(self, user, item, Y=None):
        pass

    def fix_learned_embeddings(self):
        model.user_embeddings = pyro.param('p_u').detach()
        model.item_embeddings = pyro.param('q_i').detach()

    def get_user_embeddings(self, user_ids):
        return self.user_embeddings[user_ids]
    
    def get_item_embeddings(self, item_ids=None):
        if item_ids is not None:
            return self.item_embeddings[item_ids]
        else:
            return self.item_embeddings
    
    def interact(self, user_embeddings, item_embeddings):
        relevance = (user_embeddings * item_embeddings).sum(1)
        return relevance


class PMF_SVI(Implicit, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim):

        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim

    # Here I do MAP estimation -- It differs from MLE because it's regularized by the prior.
    # The guide introduce a learnable variational parameter following a delta distribution (all prob at a single point), which will be the MAP estimate.
    def model(self, user, item, Y=None):
        # Obs noise precision (alpha^-1 in the papers)
        obs_noise = torch.tensor(2.)
        # Embeddings precision (alpha_u, alpha_i) in the paper
        # Higher values -> More precision -> Less variance -> More reguralization
        τ_u = torch.tensor(.004)
        τ_i = torch.tensor(.004)
        
        p_u = pyro.sample('p_u', dist.Normal(torch.zeros(self.n_user, self.embedding_dim), 1/τ_u.sqrt()).to_event(2))
        q_i = pyro.sample('q_i', dist.Normal(torch.zeros(self.n_item, self.embedding_dim), 1/τ_i.sqrt()).to_event(2))

        with pyro.plate('Relevance', len(user)):
            mean = (p_u[user] * q_i[item]).sum(1)
            r_ui = pyro.sample('r_ui', dist.Normal(mean, 1/obs_noise.sqrt()), obs=torch.tensor(Y).float())

    def guide(self, user, item, Y=None):
        # Init the embeddings according to N(0, 1/d)
        µ_u = pyro.param('µ_u', torch.randn(self.n_user, self.embedding_dim) / self.embedding_dim)
        µ_i = pyro.param('µ_i', torch.randn(self.n_item, self.embedding_dim) / self.embedding_dim)

        p_u = pyro.sample('p_u', dist.Delta(µ_u).to_event(2))
        q_i = pyro.sample('q_i', dist.Delta(µ_i).to_event(2))

    def fix_learned_embeddings(self):
        model.user_embeddings = pyro.param('µ_u').detach()
        model.item_embeddings = pyro.param('µ_i').detach()

    def get_user_embeddings(self, user_ids):
        return self.user_embeddings[user_ids]
    
    def get_item_embeddings(self, item_ids=None):
        if item_ids is not None:
            return self.item_embeddings[item_ids]
        else:
            return self.item_embeddings
    
    def interact(self, user_embeddings, item_embeddings):
        relevance = (user_embeddings * item_embeddings).sum(1)
        return relevance
                

