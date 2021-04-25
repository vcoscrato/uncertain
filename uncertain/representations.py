import torch
from uncertain.layers import ZeroEmbedding, ScaledEmbedding


class BiasNet(torch.nn.Module):
    """
    Bias representation.
    The score for a user-item pair is given by the
    sum of the item and the user biases.

    Parameters
    ----------
    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(self, num_users, num_items, sparse=False):

        super(BiasNet, self).__init__()

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------
        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------
        predictions: tensor
            Tensor of predictions.
        """

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return user_bias + item_bias


class FunkSVDNet(torch.nn.Module):
    """
    Bilinear factorization representation.
    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------
    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(self, num_users, num_items, embedding_dim, sparse=False):

        super(FunkSVDNet, self).__init__()

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------
        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------
        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        return dot


class CPMFNet(torch.nn.Module):
    """
    Confidence-Aware probabilistic matrix factorization representation.

    Parameters
    ----------
    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    sparse: boolean, optional
        Use sparse gradients.
    """
    def __init__(self, num_users, num_items, embedding_dim, sparse=False):

        super(CPMFNet, self).__init__()

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)

        self.user_gammas = ScaledEmbedding(num_users, 1, sparse=sparse)
        self.item_gammas = ScaledEmbedding(num_items, 1, sparse=sparse)

        self.var_activation = torch.nn.Softplus()

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------
        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------
        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids).squeeze()
        item_embedding = self.item_embeddings(item_ids).squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        user_gamma = self.user_gammas(user_ids).squeeze()
        item_gamma = self.item_gammas(item_ids).squeeze()

        var = self.var_activation(user_gamma + item_gamma)

        return dot, var


class OrdRecNet(torch.nn.Module):
    """
    OrdRec representation.

    Parameters
    ----------
    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(self, num_users, num_items, num_labels, embedding_dim, sparse=False):

        super(OrdRecNet, self).__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)

        self.user_betas = ZeroEmbedding(num_users, num_labels - 1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------
        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------
        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids).squeeze()
        item_embedding = self.item_embeddings(item_ids).squeeze()
        y = (user_embedding * item_embedding).sum(1).reshape(-1, 1)

        user_beta = self.user_betas(user_ids)
        user_beta[:, 1:] = torch.exp(user_beta[:, 1:])
        user_distribution = torch.div(1, 1 + torch.exp(y - user_beta.cumsum(1)))

        one = torch.ones((len(user_distribution), 1), device=user_beta.device)
        user_distribution = torch.cat((user_distribution, one), 1)

        user_distribution[:, 1:] -= user_distribution[:, :-1].clone()

        return user_distribution
