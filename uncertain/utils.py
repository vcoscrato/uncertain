import numpy as np


def gpu(tensor, use_cuda=False):

    if use_cuda:
        return tensor.cuda()
    else:
        return tensor


def cpu(tensor):

    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def minibatch(data, batch_size):

    for i in range(0, len(data), batch_size):
        yield data.interactions[i:i + batch_size], data.ratings[i:i + batch_size]


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterion don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def sample_items(num_items, shape, random_state=None):
    """
    Randomly sample a number of items.
    Parameters
    ----------
    num_items: int
        Total number of items from which we should sample:
        the maximum value of a sampled item id will be smaller
        than this.
    shape: int or tuple of ints
        Shape of the sampled array.
    random_state: np.random.RandomState instance, optional
        Random state to use for sampling.
    Returns
    -------
    items: np.array of shape [shape]
        Sampled item ids.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    items = random_state.randint(0, num_items, shape, dtype=np.int64)

    return items
