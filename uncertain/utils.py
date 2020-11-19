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


def minibatch(interactions, batch_size):

    for i in range(0, len(interactions), batch_size):
        yield interactions[i:i + batch_size]


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterion don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )

