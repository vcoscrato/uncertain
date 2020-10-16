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


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterion don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )

