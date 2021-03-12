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

    if data.type == 'Explicit':
        for i in range(0, len(data), batch_size):
            yield data.interactions[i:i + batch_size], data.ratings[i:i + batch_size]
    else:
        for i in range(0, len(data), batch_size):
            yield data.interactions[i:i + batch_size], None


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


def evaluate(model, test, train):

    out = {}
    loader = minibatch(test, batch_size=int(1e5))
    est = []
    if model.is_uncertain:
        unc = []
        for interactions, _ in loader:
            predictions = model.predict(interactions[:, 0], interactions[:, 1])
            est.append(predictions[0])
            unc.append(predictions[1])
        unc = torch.hstack(unc)
    else:
        for interactions, _ in loader:
            est.append(model.predict(interactions[:, 0], interactions[:, 1]))
    est = torch.hstack(est)
    
    p, r, a, s = recommendation_score(model, test, train, max_k=10)
    
    out['RMSE'] = rmse_score(est, test.ratings)
    out['Precision'] = p.mean(axis=0)
    out['Recall'] = r.mean(axis=0)
    
    if model.is_uncertain:
        error = torch.abs(test.ratings - est)
        idx = torch.randperm(len(unc))[:int(1e5)]
        quantiles = torch.quantile(unc[idx], torch.linspace(0, 1, 21, device=unc.device, dtype=unc.dtype))
        out['Quantile RMSE'] = torch.zeros(20)
        for idx in range(20):
            ind = torch.bitwise_and(quantiles[idx] <= unc, unc < quantiles[idx + 1])
            out['Quantile RMSE'][idx] = torch.sqrt(torch.square(error[ind]).mean())
        quantiles = torch.quantile(a, torch.linspace(0, 1, 21, device=a.device, dtype=a.dtype))
        out['Quantile MAP'] = torch.zeros(20)
        for idx in range(20):
            ind = torch.bitwise_and(quantiles[idx] <= a, a < quantiles[idx + 1])
            out['Quantile MAP'][idx] = p[ind, -1].mean()
        out['RRI'] = s.nansum(0) / (~s.isnan()).float().sum(0)
        out['Correlation'] = correlation(error, unc)
        out['RPI'] = rpi_score(error, unc)
        out['Classification'] = classification(error, unc)
    
    return out


def evaluate_zhu(model, test, relevance_threshold=None, max_k=20):

    out = {}
    loader = minibatch(test, batch_size=int(1e5))
    est = []
    if model.is_uncertain:
        unc = []
        for interactions, _ in loader:
            predictions = model.predict(interactions[:, 0], interactions[:, 1])
            est.append(predictions[0])
            unc.append(predictions[1])
        unc = torch.hstack(unc)
    else:
        for interactions, _ in loader:
            est.append(model.predict(interactions[:, 0], interactions[:, 1]))
    est = torch.hstack(est)
    
    error = torch.abs(test.ratings - est)
    out['MAE'] = error.mean().item()
    out['RMSE'] = error.square().mean().sqrt().item()
    
    if model.is_uncertain:
        unc_avg, unc_ste = unc.mean(), (unc - unc.mean()).abs().mean()
        precision = []
        rri = []
        denominator = torch.arange(1, max_k + 1, device=test.interactions.device)
        for user_id in range(model.num_users):
            idx = test.interactions[:, 0] == user_id
            if idx.sum() == 0:
                continue
            relevance = test.ratings[idx] >= relevance_threshold
            est_, unc_ = est[idx], unc[idx]
            recommendations = est_.argsort(descending=True)
            hit_indicator = torch.full([max_k], np.nan, device=test.interactions.device)
            hit_indicator[:len(recommendations)] = relevance[recommendations][:max_k]
            precision.append(hit_indicator.cumsum(0) / denominator)
            recommendation_unc = torch.full([max_k], np.nan, device=test.interactions.device)
            recommendation_unc[:len(recommendations)] = unc_[recommendations][:max_k]
            rri.append((hit_indicator * (unc_avg - recommendation_unc).cumsum(0)) /
                       hit_indicator.cumsum(0) / unc_ste)
    
        out['Precision'] = np.nanmean(torch.vstack(precision).cpu().numpy(), axis=0).tolist()
        out['RRI'] = np.nanmean(torch.vstack(rri).cpu().numpy(), axis=0).tolist()
    
        out['RPI'] = rpi_score(error, unc).item()
        p, s = correlation(error, unc)
        out['Pearson'], out['Spearman'] = p.item(), s.item()
        out['Quantile RMSE'] = []
        idx = torch.randperm(len(unc))[:int(1e5)]
        quantiles = torch.quantile(unc[idx], torch.linspace(0, 1, 21, device=unc.device, dtype=unc.dtype))
        for idx in range(20):
            ind = torch.bitwise_and(quantiles[idx] <= unc, unc < quantiles[idx + 1])
            out['Quantile RMSE'].append(torch.sqrt(torch.square(error[ind]).mean()).item())
    
        out['AUC'] = classification(error, unc).item()
    
    else:
        precision = []
        denominator = torch.arange(1, max_k + 1, device=test.interactions.device)
        for user_id in range(model.num_users):
            idx = test.interactions[:, 0] == user_id
            if idx.sum() == 0:
                continue
            relevance = test.ratings[idx] >= relevance_threshold
            est_ = est[idx]
    
            hits = torch.full([max_k], np.nan, device=test.interactions.device)
            hits_ = relevance[est_.argsort(descending=True)][:max_k]
            hits[:len(hits_)] = hits_
            precision.append(hits.cumsum(0) / denominator)
    
        out['Precision'] = np.nanmean(torch.vstack(precision).cpu().numpy(), axis=0).tolist()
    
    return out