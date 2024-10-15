import numpy as np

__all__ = ["correlation", "quick_correlation_one_to_many"]


def add_noise_to_constant_rows(x, epsilon=1e-15):
    """Adds noise to rows that are constant to avoid division by zero.

    Args:
        x: m x n values

    Returns:
        m x n values
    """
    if epsilon is not None:
        x = np.array(x)
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        # check which rows are nearly constant
        is_constant = np.all(np.isclose(x, x[:, 0][:, None]), axis=1)

        # add noise to constant rows
        if is_constant.any():
            x[is_constant] += np.random.uniform(
                -epsilon, epsilon, size=x[is_constant].shape
            )
    return x


def correlation(x, x_pred, epsilon=0):
    """Slow correlation computation between one x and many x_pred values including
    significance test.
    Args:
        x: n values, array like
        x_pred: m x n values

    Returns:
        array of correlations: m values
        array of pvalues: n values
    """
    x_pred = add_noise_to_constant_rows(x_pred, epsilon=epsilon)

    from scipy.stats import pearsonr

    _correlations = [pearsonr(x, _x_pred) for _x_pred in x_pred]
    correlations = [c.statistic for c in _correlations]
    pvalues = [c.pvalue for c in _correlations]
    return np.array(correlations), np.array(pvalues)


def quick_correlation_one_to_many(x, Y, zero_nans=True):
    """Fast correlation computation between one x and many Y values.

    Args:
        x: n values, array like
        Y: m x n values

    Returns:
        array of correlations: m values
    """
    x = x - np.mean(x)
    Y = Y - np.mean(Y, axis=1, keepdims=True)
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(Y, axis=1)

    if zero_nans:
        corr = (x @ Y.T) / (normx * normy)
        corr[np.isnan(corr)] = 0
        return corr
    return (x @ Y.T) / (normx * normy)
