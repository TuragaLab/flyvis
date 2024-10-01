"""Loss functions compatible with torch loss function API."""

import torch


def l2norm(y_est, y_gt, **kwargs):
    """
    Mean root cumulative squared error across last three dimensions
    """
    return (((y_est - y_gt) ** 2).sum(dim=(1, 2, 3))).sqrt().mean()


def epe(y_est, y_gt, **kwargs):
    """
    Average endpointerror, conventionally reported in optic flow tasks.
    Susceptible to outliers because it is squaring the errors.

    (#samples, #frames, ndim, #hexals or #features)
    """
    return torch.sqrt(((y_est - y_gt) ** 2).sum(dim=2)).mean()
