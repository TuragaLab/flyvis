"""Loss functions compatible with torch loss function API."""

from typing import Any

import torch

__all__ = ["l2norm", "epe"]


def l2norm(y_est: torch.Tensor, y_gt: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """
    Calculate the mean root cumulative squared error across the last three dimensions.

    Args:
        y_est: The estimated tensor.
        y_gt: The ground truth tensor.
        **kwargs: Additional keyword arguments.

    Returns:
        The mean root cumulative squared error.
    """
    return (((y_est - y_gt) ** 2).sum(dim=(1, 2, 3))).sqrt().mean()


def epe(y_est: torch.Tensor, y_gt: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """
    Calculate the average endpoint error, conventionally reported in optic flow tasks.

    Args:
        y_est: The estimated tensor with shape
            (samples, frames, ndim, hexals_or_features).
        y_gt: The ground truth tensor with the same shape as y_est.
        **kwargs: Additional keyword arguments.

    Returns:
        The average endpoint error.
    """
    return torch.sqrt(((y_est - y_gt) ** 2).sum(dim=2)).mean()
