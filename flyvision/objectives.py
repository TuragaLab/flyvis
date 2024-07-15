"""Loss functions."""

import functools

import torch


def check_shape(f):
    """To validate the shape of prediction and target arguments for
    preventing hard-to-track errors through e.g. wrong shapes and broadcasting.
    """

    @functools.wraps(f)
    def wrapper(y_gt, y_est, **kwargs):
        shape1 = y_est.shape
        shape2 = y_gt.shape
        if shape1 != shape2:
            raise AssertionError(
                f"y_est is of shape {shape1} and y_gt is of shape {shape2}"
            )
        if len(shape1) != 4:
            raise AssertionError(
                f"y_est, y_gt are of shape {shape1}. Expected four dimensions."
            )
        return f(y_gt, y_est)

    return wrapper


class Objective:
    """Base class for objectives."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def _call(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__


class L2Norm(Objective):
    """Sample-average L2 norm of prediction error."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _call(self, y_gt, y_est, **kwargs):
        return self.__class__.l2norm(y_gt, y_est, **kwargs)

    @staticmethod
    @check_shape
    def l2norm(y_gt, y_est, **kwargs):
        """
        Dims are (#samples, #frames, ndim, #hexals or #features)

        Note: the magnitude is dependent on #frames and #hexals because of sum.
        Averages over samples.
        """
        error = y_est - y_gt
        return ((error**2).sum(dim=(1, 2, 3))).sqrt().mean()


class EPE(Objective):
    """Average endpointerror, conventionally reported in optic flow tasks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _call(self, y_gt, y_est, **kwargs):
        return self.__class__.epe(y_gt, y_est, **kwargs)

    @staticmethod
    @check_shape
    def epe(y_gt, y_est, **kwargs):
        """
        Dims are (#samples, #frames, ndim, #hexals or #features)

        Note: the magnitude is independent of #frames and #hexals because the
        norm is calculated only over the two-dimensional flow field and then
        averaged.
        """
        return torch.linalg.norm(y_gt - y_est, ord=2, dim=2).mean()
