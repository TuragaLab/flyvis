import functools


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


@check_shape
def mean_root_norm(y_gt, y_est, **kwargs):
    """(#samples, n_frames, ndim, n_hexals/columns)

    Note, absolute magnitudes dependent on n_frames and n_hexals cause of sum.
    To compare losses with different n_frames, n_hexals evaluate mean_root_average_norm.
    """
    error = y_est - y_gt
    return ((error**2).sum(dim=(1, 2, 3)) + 1e-9).sqrt().mean()


@check_shape
def mean_root_average_norm(y_gt, y_est, **kwargs):
    """(#samples, n_frames, ndim, n_hexals/columns)

    Note, averages the norm in ndim (e.g. optic flow image plane coordinates)
    across n_frames and n_hexals.
    """
    error = y_est - y_gt
    return ((error**2).sum(dim=2).mean(dim=(1, 2)) + 1e-9).sqrt().mean()
