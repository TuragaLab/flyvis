"""Loss functions compatible with torch loss function API."""

import functools
import inspect

import torch


class Loss:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input, target):
        raise NotImplementedError


def check_shape(expected_dims):
    """Decorator to validate the shape of prediction and target arguments
    to prevent hard-to-track errors caused by wrong shapes and broadcasting.
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, input, target, **kwargs):
            if input.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch: input has shape {input.shape}, "
                    f"but target has shape {target.shape}"
                )
            if len(input.shape) != expected_dims:
                raise ValueError(
                    f"Expected {expected_dims} dimensions, but got {len(input.shape)}"
                )
            # Check if the original function supports kwargs
            if "kwargs" in self.sig.parameters:
                return f(self, input, target, **kwargs)
            else:
                return f(self, input, target)

        return wrapper

    return decorator


class L2Norm(Loss):
    """Sample-average L2 norm of prediction error.

    Dims are (#samples, #frames, ndim, #hexals or #features).

    Note: the magnitude is dependent on #frames and #hexals because of sum.
    Averages over samples.
    """

    def __init__(self):
        super(L2Norm, self).__init__()
        self.sig = inspect.signature(self.forward)

    @check_shape(expected_dims=4)
    def forward(self, input, target):
        return torch.sqrt(((input - target) ** 2).sum(dim=(1, 2, 3))).mean()


class EPE(Loss):
    """Average endpoint error, conventionally reported in optic flow tasks.

    Dims are (#samples, #frames, ndim, #hexals or #features).

    Note: the magnitude is independent of #frames and #hexals because the
    norm is calculated only over the two-dimensional flow field and then
    averaged.
    """

    def __init__(self):
        super(EPE, self).__init__()
        self.sig = inspect.signature(self.forward)

    @check_shape(expected_dims=4)
    def forward(self, input, target):
        return torch.sqrt(((input - target) ** 2).sum(dim=2)).mean()


def main():
    input = torch.rand((10, 5, 3, 4))
    target = torch.rand((10, 5, 3, 4))

    l2_loss = L2Norm()
    epe_loss = EPE()

    print(
        "L2 Norm Loss:",
        l2_loss(
            input,
            target,
        ),
    )
    print(
        "EPE Loss:",
        epe_loss(
            input,
            target,
        ),
    )


if __name__ == "__main__":
    main()
