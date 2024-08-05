import math

import numpy as np
import torch
import torch.nn.functional as nnf

from .augmentation import Augmentation


class Interpolate(Augmentation):
    def __init__(self, original_framerate: int, target_framerate: float, mode: str):
        self.original_framerate = original_framerate
        self.target_framerate = target_framerate
        self.mode = mode
        self.align_corners = (
            True if mode in ["linear", "bilinear", "bicubic", "trilinear"] else None
        )

    def transform(self, sequence: torch.Tensor, dim: int = 0):
        """Resamples the sequence along the specified dim.

        sequence: sequence to resample of n_dim == 3. See nnf.interpolate for details.
        dim: dim along which to resample.
        """
        assert sequence.ndim == 3, "only 3D sequences are supported"
        if sequence.dtype == torch.long:
            sequence = sequence.float()
        return nnf.interpolate(
            sequence.transpose(dim, -1),
            size=math.ceil(
                self.target_framerate / self.original_framerate * sequence.shape[dim]
            ),
            mode=self.mode,
            align_corners=self.align_corners,
        ).transpose(dim, -1)

    def piecewise_constant_indices(self, length: int):
        """Returns indices to sample from a sequence with piecewise constant
        interpolation."""
        indices = torch.arange(length, dtype=torch.float)[None, None]
        return (
            nnf.interpolate(
                indices,
                size=math.ceil(self.target_framerate / self.original_framerate * length),
                mode="nearest-exact",
                align_corners=None,
            )
            .flatten()
            .long()
        )


class CropFrames(Augmentation):
    def __init__(
        self, n_frames: int, start: int = 0, all_frames: bool = False, random=False
    ):
        self.n_frames = n_frames
        self.all_frames = all_frames
        self.start = start
        self.random = random

    def transform(self, sequence: torch.Tensor, dim: int = 0):
        """Randomly crops the sequence along the specified dim.

        sequence: sequence to crop of shape (..., n_frames, ...)
        dim: dim along which to crop.
        """
        if self.all_frames:
            return sequence
        total_seq_length = sequence.shape[dim]
        if self.n_frames > total_seq_length:
            raise ValueError(
                f"cannot crop {self.n_frames} frames from a total"
                f" of {total_seq_length} frames"
            )
        start = self.start if self.random else 0
        indx = [slice(None)] * sequence.ndim
        indx[dim] = slice(start, start + self.n_frames)
        return sequence[indx]

    def set_or_sample(self, start=None, total_sequence_length=None):
        if total_sequence_length and self.n_frames > total_sequence_length:
            raise ValueError(
                f"cannot crop {self.n_frames} frames from a total"
                f" of {total_sequence_length} frames"
            )
        if start is None and total_sequence_length:
            start = np.random.randint(
                low=0, high=total_sequence_length - self.n_frames or 1
            )
        self.start = start


# class InterpolateFrames(Augmentation):
#     def transform(self, sequence: torch.Tensor, size: int, dim: int = 0):
#         return nnf.interpolate(
#             sequence.transpose(dim, -1),
#             size=size,
#             mode="linear",
#             align_corners=True,
#         ).transpose(dim, -1)


def get_temporal_sample_indices(
    n_frames: int,
    total_seq_length: int,
    framerate: int,
    dt: float,
    augment: bool,
) -> torch.Tensor:
    """Returns temporal indices to sample from a sequence.

    Args:
        n_frames: number of sequence frames to sample from.
        total_seq_length: total sequence length.
        framerate: original framerate of the sequence.
        dt: sampling time constant.
        augment: if True, picks the start frame at random. If False,
            starts at 0.

    Note: interpolates between start_index and start_index + n_frames and
    rounds the resulting float values to integer to create indices. This can
    lead to irregularities in terms of how many times each raw data frame is
    sampled.
    """
    if n_frames > total_seq_length:
        raise ValueError(
            f"cannot interpolate between {n_frames} frames from a total"
            f" of {total_seq_length} frames"
        )
    start = 0
    if augment:
        last_valid_start = total_seq_length - n_frames or 1
        start = np.random.randint(low=0, high=last_valid_start)
    return torch.arange(start, start + n_frames - 1e-6, dt * framerate).long()


def interp(sequence: torch.Tensor, size: int) -> torch.Tensor:
    """Linearly interpolates a sequence along the first dimension.
    Args:
        sequence: sequence of shape (n_frames, n_features, n_hexals).
        size: new size for n_frames dimension after interpolation.
    """
    return
