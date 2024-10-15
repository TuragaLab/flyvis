from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as nnf

from .augmentation import Augmentation


class Interpolate(Augmentation):
    """Interpolate a sequence to a target framerate.

    Attributes:
        original_framerate (int): The original framerate of the sequence.
        target_framerate (float): The target framerate after interpolation.
        mode (str): The interpolation mode.
        align_corners (bool | None): Alignment of corners for interpolation.
    """

    def __init__(self, original_framerate: int, target_framerate: float, mode: str):
        self.original_framerate = original_framerate
        self.target_framerate = target_framerate
        self.mode = mode
        self.align_corners = (
            True if mode in ["linear", "bilinear", "bicubic", "trilinear"] else None
        )

    def transform(self, sequence: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Resample the sequence along the specified dimension.

        Args:
            sequence: Sequence to resample of ndim == 3.
            dim: Dimension along which to resample.

        Returns:
            torch.Tensor: Resampled sequence.

        Raises:
            AssertionError: If the input sequence is not 3D.
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

    def piecewise_constant_indices(self, length: int) -> torch.Tensor:
        """Return indices to sample from a sequence with piecewise constant interpolation.

        Args:
            length: Length of the original sequence.

        Returns:
            torch.Tensor: Indices for piecewise constant interpolation.
        """
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
    """Crop frames from a sequence.

    Attributes:
        n_frames (int): Number of frames to crop.
        all_frames (bool): Whether to return all frames.
        start (int): Starting frame for cropping.
        random (bool): Whether to use random cropping.
    """

    def __init__(
        self,
        n_frames: int,
        start: int = 0,
        all_frames: bool = False,
        random: bool = False,
    ):
        self.n_frames = n_frames
        self.all_frames = all_frames
        self.start = start
        self.random = random

    def transform(self, sequence: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Crop the sequence along the specified dimension.

        Args:
            sequence: Sequence to crop of shape (..., n_frames, ...).
            dim: Dimension along which to crop.

        Returns:
            torch.Tensor: Cropped sequence.

        Raises:
            ValueError: If n_frames is greater than the total sequence length.
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

    def set_or_sample(
        self, start: int | None = None, total_sequence_length: int | None = None
    ):
        """Set or sample the starting frame for cropping.

        Args:
            start: Starting frame for cropping.
            total_sequence_length: Total length of the sequence.

        Raises:
            ValueError: If n_frames is greater than the total sequence length.
        """
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


def get_temporal_sample_indices(
    n_frames: int,
    total_seq_length: int,
    framerate: int,
    dt: float,
    augment: bool,
) -> torch.Tensor:
    """Return temporal indices to sample from a sequence.

    Args:
        n_frames: Number of sequence frames to sample from.
        total_seq_length: Total sequence length.
        framerate: Original framerate of the sequence.
        dt: Sampling time constant.
        augment: If True, picks the start frame at random. If False, starts at 0.

    Returns:
        torch.Tensor: Temporal indices for sampling.

    Raises:
        ValueError: If n_frames is greater than total_seq_length.

    Note:
        Interpolates between start_index and start_index + n_frames and
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
