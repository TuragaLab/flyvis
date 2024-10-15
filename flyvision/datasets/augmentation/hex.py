"""Transformations and augmentations on hex-lattices."""

from typing import Any, List, Optional

import numpy as np
import torch

from .augmentation import Augmentation
from .utils import (
    flip_matrix,
    flip_permutation_index,
    rotation_matrix,
    rotation_permutation_index,
)

__all__ = [
    "HexRotate",
    "HexFlip",
    "ContrastBrightness",
    "PixelNoise",
    "GammaCorrection",
]


class HexRotate(Augmentation):
    """Rotate a sequence of regular hex-lattices by multiple of 60 degree.

    Args:
        extent: Extent of the regular hexagonal grid in columns.
        n_rot: Number of 60 degree rotations. 0-5.
        p_rot: Probability of rotating. If None, no rotation is performed.

    Attributes:
        extent (int): Extent of the regular hexagonal grid in columns.
        n_rot (int): Number of 60 degree rotations. 0-5.
        p_rot (float): Probability of rotating.
        permutation_indices (dict): Cached indices for rotation.
        rotation_matrices (dict): Cached rotation matrices for rotation.
    """

    def __init__(self, extent: int, n_rot: int = 0, p_rot: float = 0.5) -> None:
        self.extent = extent
        self.rotation_matrices: dict = {}
        self.permutation_indices: dict = {}
        for n in range(6):
            R2 = rotation_matrix(n * 60 * np.pi / 180, three_d=False)
            R3 = rotation_matrix(n * 60 * np.pi / 180, three_d=True)
            self.rotation_matrices[n] = [R2, R3]
            self.permutation_indices[n] = rotation_permutation_index(extent, n)
        self.n_rot = n_rot
        self.p_rot = p_rot
        self.set_or_sample(n_rot)

    @property
    def n_rot(self) -> int:
        """Get the number of rotations."""
        return self._n_rot

    @n_rot.setter
    def n_rot(self, n_rot: int) -> None:
        """Set the number of rotations."""
        self._n_rot = n_rot % 6

    def rotate(self, seq: torch.Tensor) -> torch.Tensor:
        """Rotate a sequence on a regular hexagonal lattice.

        Args:
            seq: Sequence of shape (frames, dims, hexals).

        Returns:
            Rotated sequence of the same shape as seq.
        """
        dims = seq.shape[-2]
        seq = seq[..., self.permutation_indices[self.n_rot]]
        if dims > 1:
            seq = self.rotation_matrices[self.n_rot][dims - 2] @ seq
        return seq

    def transform(self, seq: torch.Tensor, n_rot: Optional[int] = None) -> torch.Tensor:
        """Rotate a sequence on a regular hexagonal lattice.

        Args:
            seq: Sequence of shape (frames, dims, hexals).
            n_rot: Optional number of rotations to apply.

        Returns:
            Rotated sequence of the same shape as seq.
        """
        if n_rot is not None:
            self.n_rot = n_rot
        if self.n_rot > 0:
            return self.rotate(seq)
        return seq

    def set_or_sample(self, n_rot: Optional[int] = None) -> None:
        """Set or sample the number of rotations.

        Args:
            n_rot: Number of rotations to set. If None, sample randomly.
        """
        if n_rot is None:
            n_rot = (
                np.random.randint(low=1, high=6)
                if self.p_rot and self.p_rot > np.random.rand()
                else 0
            )
        self.n_rot = n_rot


class HexFlip(Augmentation):
    """Flip a sequence of regular hex-lattices across one of three hex-axes.

    Args:
        extent: Extent of the regular hexagonal grid.
        axis: Flipping axis. 0 corresponds to no flipping.
        p_flip: Probability of flipping. If None, no flipping is performed.
        flip_axes: List of valid flipping axes. Can contain 0, 1, 2, 3.

    Attributes:
        extent (int): Extent of the regular hexagonal grid.
        axis (int): Flipping axis.
        p_flip (float): Probability of flipping.
        flip_axes (List[int]): List of valid flipping axes.
        permutation_indices (dict): Cached indices for flipping.
        rotation_matrices (dict): Cached rotation matrices for flipping.

    Note:
        This is to avoid redundant transformations from rotation and flipping.
        For example, flipping across the 1st axis is equivalent to rotating by
        240 degrees and flipping across the 2nd axis.
    """

    def __init__(
        self,
        extent: int,
        axis: int = 0,
        p_flip: float = 0.5,
        flip_axes: List[int] = [0, 1, 2, 3],
    ) -> None:
        self.extent = extent
        self.rotation_matrices: dict = {}
        self.permutation_indices: dict = {}
        for n, angle in enumerate([90, 150, 210], 1):
            R2 = flip_matrix(np.radians(angle), three_d=False)
            R3 = flip_matrix(np.radians(angle), three_d=True)
            self.rotation_matrices[n] = [R2, R3]
            self.permutation_indices[n] = flip_permutation_index(extent, n)
        self.flip_axes = flip_axes
        self.axis = axis
        self.p_flip = p_flip
        self.set_or_sample(axis)

    @property
    def axis(self) -> int:
        """Get the flipping axis."""
        return self._axis

    @axis.setter
    def axis(self, axis: int) -> None:
        """Set the flipping axis."""
        assert (
            axis in self.flip_axes
        ), f"{axis} is not a valid axis. Must be in {self.flip_axes}."
        self._axis = axis

    def flip(self, seq: torch.Tensor) -> torch.Tensor:
        """Flip a sequence on a regular hexagonal lattice.

        Args:
            seq: Sequence of shape (frames, dims, hexals).

        Returns:
            Flipped sequence of the same shape as seq.
        """
        dims = seq.shape[-2]
        seq = seq[..., self.permutation_indices[self.axis]]
        if dims > 1:
            seq = self.rotation_matrices[self.axis][dims - 2] @ seq
        return seq

    def transform(self, seq: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        """Flip a sequence on a regular hexagonal lattice.

        Args:
            seq: Sequence of shape (frames, dims, hexals).
            axis: Optional flipping axis to apply.

        Returns:
            Flipped sequence of the same shape as seq.
        """
        if axis is not None:
            self.axis = axis
        if self.axis > 0:
            return self.flip(seq=seq)
        return seq

    def set_or_sample(self, axis: Optional[int] = None) -> None:
        """Set or sample the flipping axis.

        Args:
            axis: Flipping axis to set. If None, sample randomly.
        """
        if axis is None:
            axis = (
                np.random.randint(low=1, high=max(self.flip_axes) + 1)
                if self.p_flip and self.p_flip > np.random.rand()
                else 0
            )
        self.axis = axis


class ContrastBrightness(Augmentation):
    """Contrast transformation.

    The transformation is described as:
    ```python
    pixel = max(0, contrast_factor * (pixel - 0.5) + 0.5
                + contrast_factor * brightness_factor)
    ```

    Args:
        contrast_factor: Contrast factor.
        brightness_factor: Brightness factor.
        contrast_std: Standard deviation of the contrast factor.
        brightness_std: Standard deviation of the brightness factor.

    Attributes:
        contrast_std (float): Standard deviation of the contrast factor.
        brightness_std (float): Standard deviation of the brightness factor.
        contrast_factor (float): Contrast factor.
        brightness_factor (float): Brightness factor.
    """

    def __init__(
        self,
        contrast_factor: Optional[float] = None,
        brightness_factor: Optional[float] = None,
        contrast_std: float = 0.2,
        brightness_std: float = 0.1,
    ) -> None:
        self.contrast_std = contrast_std
        self.brightness_std = brightness_std
        self.set_or_sample(contrast_factor, brightness_factor)

    def transform(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to a sequence.

        Args:
            seq: Input sequence.

        Returns:
            Transformed sequence.
        """
        if self.contrast_factor is not None:
            return (
                self.contrast_factor * (seq - 0.5)
                + 0.5
                + self.contrast_factor * self.brightness_factor
            ).clamp(0)
        return seq

    def set_or_sample(
        self,
        contrast_factor: Optional[float] = None,
        brightness_factor: Optional[float] = None,
    ) -> None:
        """Set or sample contrast and brightness factors.

        Args:
            contrast_factor: Contrast factor to set. If None, sample randomly.
            brightness_factor: Brightness factor to set. If None, sample randomly.
        """
        if contrast_factor is None:
            # behaves like N(1, std) for small std, slightly biased towards
            # high contrast in particular for large std deviations
            # TODO: implement other sampling schemes
            contrast_factor = (
                np.exp(np.random.normal(0, self.contrast_std))
                if self.contrast_std
                else None
            )
        if brightness_factor is None:
            brightness_factor = (
                np.random.normal(0, self.brightness_std) if self.brightness_std else 0.0
            )
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor


class PixelNoise(Augmentation):
    """Pixelwise gaussian noise.

    The transformation is described as:
    ```python
    pixel = pixel + N(0, std)
    ```

    It biases the signal to noise ratio: high for light, low for dark pixels.

    Args:
        std: Standard deviation of the gaussian noise.

    Attributes:
        std (float): Standard deviation of the gaussian noise.
    """

    def __init__(self, std: float = 0.08) -> None:
        self.std = std

    def transform(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to a sequence.

        Args:
            seq: Input sequence.

        Returns:
            Transformed sequence.
        """
        if self.std:
            noise = torch.randn_like(seq) * self.std
            return (seq + noise).clamp(0)
        return seq

    def set_or_sample(self, std: Optional[float] = None) -> None:
        """Set or sample the standard deviation of the gaussian noise.

        Args:
            std: Standard deviation of the gaussian noise to set.
                If None, no change is made.
        """
        if std is None:
            return
        self.std = std


class GammaCorrection(Augmentation):
    """Gamma correction.

    The transformation is described as:
    ```python
    pixel = pixel ** gamma
    ```

    Gamma > 1 increases the contrast, gamma < 1 decreases the contrast.

    Args:
        gamma: Gamma value.
        std: Standard deviation of the gamma value.

    Attributes:
        gamma: float
            Gamma value.
        std: Optional[float]
            Standard deviation of the gamma value.
    """

    def __init__(self, gamma: float = 1, std: Optional[float] = None) -> None:
        self.gamma = gamma
        self.std = std

    def transform(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to a sequence.

        Args:
            seq: Input sequence.

        Returns:
            Transformed sequence.
        """
        if self.gamma:
            return seq**self.gamma
        return seq

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "gamma" and value < 0:
            raise ValueError("Gamma must be positive.")
        return super().__setattr__(name, value)

    def set_or_sample(self, gamma: Optional[float] = None) -> None:
        """Set or sample the gamma value.

        Args:
            gamma: Gamma value to set. If None, sample randomly.
        """
        if gamma is None:
            gamma = max(0, np.random.normal(1, self.std)) if self.std else 1.0
        self.gamma = gamma
