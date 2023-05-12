"""Transformations and augmentations on hex-lattices."""
import numpy as np
import torch

from typing import Any, Optional

from .augmentation import Augmentation
from flyvision import utils

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
        extent (int): extent of the regular hexagonal grid in columns.
        n_rot (int): number of 60 degree rotations. 0-5.
        p_rot (float): probability of rotating. If None, no rotation is
            performed.

    Attributes:
        same as args and
        permutation_indices (dict): cached indices for rotation.
        rotation_matrices (dict): cached rotation matrices for rotation.
    """

    def __init__(self, extent: int, n_rot: int = 0, p_rot=0.5):
        self.extent = extent
        # cache indices and matrices cause this must be fast at runtime
        # for augmenting sequences
        self.rotation_matrices = {}
        self.permutation_indices = {}
        for n in range(6):
            R2 = rotation_matrix(n * 60 * np.pi / 180, three_d=False)
            # for tasks predicting 3d variables
            R3 = rotation_matrix(n * 60 * np.pi / 180, three_d=True)
            self.rotation_matrices[n] = [R2, R3]
            self.permutation_indices[n] = rotation_permutation_index(extent, n)
        self.n_rot = n_rot
        self.p_rot = p_rot
        self.set_or_sample(n_rot)

    @property
    def n_rot(self):
        return self._n_rot

    @n_rot.setter
    def n_rot(self, n_rot):
        self._n_rot = n_rot % 6

    def rotate(self, seq: torch.Tensor):
        """Rotates a sequence on a regular hexagonal lattice.

        Args:
            seq: sequence of shape (frames, dims, hexals).
        Returns:
            tensor: rotated sequence of the same shape as seq.
        """
        dims = seq.shape[-2]

        # rearrange the hexals
        seq = seq[..., self.permutation_indices[self.n_rot]]

        # rotate the vectors in the hexal plane
        if dims > 1:
            seq = self.rotation_matrices[self.n_rot][dims - 2] @ seq

        return seq

    def __call__(self, seq: torch.Tensor, n_rot: Optional[int] = None):
        """Rotates a sequence on a regular hexagonal lattice.

        Args:
            seq: sequence of shape (frames, dims, hexals).
        Returns:
            tensor: rotated sequence of the same shape as seq.
        """
        if n_rot is not None:
            self.n_rot = n_rot
        if self.n_rot > 0:
            return self.rotate(seq)
        return seq

    def set_or_sample(self, n_rot=None):
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
        extent (int): extent of the regular hexagonal grid.
        axis (0, 1, 2, 3): flipping axis. 0 corresponds to no flipping.
        p_flip (float): probability of flipping. If None, no flipping is
            performed.

    Attributes:
        same as args and
        permutation_indices (dict): cached indices for flipping.
        rotation_matrices (dict): cached rotation matrices for flipping.
    """

    def __init__(self, extent: int, axis: int = 0, p_flip=0.5):
        self.extent = extent
        # cache indices and matrices cause this must be fast at runtime
        # for augmenting sequences
        self.rotation_matrices = {}
        self.permutation_indices = {}
        # these are the angles of the axes u, v, and z to the cartesian x-axis
        # in default hexlattice convention. for flip matrices the direction
        # does not matter, i.e. flip(angle) = flip(angle +/- 180)
        for n, angle in enumerate([90, 150, 210], 1):
            R2 = flip_matrix(np.radians(angle), three_d=False)
            # for tasks predicting 3d variables
            R3 = flip_matrix(np.radians(angle), three_d=True)
            self.rotation_matrices[n] = [R2, R3]
            self.permutation_indices[n] = flip_permutation_index(extent, n)
        self.axis = axis
        self.p_flip = p_flip
        self.set_or_sample(axis)

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, axis):
        assert axis in [
            0,
            1,
            2,
            3,
        ], f"{axis} is not a valid axis. Must be in [0, 1, 2, 3]."
        self._axis = axis

    def flip(self, seq: torch.Tensor):
        """Flips a sequence on a regular hexagonal lattice.

        Args:
            seq (array or tensor): sequence of shape (frames, dims, hexals).
        Returns:
            array or tensor: flipped sequence of the same shape as seq.
        """
        dims = seq.shape[-2]

        # rearrange the hexals
        seq = seq[..., self.permutation_indices[self.axis]]

        # rotate the vectors in the hexal plane
        if dims > 1:
            seq = self.rotation_matrices[self.axis][dims - 2] @ seq

        return seq

    def __call__(self, seq: torch.Tensor, axis: Optional[int] = None):
        """Flips a sequence on a regular hexagonal lattice.

        Args:
            seq (array or tensor): sequence of shape (frames, dims, hexals).
        Returns:
            array or tensor: flipped sequence of the same shape as seq.
        """
        if axis is not None:
            self.axis = axis
        if self.axis > 0:
            return self.flip(seq=seq)
        return seq

    def set_or_sample(self, axis=None):
        if axis is None:
            axis = (
                np.random.randint(low=1, high=4)
                if self.p_flip and self.p_flip > np.random.rand()
                else 0
            )
        self.axis = axis


class ContrastBrightness(Augmentation):
    """Contrast transformation.

    The transformation is described as:
        pixel = max(0, contrast_factor * (pixel - 0.5) + 0.5
                    + contrast_factor * brightness_factor)

    Args:
        contrast_factor (float): contrast factor.
        brightness_factor (float): brightness factor.
        contrast_std (float): standard deviation of the contrast factor.
        brightness_std (float): standard deviation of the brightness factor.

    Attributes:
        same as args.
    """

    def __init__(
        self,
        contrast_factor=None,
        brightness_factor=None,
        contrast_std=0.2,
        brightness_std=0.1,
    ):
        self.contrast_std = contrast_std
        self.brightness_std = brightness_std
        self.set_or_sample(contrast_factor, brightness_factor)

    def __call__(self, seq: torch.Tensor):
        """Applies the transformation to a sequence.

        Args:
            seq: sequence.
        """
        if self.contrast_factor is not None:
            return (
                self.contrast_factor * (seq - 0.5)
                + 0.5
                + self.contrast_factor * self.brightness_factor
            ).clamp(0)
        return seq

    def set_or_sample(self, contrast_factor=None, brightness_factor=None):
        if contrast_factor is None:
            # behaves like N(1, std) if std is small, slightly biased towards
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
        pixel = pixel + N(0, std)

    It biases the signal to noise ratio: high for light, low for dark pixels.

    Args:
        std (float): standard deviation of the gaussian noise.

    Attributes:
        same as args
    """

    def __init__(self, std=0.08):
        self.std = std

    def __call__(self, seq: torch.Tensor):
        """Applies the transformation to a sequence.

        Args:
            seq: sequence.
        """
        if self.std:
            noise = torch.randn_like(seq) * self.std
            return (seq + noise).clamp(0)
        return seq

    def set_or_sample(self, std=None):
        if std is None:
            return
        self.std = std


class GammaCorrection(Augmentation):
    """Gamma correction.

    The transformation is described as:
        pixel = pixel ** gamma

    gamma > 1 increases the contrast, gamma < 1 decreases the contrast.

    Args:
        gamma (float): gamma value
        std (float): standard deviation of the gamma value
    Attributes:
        same as args
    """

    def __init__(self, gamma=1, std=None):
        self.gamma = gamma
        self.std = std

    def __call__(self, seq: torch.Tensor):
        """Applies the transformation to a sequence.

        Args:
            seq: sequence.
        """
        if self.gamma:
            return seq**self.gamma
        return seq

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "gamma":
            if value < 0:
                raise ValueError("Gamma must be positive.")
        return super().__setattr__(name, value)

    def set_or_sample(self, gamma=None):
        if gamma is None:
            gamma = max(0, np.random.normal(1, self.std)) if self.std else 1.0
        self.gamma = gamma


def rotation_matrix(angle_in_rad, three_d=False):
    if three_d:
        return torch.tensor(
            np.array(
                [
                    [np.cos(angle_in_rad), -np.sin(angle_in_rad), 0],
                    [np.sin(angle_in_rad), np.cos(angle_in_rad), 0],
                    [0, 0, 1],
                ]
            ),
            dtype=torch.float,
        )
    return torch.tensor(
        np.array(
            [
                [np.cos(angle_in_rad), -np.sin(angle_in_rad)],
                [np.sin(angle_in_rad), np.cos(angle_in_rad)],
            ]
        ),
        dtype=torch.float,
    )


def rotation_permutation_index(extent, n_rot):
    u, v = utils.hex_utils.get_hex_coords(extent)
    u_new, v_new = rotate_Nx60(u, v, n_rot)
    return utils.hex_utils.sort_u_then_v_index(u_new, v_new)


def rotate_Nx60(u, v, n):
    """Rotation of hex coordinates by multiples of 60 degrees.

    Ressource: http://devmag.org.za/2013/08/31/geometry-with-hex-coordinates/
    """

    def rotate(u, v):
        """R = [[0, -1], [1, 1]]"""
        return -v, u + v

    for i in range(n % 6):
        u, v = rotate(u, v)

    return u, v


def flip_matrix(angle_in_rad, three_d=False):
    """
    Reference: https://math.stackexchange.com/questions/807031/derive-a-transformation-matrix-that-mirrors-the-image-over-a-line-passing-throug
    """
    if three_d:
        return torch.tensor(
            np.array(
                [
                    [np.cos(2 * angle_in_rad), np.sin(2 * angle_in_rad), 0],
                    [
                        np.sin(2 * angle_in_rad),
                        -np.cos(2 * angle_in_rad),
                        0,
                    ],
                    [0, 0, 1],
                ]
            ),
            dtype=torch.float,
        )
    return torch.tensor(
        np.array(
            [
                [np.cos(2 * angle_in_rad), np.sin(2 * angle_in_rad)],
                [np.sin(2 * angle_in_rad), -np.cos(2 * angle_in_rad)],
            ]
        ),
        dtype=torch.float,
    )


def flip_permutation_index(extent, axis):
    """Get indices used to flip the sequence."""
    u, v = utils.hex_utils.get_hex_coords(extent)
    if axis == 1:
        # flip across v = 0, that is the x axis.
        u_new = u + v
        v_new = -v
    elif axis == 2:
        # flip across u = 0, that is the y axis.
        u_new = -u
        v_new = u + v
    elif axis == 3:
        # flip across u + v = 0, that is the 'z' axis of the hex lattice.
        u_new = -v
        v_new = -u
    else:
        raise ValueError("axis must be in [1, 2, 3].")
    return utils.hex_utils.sort_u_then_v_index(u_new, v_new)
