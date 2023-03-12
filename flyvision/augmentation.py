"""Transformations and augmentations on hex-lattices."""
import numpy as np
import torch

from flyvision.utils.hex_utils import get_hex_coords, sort_u_then_v_index

__all__ = [
    "HexRotate",
    "HexFlip",
    "Jitter",
    "Noise",
    "Invert",
    "rotate_Nx60",
]


class HexRotate:
    """Rotate a sequence of regular hex-lattices by Nx60 degree.

    Args:
        radius (int): radius of the regular hexagonal grid.
        n_rot (int): number of 60 degree rotations. 0-5.
    """

    def __init__(self, radius, n_rot):
        self.radius = radius
        self._cached_matrices = {}
        self._cached_indices = {}
        for n in range(6):
            R2 = self._get_rot_mat(n * 60 * np.pi / 180, three_d=False)
            R3 = self._get_rot_mat(n * 60 * np.pi / 180, three_d=True)
            self._cached_matrices[n] = {"2d": R2, "3d": R3}
            self._cached_indices[n] = self._get_rot_index(radius, n)
        self.n_rot = n_rot % 6

    @property
    def n_rot(self):
        return self._n_rot

    @n_rot.setter
    def n_rot(self, n_rot):
        self._n_rot = n_rot % 6
        if self._n_rot > 0:
            self.index = self._cached_indices[self._n_rot]
            self.rotation_matrix = self._cached_matrices[self._n_rot]

    def _get_rot_mat(self, angle_in_rad, three_d=False):
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

    def _get_rot_index(self, radius, n_rot):
        u, v = get_hex_coords(radius)
        u_new, v_new = rotate_Nx60(u, v, n_rot)
        return sort_u_then_v_index(u_new, v_new)

    def rotate(self, seq, _type):
        """Rotates a sequence on a regular hexagonal lattice.

        Args:
            seq (array or tensor): sequence of shape (frames, dims, hexals).
            _type ([type]): task type "lum", "depth", "flow", or including
                "ego". If lum or depth, the hexals are simply rearranged.
                In the flow and egomotion scenario, arrows are rotated
                accordingly.

        Returns:
            array or tensor: rotated sequence of the same shape as seq.
        """
        if _type in ["lum", "depth"]:
            # Rearrange the hexals and return.
            seq = seq[..., self.index]
            return seq
        elif _type == "flow":
            # Rearrange the hexals before rotating the flow vectors.
            seq = seq[..., self.index]
        dims = "2d"
        if "ego" in _type:
            # Flow vectors are two-dimensional - while egomotion has three
            # dimensions.
            dims = "3d"
        if isinstance(seq, torch.Tensor):
            seq = self.rotation_matrix[dims] @ seq
        else:
            seq = self.rotation_matrix[dims].cpu().numpy() @ seq
        return seq

    def __call__(self, seq, _type=False):
        if self.n_rot > 0:
            return self.rotate(seq, _type)
        return seq


class HexFlip:
    """Flip a sequence of regular hex-lattices across one of three hex-axes.

    Args:
        radius (int): radius of the regular hexagonal grid.
        axis (None, 0, 1, or 2): flipping axis.
    """

    _type = ""

    def __init__(self, radius, axis):
        self.radius = radius
        self._cached_matrices = {}
        self._cached_indices = {}

        # These are the angles of the axes u, v, and z to the cartesian x-axis
        # in tschopp hexpixel convention. For flip matrices the orientation
        # does not matter, i.e. flip(angle) = flip(angle +/- 180)
        for n, angle in enumerate([90, 150, 210]):
            R2 = self._get_rot_mat(np.radians(angle), three_d=False)
            R3 = self._get_rot_mat(np.radians(angle), three_d=True)
            self._cached_matrices[n] = {"2d": R2, "3d": R3}
            self._cached_indices[n] = self._get_flip_index(radius, n)
        self.axis = axis

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, axis):
        assert axis in [None, 0, 1, 2]
        self._axis = axis
        if axis is not None:
            self.index = self._cached_indices[self._axis]
            self.rotation_matrix = self._cached_matrices[self._axis]

    def _get_rot_mat(self, angle_in_rad, three_d=False):
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

    def _get_flip_index(self, radius, axis):
        """Get indices used to flip the sequence."""
        u, v = get_hex_coords(radius)
        if axis == 0:
            # flip across v = 0, that is the x axis.
            u_new = u + v
            v_new = -v
        elif axis == 1:
            # flip across u = 0, that is the y axis.
            u_new = -u
            v_new = u + v
        elif axis == 2:
            # flip across u + v = 0, that is the 'z' axis of the hex lattice.
            u_new = -v
            v_new = -u
        return sort_u_then_v_index(u_new, v_new)

    def flip(self, seq, _type):
        """Flips a sequence on a regular hexagonal lattice.

        Args:
            seq (array or tensor): sequence of shape (frames, dims, hexals).
            _type ([type]): task type "lum", "depth", "flow", or including
                "ego". If lum or depth, the hexals are simply rearranged.
                In the flow and egomotion scenario, arrows are rotated
                accordingly.

        Returns:
            array or tensor: flipped sequence of the same shape as seq.
        """
        if _type in ["lum", "depth"]:
            seq = seq[..., self.index]
            return seq
        elif _type == "flow":
            seq = seq[..., self.index]
        dims = "2d"
        if "ego" in _type:
            dims = "3d"
        if isinstance(seq, torch.Tensor):
            seq = self.rotation_matrix[dims] @ seq
        else:
            seq = self.rotation_matrix[dims].cpu().numpy() @ seq
        return seq

    def __call__(self, seq, _type=False):
        if self.axis is not None:
            return self.flip(seq=seq, _type=_type)
        return seq


class DepthAugment:
    """Contrast and brightness transformation based on depth.


    The transformation is described as:
        pixel = (1/depth**2) / max(1/depth**2) * (pixel - 0.5) + 0.5

    Thus, the contrast decays with a square law in distance.
    """

    def __init__(self, enable=False):
        self.enable = enable

    def __call__(self, seq, depth=None):
        if self.enable and depth is not None:
            contrast = 1 / depth**2
            contrast /= contrast.max(dim=-1, keepdim=True).values
            return contrast * (seq - 0.5) + 0.5
        return seq
        # if self.contrast_factor == 0:
        #     return seq
        # return (
        #     self.contrast_factor * (seq - 0.5)
        #     + 0.5
        #     + self.contrast_factor * self.brightness_factor
        # ).clamp(0)


class Jitter:
    """Contrast transformation.


    The transformation is described as:
        pixel = contrast_factor * (pixel - 0.5) + 0.5
                + contrast_factor * max(0, brightness_factor),

    where the contrast_factor and brightness_factor are thought to be randomly
    sampled by
        contrast_factor ~ exp(N(0, sigma))
        brightness_factor ~ N(0, sigma).

    We multiply the brightness factor with the contrast factor to correlate both.
    """

    def __init__(self, contrast_factor, brightness_factor):
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor

    def __call__(self, seq):
        if self.contrast_factor:
            return (
                self.contrast_factor * (seq - 0.5)
                + 0.5
                + self.contrast_factor * self.brightness_factor
            ).clamp(0)
        return seq


class Noise:
    """Pixelwise gaussian noise with mean 0 and given std.

    Biases the signal to noise ratio.
    """

    def __init__(self, std):
        self.std = std

    def __call__(self, seq):
        # breakpoint()
        if self.std:
            noise = torch.randn_like(seq) * self.std
            return (seq + noise).clamp(0)
        return seq


class Invert:
    """Inverts the gray value sequence."""

    def __init__(self, invert):
        self.invert = invert

    def __call__(self, seq: torch.Tensor):
        if self.invert:
            return (1 - seq).clip(0)
        return seq


def rotate_Nx60(u, v, n):
    """Rotation in hex coordinates.

    Ressource: http://devmag.org.za/2013/08/31/geometry-with-hex-coordinates/
    """

    def rotate(u, v):
        """R = [[0, -1], [1, 1]]"""
        return -v, u + v

    for i in range(n % 6):
        u, v = rotate(u, v)

    return u, v
