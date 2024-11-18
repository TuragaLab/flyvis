from typing import Literal, Tuple

import numpy as np
import torch

from flyvis.utils import hex_utils


def rotation_matrix(angle_in_rad: float, three_d: bool = False) -> torch.Tensor:
    """Generate a rotation matrix.

    Args:
        angle_in_rad: Rotation angle in radians.
        three_d: If True, generate a 3D rotation matrix.

    Returns:
        Rotation matrix as a torch.Tensor.
    """
    if three_d:
        return torch.tensor(
            np.array([
                [np.cos(angle_in_rad), -np.sin(angle_in_rad), 0],
                [np.sin(angle_in_rad), np.cos(angle_in_rad), 0],
                [0, 0, 1],
            ]),
            dtype=torch.float,
        )
    return torch.tensor(
        np.array([
            [np.cos(angle_in_rad), -np.sin(angle_in_rad)],
            [np.sin(angle_in_rad), np.cos(angle_in_rad)],
        ]),
        dtype=torch.float,
    )


def rotation_permutation_index(extent: int, n_rot: int) -> torch.Tensor:
    """Calculate rotation permutation indices for hex coordinates.

    Args:
        extent: Extent of the regular hexagonal grid.
        n_rot: Number of 60-degree rotations.

    Returns:
        Permutation indices as a torch.Tensor.
    """
    u, v = hex_utils.get_hex_coords(extent)
    u_new, v_new = rotate_Nx60(u, v, n_rot)
    return hex_utils.sort_u_then_v_index(u_new, v_new)


def rotate_Nx60(u: np.ndarray, v: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate hex coordinates by multiples of 60 degrees.

    Args:
        u: U coordinates of hex grid.
        v: V coordinates of hex grid.
        n: Number of 60-degree rotations.

    Returns:
        Tuple of rotated U and V coordinates.

    Note:
        Resource: http://devmag.org.za/2013/08/31/geometry-with-hex-coordinates/
    """

    def rotate(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate hex coordinates by 60 degrees.

        Rotation matrix R = [[0, -1], [1, 1]]
        """
        return -v, u + v

    for _ in range(n % 6):
        u, v = rotate(u, v)

    return u, v


def flip_matrix(angle_in_rad: float, three_d: bool = False) -> torch.Tensor:
    """Generate a flip matrix for mirroring over a line.

    Args:
        angle_in_rad: Angle of the flip axis in radians.
        three_d: If True, generate a 3D flip matrix.

    Returns:
        Flip matrix as a torch.Tensor.

    Note:
        Reference: https://math.stackexchange.com/questions/807031/
    """
    if three_d:
        return torch.tensor(
            np.array([
                [np.cos(2 * angle_in_rad), np.sin(2 * angle_in_rad), 0],
                [np.sin(2 * angle_in_rad), -np.cos(2 * angle_in_rad), 0],
                [0, 0, 1],
            ]),
            dtype=torch.float,
        )
    return torch.tensor(
        np.array([
            [np.cos(2 * angle_in_rad), np.sin(2 * angle_in_rad)],
            [np.sin(2 * angle_in_rad), -np.cos(2 * angle_in_rad)],
        ]),
        dtype=torch.float,
    )


def flip_permutation_index(extent: int, axis: Literal[1, 2, 3]) -> torch.Tensor:
    """Get indices used to flip the sequence.

    Args:
        extent: Extent of the regular hexagonal grid.
        axis: Axis to flip across (1, 2, or 3).

    Returns:
        Permutation indices as a torch.Tensor.

    Raises:
        ValueError: If axis is not in [1, 2, 3].
    """
    u, v = hex_utils.get_hex_coords(extent)
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
    return hex_utils.sort_u_then_v_index(u_new, v_new)
