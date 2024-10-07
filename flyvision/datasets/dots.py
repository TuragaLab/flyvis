from itertools import product
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from datamate import Namespace

import flyvision
from flyvision.utils import hex_utils
from flyvision.utils.hex_utils import Hexal, HexLattice

from .datasets import StimulusDataset
from .rendering.utils import pad, resample

__all__ = ["Dots", "CentralImpulses", "SpatialImpulses"]


class Dots(StimulusDataset):
    """Render flashes aka dots per ommatidia.

    Note:
        Renders directly in receptor space, does not use BoxEye or HexEye as eye-model.

    Attributes:
        augment (bool): Flag for data augmentation.
        dt (Optional[float]): Time step.
        framerate (Optional[float]): Frame rate.
        n_sequences (int): Number of sequences.
        arg_df (Optional[pd.DataFrame]): DataFrame containing stimulus parameters.
    """

    augment: bool = False
    dt: Optional[float] = None
    framerate: Optional[float] = None
    n_sequences: int = 0
    arg_df: Optional[pd.DataFrame] = None

    def __init__(
        self,
        dot_column_radius: int = 0,
        max_extent: int = 15,
        bg_intensity: float = 0.5,
        t_stim: float = 5,
        dt: float = 1 / 200,
        t_impulse: Optional[float] = None,
        n_ommatidia: int = 721,
        t_pre: float = 2.0,
        t_post: float = 0,
        intensity: float = 1,
        mode: str = "sustained",
        device: torch.device = flyvision.device,
    ):
        """Initialize the Dots stimulus dataset.

        Args:
            dot_column_radius: Radius of the dot column.
            max_extent: Maximum extent of the stimulus.
            bg_intensity: Background intensity.
            t_stim: Stimulus duration.
            dt: Time step.
            t_impulse: Impulse duration.
            n_ommatidia: Number of ommatidia.
            t_pre: Pre-stimulus duration.
            t_post: Post-stimulus duration.
            intensity: Stimulus intensity.
            mode: Stimulus mode ('sustained' or 'impulse').
            device: Torch device for computations.

        Raises:
            ValueError: If dot_column_radius is greater than max_extent.
        """
        if dot_column_radius > max_extent:
            raise ValueError("dot_column_radius must be smaller than max_extent")
        self.config = Namespace(
            dot_column_radius=dot_column_radius,
            max_extent=max_extent,
            bg_intensity=bg_intensity,
            t_stim=t_stim,
            dt=dt,
            n_ommatidia=n_ommatidia,
            t_pre=t_pre,
            t_post=t_post,
            intensity=intensity,
            mode=mode,
            t_impulse=t_impulse,
        )

        self.t_stim = t_stim
        self._t_pre = t_pre
        self._t_post = t_post

        self.n_ommatidia = n_ommatidia
        self.offsets = np.arange(self.n_ommatidia)

        u, v = hex_utils.get_hex_coords(hex_utils.get_hextent(n_ommatidia))
        extent_condition = (
            (-max_extent <= u)
            & (u <= max_extent)
            & (-max_extent <= v)
            & (v <= max_extent)
            & (-max_extent <= u + v)
            & (u + v <= max_extent)
        )
        self.u = u[extent_condition]
        self.v = v[extent_condition]
        # self.offsets = self.offsets[extent_condition]
        self.extent_condition = extent_condition

        # to have multi column dots at every location, construct coordinate_indices
        # for each central column
        coordinate_indices = []
        for u, v in zip(self.u, self.v):
            ring = HexLattice.filled_circle(
                radius=dot_column_radius, center=Hexal(u, v, 0), as_lattice=True
            )
            # mask = np.array([~np.isnan(h.value) for h in h1])
            coordinate_indices.append(self.offsets[ring.where(1)])

        self.max_extent = max_extent
        self.bg_intensity = bg_intensity

        self.intensities = [2 * bg_intensity - intensity, intensity]
        self.device = device
        self.mode = mode

        self.params = [
            (*p[0], p[-1])
            for p in list(
                product(
                    zip(self.u, self.v, self.offsets, coordinate_indices),
                    self.intensities,
                )
            )
        ]

        self.arg_df = pd.DataFrame(
            self.params,
            columns=["u", "v", "offset", "coordinate_index", "intensity"],
        )

        self.dt = dt
        self.t_impulse = t_impulse or self.dt

    def _params(self, key: int) -> np.ndarray:
        """Get parameters for a specific key.

        Args:
            key: Index of the parameters to retrieve.

        Returns:
            Array of parameters for the given key.
        """
        return self.arg_df.iloc[key].values

    def get_item(self, key: int) -> torch.Tensor:
        """Get a stimulus item for a specific key.

        Args:
            key: Index of the item to retrieve.

        Returns:
            Tensor representing the stimulus sequence.
        """
        # create maps with background value
        _dot = (
            torch.ones(self.n_ommatidia, device=self.device)[None, None]
            * self.bg_intensity
        )
        # fill at the ommatitdium at offset with intensity
        _, _, _, coordinate_index, intensity = self._params(key)
        _dot[:, :, coordinate_index] = torch.tensor(intensity, device=self.device).float()

        # repeat for stustained stimulus
        if self.mode == "sustained":
            sequence = resample(_dot, self.t_stim, self.dt, dim=1, device=self.device)

        elif self.mode == "impulse":
            # pad remaining stimulus duration i.e. self.t_stim - self.dt with
            # background intensity
            if self.t_impulse == self.dt:
                sequence = pad(
                    _dot,
                    self.t_stim,
                    self.dt,
                    mode="end",
                    fill=self.bg_intensity,
                )
            # first resample for t_impulse/dt then pad remaining stimulus
            # duration, i.e. t_stim - t_impulse with background intensity
            else:
                sequence = resample(
                    _dot, self.t_impulse, self.dt, dim=1, device=self.device
                )
                sequence = pad(
                    sequence,
                    self.t_stim,
                    self.dt,
                    mode="end",
                    fill=self.bg_intensity,
                )

        # pad with pre stimulus background
        sequence = pad(
            sequence,
            self.t_stim + self.t_pre,
            self.dt,
            mode="start",
            fill=self.bg_intensity,
        )
        # pad with post stimulus background
        sequence = pad(
            sequence,
            self.t_stim + self.t_pre + self.t_post,
            self.dt,
            mode="end",
            fill=self.bg_intensity,
        )
        return sequence.squeeze()

    def __len__(self) -> int:
        """Get the number of items in the dataset.

        Returns:
            Number of items in the dataset.
        """
        return len(self.arg_df)

    def get_sequence_id_from_arguments(self, u: float, v: float, intensity: float) -> int:
        """Get sequence ID from given arguments.

        Args:
            u: U-coordinate.
            v: V-coordinate.
            intensity: Stimulus intensity.

        Returns:
            Sequence ID.
        """
        return self._get_sequence_id_from_arguments(locals())

    @property
    def t_pre(self) -> float:
        """Get pre-stimulus duration."""
        return self._t_pre

    @property
    def t_post(self) -> float:
        """Get post-stimulus duration."""
        return self._t_post


class CentralImpulses(StimulusDataset):
    """Flashes at the center of the visual field for temporal receptive field mapping.

    Attributes:
        arg_df (Optional[pd.DataFrame]): DataFrame containing stimulus parameters.
        augment (bool): Flag for data augmentation.
        dt (Optional[float]): Time step.
        framerate (Optional[float]): Frame rate.
        n_sequences (Optional[int]): Number of sequences.
    """

    arg_df: Optional[pd.DataFrame] = None
    augment: bool = False
    dt: Optional[float] = None
    framerate: Optional[float] = None
    n_sequences: Optional[int] = None

    def __init__(
        self,
        impulse_durations: List[float] = [5e-3, 20e-3, 50e-3, 100e-3, 200e-3, 300e-3],
        dot_column_radius: int = 0,
        bg_intensity: float = 0.5,
        t_stim: float = 5,
        dt: float = 0.005,
        n_ommatidia: int = 721,
        t_pre: float = 2.0,
        t_post: float = 0,
        intensity: float = 1,
        mode: str = "impulse",
        device: torch.device = flyvision.device,
    ):
        """Initialize the CentralImpulses dataset.

        Args:
            impulse_durations: List of impulse durations.
            dot_column_radius: Radius of the dot column.
            bg_intensity: Background intensity.
            t_stim: Stimulus duration.
            dt: Time step.
            n_ommatidia: Number of ommatidia.
            t_pre: Pre-stimulus duration.
            t_post: Post-stimulus duration.
            intensity: Stimulus intensity.
            mode: Stimulus mode.
            device: Torch device for computations.
        """
        self.dots = Dots(
            dot_column_radius=dot_column_radius,
            max_extent=dot_column_radius,
            bg_intensity=bg_intensity,
            t_stim=t_stim,
            dt=dt,
            n_ommatidia=n_ommatidia,
            t_pre=t_pre,
            t_post=t_post,
            intensity=intensity,
            mode=mode,
            device=device,
        )
        self.impulse_durations = impulse_durations
        self.config = self.dots.config
        self.config.update(impulse_durations=impulse_durations)
        self.params = [
            (*p[0], p[1])
            for p in product(self.dots.arg_df.values.tolist(), impulse_durations)
        ]
        self.arg_df = pd.DataFrame(
            self.params,
            columns=[
                "u",
                "v",
                "offset",
                "coordinate_index",
                "intensity",
                "t_impulse",
            ],
        )
        self.dt = dt

    def __len__(self) -> int:
        """Get the number of items in the dataset.

        Returns:
            Number of items in the dataset.
        """
        return len(self.arg_df)

    def _params(self, key: int) -> np.ndarray:
        """Get parameters for a specific key.

        Args:
            key: Index of the parameters to retrieve.

        Returns:
            Array of parameters for the given key.
        """
        return self.arg_df.iloc[key].values

    def get_item(self, key: int) -> torch.Tensor:
        """Get a stimulus item for a specific key.

        Args:
            key: Index of the item to retrieve.

        Returns:
            Tensor representing the stimulus sequence.
        """
        u, v, offset, coordinate_index, intensity, t_impulse = self._params(key)
        self.dots.t_impulse = t_impulse
        return self.dots[self.dots.get_sequence_id_from_arguments(u, v, intensity)]

    get_sequence_id_from_arguments = StimulusDataset._get_sequence_id_from_arguments

    @property
    def t_pre(self) -> float:
        """Get pre-stimulus duration."""
        return self.dots.t_pre

    @property
    def t_post(self) -> float:
        """Get post-stimulus duration."""
        return self.dots.t_post

    def __repr__(self) -> str:
        """Get string representation of the dataset."""
        return repr(self.arg_df)


class SpatialImpulses(StimulusDataset):
    """Spatial flashes for spatial receptive field mapping.

    Attributes:
        arg_df (Optional[pd.DataFrame]): DataFrame containing stimulus parameters.
        augment (bool): Flag for data augmentation.
        dt (Optional[float]): Time step.
        framerate (Optional[float]): Frame rate.
        n_sequences (Optional[int]): Number of sequences.
    """

    arg_df: Optional[pd.DataFrame] = None
    augment: bool = False
    dt: Optional[float] = None
    framerate: Optional[float] = None
    n_sequences: Optional[int] = None

    def __init__(
        self,
        impulse_durations: List[float] = [5e-3, 20e-3],
        max_extent: int = 4,
        dot_column_radius: int = 0,
        bg_intensity: float = 0.5,
        t_stim: float = 5,
        dt: float = 0.005,
        n_ommatidia: int = 721,
        t_pre: float = 2.0,
        t_post: float = 0,
        intensity: float = 1,
        mode: str = "impulse",
        device: torch.device = flyvision.device,
    ):
        """Initialize the SpatialImpulses dataset.

        Args:
            impulse_durations: List of impulse durations.
            max_extent: Maximum extent of the stimulus.
            dot_column_radius: Radius of the dot column.
            bg_intensity: Background intensity.
            t_stim: Stimulus duration.
            dt: Time step.
            n_ommatidia: Number of ommatidia.
            t_pre: Pre-stimulus duration.
            t_post: Post-stimulus duration.
            intensity: Stimulus intensity.
            mode: Stimulus mode.
            device: Torch device for computations.
        """
        self.dots = Dots(
            dot_column_radius=dot_column_radius,
            max_extent=max_extent,
            bg_intensity=bg_intensity,
            t_stim=t_stim,
            dt=dt,
            n_ommatidia=n_ommatidia,
            t_pre=t_pre,
            t_post=t_post,
            intensity=intensity,
            mode=mode,
            device=device,
        )
        self.dt = dt
        self.impulse_durations = impulse_durations

        self.config = self.dots.config
        self.config.update(impulse_durations=impulse_durations)

        self.params = [
            (*p[0], p[1])
            for p in product(self.dots.arg_df.values.tolist(), impulse_durations)
        ]
        self.arg_df = pd.DataFrame(
            self.params,
            columns=[
                "u",
                "v",
                "offset",
                "coordinate_index",
                "intensity",
                "t_impulse",
            ],
        )

    def __len__(self) -> int:
        """Get the number of items in the dataset.

        Returns:
            Number of items in the dataset.
        """
        return len(self.arg_df)

    def _params(self, key: int) -> np.ndarray:
        """Get parameters for a specific key.

        Args:
            key: Index of the parameters to retrieve.

        Returns:
            Array of parameters for the given key.
        """
        return self.arg_df.iloc[key].values

    def get_item(self, key: int) -> torch.Tensor:
        """Get a stimulus item for a specific key.

        Args:
            key: Index of the item to retrieve.

        Returns:
            Tensor representing the stimulus sequence.
        """
        u, v, offset, coordinate_index, intensity, t_impulse = self._params(key)
        self.dots.t_impulse = t_impulse
        return self.dots[self.dots.get_sequence_id_from_arguments(u, v, intensity)]

    get_sequence_id_from_arguments = StimulusDataset._get_sequence_id_from_arguments

    @property
    def t_pre(self) -> float:
        """Get pre-stimulus duration."""
        return self.dots.t_pre

    @property
    def t_post(self) -> float:
        """Get post-stimulus duration."""
        return self.dots.t_post

    def __repr__(self) -> str:
        """Get string representation of the dataset."""
        return repr(self.arg_df)
