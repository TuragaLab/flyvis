from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from datamate import Namespace

from flyvision.datasets.datasets import StimulusDataset
from flyvision.rendering.utils import pad, resample
from flyvision.utils import hex_utils
from flyvision.utils.hex_utils import HexLattice, Hexal


class Dots(StimulusDataset):
    augment = False
    dt = None
    framerate = None
    n_sequences = 0
    arg_df = None

    def __init__(
        self,
        dot_column_radius=0,
        max_extent=15,
        bg_intensity=0.5,
        t_stim=5,
        dt=1 / 200,
        t_impulse=None,
        n_ommatidia=721,
        t_pre=2.0,
        t_post=0,
        intensity=1,
        mode="sustained",
        device="cuda",
    ):
        if dot_column_radius > max_extent:
            raise ValueError("dot_column_radius must be smaller than max_extent")
        self.spec = Namespace(
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

    def _params(self, key):
        return self.arg_df.iloc[key].values

    def get_item(self, key):
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

    def __len__(self):
        return len(self.arg_df)

    def mask(self, intensity=None, offset=None, u=None, v=None):
        values = self.arg_df.values

        def iterparam(param, name, axis, and_condition):
            condition = np.zeros(len(values)).astype(bool)
            if isinstance(param, Iterable):
                for p in param:
                    _new = values.take(axis, axis=1) == p
                    assert any(_new), f"{name} {p} not in dataset."
                    condition = np.logical_or(condition, _new)
            else:
                _new = values.take(axis, axis=1) == param
                assert any(_new), f"{name} {param} not in dataset."
                condition = np.logical_or(condition, _new)
            return condition & and_condition

        condition = np.ones(len(values)).astype(bool)
        if offset is not None:
            condition = iterparam(offset, "offset", 2, condition)
        if intensity is not None:
            condition = iterparam(intensity, "intensity", 4, condition)
        if u is not None:
            condition = iterparam(u, "u", 0, condition)
        if v is not None:
            condition = iterparam(v, "v", 1, condition)
        return condition

    def get_sequence_id_from_arguments(self, u, v, intensity):
        return self._get_sequence_id_from_arguments(locals())

    def stimulus(self, intensity=None, offset=None, u=None, v=None):
        mask = self.mask(intensity, offset, u, v)
        return torch.stack([self[i] for i in np.arange(len(mask))[mask]]).cpu().numpy()

    @property
    def t_pre(self):
        return self._t_pre

    @property
    def t_post(self):
        return self._t_post


class CentralImpulses(StimulusDataset):
    arg_df = None
    augment = False
    dt = None
    framerate = None
    n_sequences = None

    def __init__(
        self,
        impulse_durations=[5e-3, 20e-3, 50e-3, 100e-3, 200e-3, 300e-3],
        dot_column_radius=0,
        bg_intensity=0.5,
        t_stim=5,
        dt=0.005,
        n_ommatidia=721,
        t_pre=2.0,
        t_post=0,
        intensity=1,
        mode="impulse",
        device="cuda",
    ):
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

    def __len__(self):
        return len(self.arg_df)

    def _params(self, key):
        return self.arg_df.iloc[key].values

    def get_item(self, key):
        u, v, offset, coordinate_index, intensity, t_impulse = self._params(key)
        self.dots.t_impulse = t_impulse
        return self.dots[self.dots.get_sequence_id_from_arguments(u, v, intensity)]

    get_sequence_id_from_arguments = StimulusDataset._get_sequence_id_from_arguments

    @property
    def t_pre(self):
        return self.dots.t_pre

    @property
    def t_post(self):
        return self.dots.t_post

    def __repr__(self):
        return repr(self.arg_df)


class SpatialImpulses(StimulusDataset):
    arg_df = None
    augment = False
    dt = None
    framerate = None
    n_sequences = None

    def __init__(
        self,
        impulse_durations=[5e-3, 20e-3],
        max_extent=4,
        dot_column_radius=0,
        bg_intensity=0.5,
        t_stim=5,
        dt=0.005,
        n_ommatidia=721,
        t_pre=2.0,
        t_post=0,
        intensity=1,
        mode="impulse",
        device="cuda",
    ):
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

    def __len__(self):
        return len(self.arg_df)

    def _params(self, key):
        return self.arg_df.iloc[key].values

    def get_item(self, key):
        u, v, offset, coordinate_index, intensity, t_impulse = self._params(key)
        self.dots.t_impulse = t_impulse
        return self.dots[self.dots.get_sequence_id_from_arguments(u, v, intensity)]

    get_sequence_id_from_arguments = StimulusDataset._get_sequence_id_from_arguments

    @property
    def t_pre(self):
        return self.dots.t_pre

    @property
    def t_post(self):
        return self.dots.t_post

    def __repr__(self):
        return repr(self.arg_df)
