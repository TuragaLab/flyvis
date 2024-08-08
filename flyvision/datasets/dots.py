from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dvs import exp_path_context
from dvs.datasets.base import StimulusDataset
from dvs.datasets.wraps import init_network_wrap
from dvs.plots import plt_utils
from dvs.stim_gen.eye import pad, resample
from dvs.utils import activity_utils, hex_utils
from dvs.utils.datawrapper import Namespace
from dvs.utils.hex_utils import HexLattice, Hexal
from dvs.utils.tensor_utils import scatter_mean
from matplotlib import ticker


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
        subwrap="dots",
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
            subwrap=subwrap,
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
            ring = HexLattice.filled_ring(
                radius=dot_column_radius, center=Hexal(u, v, 0), as_lattice=True
            )
            # mask = np.array([~np.isnan(h.value) for h in h1])
            coordinate_indices.append(self.offsets[ring.where(1)])

        self.vline = hex_utils.hexline(-15, 0, 15, 0)
        hline1 = hex_utils.hexline(7, -15, -8, 15, eps=1e-6)
        hline2 = hex_utils.hexline(8, -15, -7, 15, eps=-1e-6)
        hline = hline1 | hline2
        self.hline = hline

        self.max_extent = max_extent
        self.bg_intensity = bg_intensity

        self.intensities = [2 * bg_intensity - intensity, intensity]
        self.device = device
        self.mode = mode
        self.subwrap = subwrap

        #         self.params = list(product(self.u, self.v, self.offsets, self.intensities))

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

        #         self._build()
        self.dt = dt
        self.t_impulse = t_impulse or self.dt

    def _init_tnn(self, tnn):
        self.tnn = tnn
        with exp_path_context():
            if isinstance(tnn, (str, Path)):
                self.tnn, _ = init_network_wrap(tnn, None, None)
            self.central_activity = activity_utils.CentralActivity(
                self.tnn[self.subwrap].network_states.nodes.activity_central[:],
                self.tnn.ctome,
                keepref=True,
            )
        self.spec = self.tnn[self.subwrap].spec

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
        """
        #TODO: docstring
        """
        mask = self.mask(intensity, offset, u, v)
        return torch.stack([self[i] for i in np.arange(len(mask))[mask]]).cpu().numpy()

    def response(
        self,
        node_type=None,
        intensity=None,
        offset=None,
        u=None,
        v=None,
        pre_stim=False,
    ):
        assert self.tnn
        mask = self.mask(intensity=intensity, offset=offset, u=u, v=v)

        time_slice = slice(None)
        if not pre_stim:
            time_slice = slice(int(self.t_pre / self.dt), None)
        if node_type is not None:
            return self.central_activity[node_type][mask][:, time_slice].squeeze()
        else:
            return self.central_activity[:][mask][:, time_slice].squeeze()

    def receptive_field(self, node_type, intensity, time_window=None):
        response = self.response(node_type=node_type, intensity=intensity)
        # n_frames, #n_hexals
        rf = np.ones([response.shape[1], len(self.extent_condition)]) * np.nan
        rf[:, self.extent_condition] = response.T

        if time_window is not None:
            rf = rf[time_window[0] // self.dt : (time_window[1] + self.dt) // self.dt]

        return rf

    def receptive_field_vertical(self, node_type, intensity):
        rf = self.receptive_field(node_type, intensity)
        return rf[:, self.vline]

    def receptive_field_horizontal(self, node_type, intensity):
        rf = torch.tensor(self.receptive_field(node_type, intensity), device="cpu")
        u, v = hex_utils.get_hex_coords(15)
        rf = (
            scatter_mean(
                rf[:, self.hline],
                torch.tensor(v[self.hline] - v.min(), device="cpu").long(),
            )
            .cpu()
            .numpy()
        )
        return rf

    @property
    def t_pre(self):
        return self._t_pre

    @property
    def t_post(self):
        return self._t_post

    def plot(
        self,
        node_type,
        intensity,
        orientation,
        figsize=[5, 4],
        rf=None,
        subtract_baseline=True,
        xlim=(-3, 0),
        ylim=(-6, 6),
        xticks=6,
        fontsize=10,
        cbar=False,
        fig=None,
        ax=None,
        cmap=plt.cm.coolwarm,
        as_delay=True,
    ):
        # plt.style.use('ggplot')
        plt.rcParams["axes.axisbelow"] = False

        if orientation == "vertical":
            if rf is None:
                rf = self.receptive_field_vertical(node_type, intensity)
            ylabel = "elevation"
        elif orientation == "horizontal":
            if rf is None:
                rf = self.receptive_field_horizontal(node_type, intensity)
            ylabel = "azimuth"

        if subtract_baseline:
            rf -= rf[0, :][None]

        n_frames, n_columns = rf.shape

        fig, ax = plt_utils.init_plot(figsize=figsize, fontsize=fontsize, fig=fig, ax=ax)
        crange = np.abs(rf).max()
        norm = plt_utils.get_norm(vmin=-crange, vmax=crange, midpoint=0)

        time = np.arange(-n_frames + 1, 1) * self.dt * 1000
        angles = np.arange(-n_columns // 2 + 1, n_columns // 2 + 1) * 5.8

        if as_delay:
            ax.pcolormesh(
                time,
                angles,
                rf.T[:, ::-1],
                cmap=cmap,
                norm=norm,
                zorder=1,
                shading="nearest",
            )
            ax.set_xlabel("delay (ms)", fontsize=fontsize)
        else:
            ax.pcolormesh(
                -1 * time[::-1],
                angles,
                rf.T,
                cmap=cmap,
                norm=norm,
                zorder=1,
                shading="nearest",
            )
            ax.set_xlabel("time (ms)", fontsize=fontsize)

        ax.locator_params(nbins=5)

        ax.set_yticks(angles)

        angles = angles.astype(str)

        ax.set_yticklabels([angle + "\N{DEGREE SIGN}" for angle in angles])
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.yaxis.grid(True)
        ax.set_ylim(5.8 * ylim[0], 5.8 * ylim[1])

        # ax.set_xticks(np.arange(self.t_stim + 1) * 1 / self.dt)
        # ax.set_xticklabels(np.arange(-self.t_stim, 1))
        ax.set_xlim(*xlim)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(xticks))
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        # ax.set_xticklabels([label_format.format(x) for x in ticks_loc])

        if cbar:
            cbar = plt_utils.add_colorbar(
                fig, ax, cmap=plt.cm.seismic, norm=norm, fontsize=fontsize
            )
            cbar.ax.locator_params(nbins=5)

        ax.set_title(node_type, fontsize=fontsize)

        return fig, ax

    def plot_hex(
        self,
        node_type,
        intensity,
        figsize=[5, 4],
        rf=None,
        xlim=(-3, 0),
        xticks=6,
        fontsize=10,
    ):
        rf = self.receptive_field(node_type, intensity)


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
        subwrap="centralimpulses",
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
            subwrap=subwrap,
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
        subwrap="spatialimpulses",
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
            subwrap=subwrap,
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
