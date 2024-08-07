import logging
from itertools import product
from numbers import Number
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datamate import Directory, Namespace, root
from matplotlib.patches import RegularPolygon
from tqdm.auto import tqdm

from flyvision import renderings_dir
from flyvision.datasets.datasets import StimulusDataset
from flyvision.plots.plots import quick_hex_scatter
from flyvision.plots.plt_utils import init_plot
from flyvision.rendering import HexEye
from flyvision.rendering.utils import pad, resample, shuffle

logging = logging.getLogger()


@root(renderings_dir)
class RenderedOffsets(Directory):
    """Rendered offsets for the moving bar stimulus.

    Note: This class is used to precompute the offsets for the moving bar (edge) stimuli
    and store them in a directory. At runtime, the offsets are resampled to allow for
    changing the time step and to save GB of storage.
    """

    class Config(dict):
        offsets: list = list(range(-10, 11))
        angles: list = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        widths: list = [1, 2, 4]
        intensities: list = [0, 1]
        led_width: float = np.radians(2.25)
        height: float = np.radians(2.25) * 9
        n_bars: int = 1
        bg_intensity: float = 0.5
        bar_loc_horizontal: float = np.radians(90)

    def __init__(self, config: Config):
        eye = HexEye(721, 25)

        params = list(product(config.angles, config.widths, config.intensities))

        sequences = {}
        _tqdm = tqdm(total=len(params), desc="building stimuli")
        for angle in config.angles:
            for width in config.widths:
                for intensity in config.intensities:
                    offset_bars = eye.offset_bars(
                        bar_width_rad=width * config.led_width,
                        bar_height_rad=config.height,
                        n_bars=config.n_bars,
                        offsets=np.array(config.offsets) * config.led_width,
                        bar_intensity=intensity,
                        bg_intensity=config.bg_intensity,
                        moving_angle=angle,
                        bar_loc_horizontal=config.bar_loc_horizontal,
                    )
                    sequences[(angle, width, intensity)] = offset_bars
                    _tqdm.update()

        _tqdm.close()
        self.offsets = torch.stack([sequences[p] for p in params]).cpu().numpy()


class MovingBar(StimulusDataset):
    """Moving bar stimulus.

    Args:
        widths: list of int, optional
            Width of the bar in half ommatidia.
        offsets: tuple of int, optional
            Tuple of the first and last offset to the central column in half ommatidia.
        intensities: list of int, optional
            Intensity of the bar.
        speeds: list of float, optional
            Speed of the bar in half ommatidia per second.
        height: int, optional
            Height of the bar in half ommatidia.
        dt: float, optional
            Time step in seconds.
        subdir: str, optional
            Subdirectory where the stimulus is stored.
        device: str, optional
            Device to store the stimulus.
        bar_loc_horizontal: float, optional
            Horizontal location of the bar in radians from left to right of image plane.
            Radians(90) is the center.
        post_pad_mode: str, optional
            Padding mode after the stimulus. One of 'continue', 'value', 'reflect'.
            If 'value' the padding is filled with the value of `bg_intensity`.
        t_pre: float, optional
            Time before the stimulus in seconds.
        t_post: float, optional
            Time after the stimulus in seconds.
        build_stim_on_init: bool, optional
            Build the stimulus on initialization.
        shuffle_offsets: bool, optional
            Shuffle the offsets to remove spatio-temporal correlation.
        seed: int, optional
            Seed for the random state.
        angles: list of int, optional
            List of angles in degrees.
    """

    augment = False
    n_sequences = 0
    framerate = None

    arg_df: pd.DataFrame = None

    def __init__(
        self,
        widths=[1, 2, 4],  # in 1 * radians(2.25) led size
        offsets=(-10, 11),  # in 1 * radians(2.25) led size
        intensities=[0, 1],
        speeds=[2.4, 4.8, 9.7, 13, 19, 25],  # in 1 * radians(5.8) / s
        height=9,  # in 1 * radians(2.25) led size
        dt=1 / 200,
        subdir="movingbar",
        device="cuda",
        bar_loc_horizontal=np.radians(90),
        post_pad_mode="value",
        t_pre=1.0,
        t_post=1.0,
        build_stim_on_init=True,  # can speed up things downstream if only responses
        # are needed
        shuffle_offsets=False,  # shuffle offsets to remove spatio-temporal correlation
        # -- can be used as stimulus to compute a baseline of motion selectivity
        seed=0,  # only for shuffle_offsets
        angles=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    ):
        super().__init__()
        # HexEye parameter
        self.omm_width = np.radians(5.8)  # radians(5.8 degree)

        # Monitor parameter
        self.led_width = np.radians(2.25)  # Gruntman et al. 2018

        _locals = locals()
        self.config = Namespace({
            arg: _locals[arg]
            for arg in [
                "widths",
                "offsets",
                "intensities",
                "speeds",
                "height",
                "bar_loc_horizontal",
                "shuffle_offsets",
                "post_pad_mode",
                "t_pre",
                "t_post",
                "dt",
                "angles",
            ]
        })

        # Stim parameter
        self.angles = np.array(angles)
        self.widths = np.array(widths)  # half ommatidia
        if len(offsets) == 2:
            self.offsets = np.arange(*offsets)  # half ommatidia
        else:
            assert (
                np.mean(offsets[1:] - offsets[:-1]) == 1
            )  # t_stim assumes spacing of 1 corresponding to 2.25 deg
            self.offsets = offsets
        self.intensities = np.array(intensities)
        self.speeds = np.array(speeds)
        self.bg_intensity = 0.5
        self.n_bars = 1
        self.bar_loc_horizontal = bar_loc_horizontal

        self.t_stim = (len(self.offsets) * self.led_width) / (
            self.speeds * self.omm_width
        )
        self.t_stim_max = np.max(self.t_stim)

        self._speed_to_t_stim = dict(zip(self.speeds, self.t_stim))

        self.height = self.led_width * height

        self.post_pad_mode = post_pad_mode
        self._t_pre = t_pre
        self._t_post = t_post

        params = [
            (*p[:-1], *p[-1])
            for p in list(
                product(
                    self.angles,
                    self.widths,
                    self.intensities,
                    zip(self.t_stim, self.speeds),
                )
            )
        ]
        self.arg_df = pd.DataFrame(
            params, columns=["angle", "width", "intensity", "t_stim", "speed"]
        )

        self.arg_group_df = self.arg_df.groupby(
            ["angle", "width", "intensity"], sort=False, as_index=False
        ).all()

        self.subdir = subdir

        self.device = device
        self.shuffle_offsets = shuffle_offsets
        self.randomstate = None
        if self.shuffle_offsets:
            self.randomstate = np.random.RandomState(seed=seed)

        self._dt = dt

        self._built = False
        if build_stim_on_init:
            self._build()
            self._resample()
            self._built = True

    @property
    def dt(self):
        return getattr(self, "_dt", None)

    @dt.setter
    def dt(self, value):
        if self._dt == value:
            self._dt = value
            if self._built:
                self._resample()
            return
        logging.warning(
            f"Cannot override dt={value} because responses with dt={self._dt} are "
            "initialized."
            f" Keeping dt={self._dt}."
        )

    def __len__(self):
        return len(self.arg_df)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n"
            + "Config:\n"
            + repr(self.config)
            + "Stimulus parameter:\n"
            + repr(self.arg_df)
        )

    @property
    def t_pre(self):
        return self._t_pre

    @property
    def t_post(self):
        return self._t_post

    def _build(self):
        self.wrap = RenderedOffsets(
            dict(
                angles=self.angles,
                widths=self.widths,
                intensities=self.intensities,
                offsets=self.offsets,
                led_width=self.led_width,
                height=self.height,
                n_bars=self.n_bars,
                bg_intensity=self.bg_intensity,
                bar_loc_horizontal=self.bar_loc_horizontal,
            )
        )

        self._offsets = torch.tensor(self.wrap.offsets[:], device=self.device)
        self._built = True

    def _resample(self):
        # resampling at runtime to allow for changing dt and to save GB of
        # storage.
        self.sequences = {}
        self.indices = {}
        for t, speed in zip(self.t_stim, self.speeds):
            sequence, indices = resample(
                self._offsets,
                t,
                self.dt,
                dim=1,
                device=self.device,
                return_indices=True,
            )
            if self.shuffle_offsets:
                # breakpoint()
                sequence = shuffle(sequence, self.randomstate)
            sequence = pad(
                sequence,
                t + self.t_pre,
                self.dt,
                mode="start",
                fill=self.bg_intensity,
            )
            sequence = pad(
                sequence,
                t + self.t_pre + self.t_post,
                self.dt,
                mode="end",
                pad_mode=self.post_pad_mode,
                fill=self.bg_intensity,
            )
            # Because we fix the distance that the bar moves but vary speeds we
            # have different stimulation times. To make all sequences equal
            # length for storing them in a single tensor, we pad them with nans
            # based on the maximal stimulation time (slowest speed). The nans
            # can later be removed before processing the traces.
            sequence = pad(
                sequence,
                self.t_stim_max + self.t_pre + self.t_post,
                self.dt,
                mode="end",
                fill=np.nan,
            )
            self.sequences[speed] = sequence
            self.indices[speed] = indices

    def _key(self, angle, width, intensity, speed):
        try:
            return self.arg_df.query(
                f"angle=={angle}"
                f" & width=={width}"
                f" & intensity == {intensity}"
                f" & speed == {speed}"
            ).index.values.item()
        except ValueError:
            raise ValueError(
                f"angle: {angle}, width: {width}, intensity: {intensity}, "
                f"speed: {speed} invalid."
            ) from None

    def get_sequence_id_from_arguments(self, angle, width, intensity, speed):
        # generic of _key
        return self._get_sequence_id_from_arguments(locals())

    def _params(self, key):
        return self.arg_df.iloc[key].values

    def _group_key(self, angle, width, intensity):
        return self.arg_group_df.query(
            f"angle=={angle}" f" & width=={width}" f" & intensity == {intensity}"
        ).index.values.item()

    def _group_params(self, key):
        return self.arg_group_df.iloc[key].values

    def get(self, angle, width, intensity, speed):
        key = self._key(angle, width, intensity, speed)
        return self[key]

    def get_item(self, key):
        angle, width, intensity, _, speed = self._params(key)
        return self.sequences[speed][self._group_key(angle, width, intensity)]

    def mask(self, angle=None, width=None, intensity=None, speed=None, t_stim=None):
        # 22x faster than df.query
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
        if angle is not None:
            condition = iterparam(angle, "angle", 0, condition)
        if width is not None:
            condition = iterparam(width, "width", 1, condition)
        if intensity is not None:
            condition = iterparam(intensity, "intensity", 2, condition)
        if t_stim is not None:
            condition = iterparam(t_stim, "t_stim", 3, condition)
        if speed is not None:
            condition = iterparam(speed, "speed", 4, condition)
        return condition

    def time_window(
        self,
        speed,
        from_column=-1,
        to_column=2,  # , start=-10, end=11, t_pre=1
    ):
        return time_window(
            speed, from_column, to_column, self.offsets[0], self.offsets[1]
        )

    def mask_between_seconds(self, t_start, t_end):
        time = self.time
        return (time >= t_start) & (time <= t_end)

    @property
    def time(self):
        return np.arange(-self.t_pre, self.t_stim_max + self.t_post - self.dt, self.dt)

    def stimulus(
        self,
        angle=None,
        width=None,
        intensity=None,
        speed=None,
        pre_stim=True,
        post_stim=True,
    ):
        """
        #TODO: docstring
        """
        key = self._key(angle, width, intensity, speed)
        stim = self[key][:, 360].cpu().numpy()
        if not post_stim:
            stim = filter_post([stim], self.t_post, self.dt).squeeze()
        if not pre_stim:
            stim = filter_pre(stim[None], self.t_pre, self.dt).squeeze()
        return stim

    def stimulus_parameters(self, angle=None, width=None, intensity=None, speed=None):
        def _number_to_list(*args):
            returns = tuple()
            for arg in args:
                if isinstance(arg, Number):
                    returns += ([arg],)
                else:
                    returns += (arg,)
            return returns

        angle, width, speed, intensity = _number_to_list(angle, width, speed, intensity)
        angle = angle or self.angles
        width = width or self.widths
        intensity = intensity or self.intensities
        speed = speed or self.speeds
        return angle, width, intensity, speed

    def sample_shape(
        self,
        angle=None,
        width=None,
        intensity=None,
        speed=None,
    ):
        if isinstance(angle, Number):
            angle = [angle]
        if isinstance(width, Number):
            width = [width]
        if isinstance(speed, Number):
            speed = [speed]
        if isinstance(intensity, Number):
            intensity = [intensity]
        angle = angle or self.angles
        width = width or self.widths
        intensity = intensity or self.intensities
        speed = speed or self.speeds
        return (
            len(angle),
            len(width),
            len(intensity),
            len(speed),
        )

    def time_to_center(self, speed: float) -> float:
        """Returns the time in s at which the bar reaches the center.

        Note: time = distance / velocity, i.e.
            time = (n_leds * led_width) / (speed * omm_width)
            with speed in ommatidia / s.
        """
        return np.abs(self.config.offsets[0]) * self.led_width / (speed * self.omm_width)

    def get_time_with_origin_at_onset(self):
        """Time with 0 at the onset of the stimulus."""
        return np.linspace(
            -self.t_pre,
            self.t_stim_max - self.t_pre + self.t_post,
            int(self.t_stim_max / self.dt)
            + int(self.t_post / self.dt)
            + int(self.t_pre / self.dt),
        )

    def get_time_with_origin_at_center(self, speed: float):
        """Time with 0 where the bar reaches the central column."""
        time_to_center = self.time_to_center(speed)
        n_steps = (
            int(self.t_stim_max / self.dt)
            + int(self.t_post / self.dt)
            + int(self.t_pre / self.dt)
        )
        return np.linspace(
            -(self.t_pre + time_to_center),
            n_steps * self.dt - (self.t_pre + time_to_center),
            n_steps,
        )

    def stimulus_cartoon(
        self,
        angle,
        width,
        intensity,
        speed,
        time_after_stimulus_onset=0.5,
        fig=None,
        ax=None,
        facecolor="#000000",
        cmap=plt.cm.bone,
        alpha=0.5,
        vmin=0,
        vmax=1,
        edgecolor="none",
        central_hex_color="#2f7cb9",
        **kwargs,
    ):
        fig, ax = init_plot(fig=fig, ax=ax)

        time = (
            np.arange(
                0,
                self.t_pre + self.t_stim_max + self.t_post - self.dt,
                self.dt,
            )
            - self.t_pre
        )
        index = np.argmin(np.abs(time - time_after_stimulus_onset))

        fig, ax, _ = quick_hex_scatter(
            self.get(angle=angle, width=width, speed=speed, intensity=intensity)
            .cpu()
            .numpy()[index],
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            figsize=[1, 1],
            max_extent=5,
            fig=fig,
            ax=ax,
            cmap=cmap,
            alpha=alpha,
            edgecolor=edgecolor,
            **kwargs,
        )
        rotation = np.array([
            [
                np.cos(np.radians(angle - 90)),
                -np.sin(np.radians(angle - 90)),
            ],
            [
                np.sin(np.radians(angle - 90)),
                np.cos(np.radians(angle - 90)),
            ],
        ])
        x = rotation @ np.array([0, -5])
        dx = rotation @ np.array([0, 1])
        ax.arrow(
            *x,
            *dx,
            facecolor=facecolor,
            width=0.75,
            head_length=2.5,
            edgecolor="k",
            linewidth=0.25,
        )
        _hex = RegularPolygon(
            (0, 0),
            numVertices=6,
            radius=1,
            linewidth=1,
            orientation=np.radians(30),
            edgecolor=central_hex_color,
            facecolor=central_hex_color,
            alpha=1,
            ls="-",
        )
        ax.add_patch(_hex)

        return fig, ax


class MovingEdge(MovingBar):
    """Moving edge stimulus.

    Args:
        offsets: tuple of int, optional
            Tuple of the first and last offset to the central column in half ommatidia.
        intensities: list of int, optional
            Intensity of the bar.
        speeds: list of float, optional
            Speed of the bar in half ommatidia per second.
        height: int, optional
            Height of the bar in half ommatidia.
        dt: float, optional
            Time step in seconds.
        subdir: str, optional
            Subdirectory where the stimulus is stored.
        device: str, optional
            Device to store the stimulus.
        post_pad_mode: str, optional
            Padding mode after the stimulus. One of 'continue', 'value', 'reflect'.
            If 'value' the padding is filled with the value of `bg_intensity`.
        t_pre: float, optional
            Time before the stimulus in seconds.
        t_post: float, optional
            Time after the stimulus in seconds.
        build_stim_on_init: bool, optional
            Build the stimulus on initialization.
        shuffle_offsets: bool, optional
            Shuffle the offsets to remove spatio-temporal correlation.
        seed: int, optional
            Seed for the random state.
        angles: list of int, optional
            List of angles in degrees.
    """

    def __init__(
        self,
        offsets=(-10, 11),  # in 1 * radians(2.25) led size
        intensities=[0, 1],
        speeds=[2.4, 4.8, 9.7, 13, 19, 25],  # in 1 * radians(5.8) / s
        height=9,  # in 1 * radians(2.25) led size
        dt=1 / 200,
        subdir="movingbar",
        device="cuda",
        post_pad_mode="continue",
        t_pre=1.0,
        t_post=1.0,
        build_stim_on_init=True,  # can speed up things downstream if only responses
        # are needed
        shuffle_offsets=False,  # shuffle offsets to remove spatio-temporal correlation
        # -- can be used as stimulus to compute a baseline of motion selectivity
        seed=0,  # only for shuffle_offsets
        angles=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    ):
        super().__init__(
            # arbitrary large value to make the 'bar' wide enough to appear as an edge
            widths=[80],
            offsets=offsets,
            intensities=intensities,
            speeds=speeds,
            height=height,
            dt=dt,
            subdir=subdir,
            device=device,
            # the center of the bar will start at the left edge of the screen
            bar_loc_horizontal=np.radians(0),
            post_pad_mode=post_pad_mode,
            t_pre=t_pre,
            t_post=t_post,
            build_stim_on_init=build_stim_on_init,
            shuffle_offsets=shuffle_offsets,
            seed=seed,
            angles=angles,
        )


def filter_post(resp, t_post, dt):
    """To remove the post stimulus time from responses.

    Because we fix the distance that the bar moves but vary speeds we
    have different stimulation times. To make all sequences equal
    length for storing them in a single tensor, we pad them with nans
    based on the maximal stimulation time (slowest speed).

    The post stimulus time is per speed in
    'first_nan - int(t_post / dt):first_nan'.

    Assuming resp can be partitioned along the temporal dimension as |pre stim|
    stimulus   |    post stimulus       |        nan padding        |
    |0:t_pre:|t_pre + t_stim:|t_pre + t_stim + t_post:|t_pre + t_stim_max +
    t_post|

    Args: resp of shape (n_samples, n_frames, ...)
    """
    _resp = []
    # for each sample
    for r in resp:
        # create a temporal mask
        mask = np.ones(r.shape[0]).astype(bool)
        # check where the nans are in (temporal, <#cell_type>)
        _nans = np.isnan(r)
        # if there are nans
        if _nans.any():
            # find the first nan (but latest across cell types, cause it needs
            # time to propagate)
            _first_nan_index = np.argmax(_nans, axis=0).max()

            # mask out the post stimulus time
            mask[_first_nan_index - int(t_post / dt) : _first_nan_index] = False

        else:
            mask[-int(t_post / dt) :] = False

        _resp.append(r[mask])
    return np.array(_resp)


def filter_pre(resp, t_pre, dt):
    """
    Args: resp of shape (n_samples, n_frames, ...)
    """
    return resp[:, int(t_pre / dt) :]


def time_window(speed, from_column=-1.5, to_column=1.5, start=-10, end=11):
    """Calculate start and end time when the bar passes from_column to_column.

    speed: in columns/s, i.e. 5.8deg / s
    from_column: in columns, i.e. 5.8deg
    to_column: in columns, ie. 5.8deg
    start: in led, i.e. 2.25 deg
    end: in led, i.e. 2.25 deg
    """
    start_in_columns = start * 2.25 / 5.8  # in 5.8deg
    end_in_columns = end * 2.25 / 5.8  # in 5.8deg

    # to make it symmetric around the central column, add a single led width
    # i.e. 2.25 deg in units of columns
    to_column += 2.25 / 5.8

    assert abs(start_in_columns) >= abs(from_column)
    assert abs(end_in_columns) >= abs(to_column)

    # calculate when the edge is at the from_column
    t_start = (abs(start_in_columns) - abs(from_column)) / speed
    # to when it's at the to_column
    t_end = t_start + (to_column - from_column) / speed
    return t_start, t_end


def mask_between_seconds(
    t_start, t_end, time=None, t_pre=None, t_stim=None, t_post=None, dt=None
):
    time = time if time is not None else np.arange(-t_pre, t_stim + t_post - dt, dt)
    return (time >= t_start) & (time <= t_end)
