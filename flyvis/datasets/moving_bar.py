import logging
from itertools import product
from numbers import Number
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datamate import Directory, Namespace, root
from matplotlib.colors import Colormap
from matplotlib.patches import RegularPolygon
from tqdm.auto import tqdm
from typing_extensions import Literal

import flyvis
from flyvis import renderings_dir
from flyvis.analysis.visualization.plots import quick_hex_scatter
from flyvis.analysis.visualization.plt_utils import init_plot
from flyvis.datasets.datasets import StimulusDataset

from .rendering import HexEye
from .rendering.utils import pad, resample, shuffle

logging = logging.getLogger(__name__)

__all__ = ["RenderedOffsets", "MovingBar", "MovingEdge"]


@root(renderings_dir)
class RenderedOffsets(Directory):
    """Rendered offsets for the moving bar stimulus.

    This class precomputes the offsets for moving bar (edge) stimuli and stores them
    in a directory. At runtime, the offsets are resampled to efficiently generate
    stimuli with different durations and temporal resolutions.

    Args:
        offsets: List of offset values.
        angles: List of angle values in degrees.
        widths: List of width values.
        intensities: List of intensity values.
        led_width: Width of LED in radians.
        height: Height of the bar in radians.
        n_bars: Number of bars.
        bg_intensity: Background intensity.
        bar_loc_horizontal: Horizontal location of the bar in radians.

    Attributes:
        offsets (ArrayFile): Rendered offsets for different stimulus parameters.
    """

    def __init__(
        self,
        offsets: list[int] = list(range(-10, 11)),
        angles: list[int] = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
        widths: list[int] = [1, 2, 4],
        intensities: list[int] = [0, 1],
        led_width: float = np.radians(2.25),
        height: float = np.radians(2.25) * 9,
        n_bars: int = 1,
        bg_intensity: float = 0.5,
        bar_loc_horizontal: float = np.radians(90),
    ):
        eye = HexEye(721, 25)

        params = list(product(angles, widths, intensities))

        sequences = {}
        _tqdm = tqdm(total=len(params), desc="building stimuli")
        for angle in angles:
            for width in widths:
                for intensity in intensities:
                    offset_bars = eye.render_offset_bars(
                        bar_width_rad=width * led_width,
                        bar_height_rad=height,
                        n_bars=n_bars,
                        offsets=np.array(offsets) * led_width,
                        bar_intensity=intensity,
                        bg_intensity=bg_intensity,
                        moving_angle=angle,
                        bar_loc_horizontal=bar_loc_horizontal,
                    )
                    sequences[(angle, width, intensity)] = offset_bars
                    _tqdm.update()

        _tqdm.close()
        self.offsets = torch.stack([sequences[p] for p in params]).cpu().numpy()


class MovingBar(StimulusDataset):
    """Moving bar stimulus.

    Args:
        widths: Width of the bar in half ommatidia.
        offsets: First and last offset to the central column in half ommatidia.
        intensities: Intensity of the bar.
        speeds: Speed of the bar in half ommatidia per second.
        height: Height of the bar in half ommatidia.
        dt: Time step in seconds.
        device: Device to store the stimulus.
        bar_loc_horizontal: Horizontal location of the bar in radians from left to
            right of image plane. np.radians(90) is the center.
        post_pad_mode: Padding mode after the stimulus. One of 'continue', 'value',
            'reflect'. If 'value' the padding is filled with `bg_intensity`.
        t_pre: Time before the stimulus in seconds.
        t_post: Time after the stimulus in seconds.
        build_stim_on_init: Build the stimulus on initialization.
        shuffle_offsets: Shuffle the offsets to remove spatio-temporal correlation.
        seed: Seed for the random state.
        angles: List of angles in degrees.

    Attributes:
        config (Namespace): Configuration parameters.
        omm_width (float): Width of ommatidium in radians.
        led_width (float): Width of LED in radians.
        angles (np.ndarray): Array of angles in degrees.
        widths (np.ndarray): Array of widths in half ommatidia.
        offsets (np.ndarray): Array of offsets in half ommatidia.
        intensities (np.ndarray): Array of intensities.
        speeds (np.ndarray): Array of speeds in half ommatidia per second.
        bg_intensity (float): Background intensity.
        n_bars (int): Number of bars.
        bar_loc_horizontal (float): Horizontal location of bar in radians.
        t_stim (np.ndarray): Stimulation times for each speed.
        t_stim_max (float): Maximum stimulation time.
        height (float): Height of bar in radians.
        post_pad_mode (str): Padding mode after the stimulus.
        arg_df (pd.DataFrame): DataFrame of stimulus parameters.
        arg_group_df (pd.DataFrame): Grouped DataFrame of stimulus parameters.
        device (str): Device for storing stimuli.
        shuffle_offsets (bool): Whether to shuffle offsets.
        randomstate (np.random.RandomState): Random state for shuffling.
    """

    arg_df: pd.DataFrame = None

    def __init__(
        self,
        widths: list[int] = [1, 2, 4],
        offsets: tuple[int, int] = (-10, 11),
        intensities: list[float] = [0, 1],
        speeds: list[float] = [2.4, 4.8, 9.7, 13, 19, 25],
        height: int = 9,
        dt: float = 1 / 200,
        device: str = flyvis.device,
        bar_loc_horizontal: float = np.radians(90),
        post_pad_mode: Literal["continue", "value", "reflect"] = "value",
        t_pre: float = 1.0,
        t_post: float = 1.0,
        build_stim_on_init: bool = True,
        shuffle_offsets: bool = False,
        seed: int = 0,
        angles: list[int] = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    ) -> None:
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
    def dt(self) -> float:
        """Time step in seconds."""
        return getattr(self, "_dt", None)

    @dt.setter
    def dt(self, value: float) -> None:
        if self._dt == value:
            self._dt = value
            if self._built:
                self._resample()
            return
        logging.warning(
            "Cannot override dt=%s because responses with dt=%s are initialized. "
            "Keeping dt=%s.",
            value,
            self._dt,
            self._dt,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}\n"
            + "Config:\n"
            + repr(self.config)
            + "Stimulus parameter:\n"
            + repr(self.arg_df)
        )

    @property
    def t_pre(self) -> float:
        """Time before stimulus onset in seconds."""
        return self._t_pre

    @property
    def t_post(self) -> float:
        """Time after stimulus offset in seconds."""
        return self._t_post

    def _build(self) -> None:
        """Build the stimulus."""
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

    def _resample(self) -> None:
        """Resample the stimulus at runtime."""
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

    def _key(self, angle: float, width: float, intensity: float, speed: float) -> int:
        """Get the key for a specific stimulus configuration."""
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

    def get_sequence_id_from_arguments(
        self, angle: float, width: float, intensity: float, speed: float
    ) -> int:
        """Get sequence ID from stimulus arguments."""
        return self.get_stimulus_index(locals())

    def _params(self, key: int) -> np.ndarray:
        """Get parameters for a given key."""
        return self.arg_df.iloc[key].values

    def _group_key(self, angle: float, width: float, intensity: float) -> int:
        """Get group key for a specific stimulus configuration."""
        return self.arg_group_df.query(
            f"angle=={angle}" f" & width=={width}" f" & intensity == {intensity}"
        ).index.values.item()

    def _group_params(self, key: int) -> np.ndarray:
        """Get group parameters for a given key."""
        return self.arg_group_df.iloc[key].values

    def get(
        self, angle: float, width: float, intensity: float, speed: float
    ) -> torch.Tensor:
        """Get stimulus for specific parameters."""
        key = self._key(angle, width, intensity, speed)
        return self[key]

    def get_item(self, key: int) -> torch.Tensor:
        """Get stimulus for a specific key."""
        angle, width, intensity, _, speed = self._params(key)
        return self.sequences[speed][self._group_key(angle, width, intensity)]

    def mask(
        self,
        angle: Optional[float] = None,
        width: Optional[float] = None,
        intensity: Optional[float] = None,
        speed: Optional[float] = None,
        t_stim: Optional[float] = None,
    ) -> np.ndarray:
        """Create a mask for specific stimulus parameters."""
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

    @property
    def time(self) -> np.ndarray:
        """Time array for the stimulus."""
        return np.arange(-self.t_pre, self.t_stim_max + self.t_post - self.dt, self.dt)

    def stimulus(
        self,
        angle: Optional[float] = None,
        width: Optional[float] = None,
        intensity: Optional[float] = None,
        speed: Optional[float] = None,
        pre_stim: bool = True,
        post_stim: bool = True,
    ) -> np.ndarray:
        """Get stimulus for specific parameters.

        Args:
            angle: Angle of the bar.
            width: Width of the bar.
            intensity: Intensity of the bar.
            speed: Speed of the bar.
            pre_stim: Include pre-stimulus period.
            post_stim: Include post-stimulus period.

        Returns:
            Stimulus array.
        """
        key = self._key(angle, width, intensity, speed)
        stim = self[key][:, 360].cpu().numpy()
        if not post_stim:
            stim = filter_post([stim], self.t_post, self.dt).squeeze()
        if not pre_stim:
            stim = filter_pre(stim[None], self.t_pre, self.dt).squeeze()
        return stim

    def stimulus_parameters(
        self,
        angle: Optional[float] = None,
        width: Optional[float] = None,
        intensity: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> tuple[list, ...]:
        """Get stimulus parameters."""

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
        angle: Optional[float] = None,
        width: Optional[float] = None,
        intensity: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> tuple[int, ...]:
        """Get shape of stimulus sample for given parameters."""
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
        """Calculate time for bar to reach center at given speed."""
        # Note: time = distance / velocity, i.e.
        #     time = (n_leds * led_width) / (speed * omm_width)
        #     with speed in ommatidia / s.
        return np.abs(self.config.offsets[0]) * self.led_width / (speed * self.omm_width)

    def get_time_with_origin_at_onset(self) -> np.ndarray:
        """Get time array with origin at stimulus onset."""
        return np.linspace(
            -self.t_pre,
            self.t_stim_max - self.t_pre + self.t_post,
            int(self.t_stim_max / self.dt)
            + int(self.t_post / self.dt)
            + int(self.t_pre / self.dt),
        )

    def get_time_with_origin_at_center(self, speed: float) -> np.ndarray:
        """Get time array with origin where bar reaches central column."""
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
        angle: float,
        width: float,
        intensity: float,
        speed: float,
        time_after_stimulus_onset: float = 0.5,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        facecolor: str = "#000000",
        cmap: Colormap = plt.cm.bone,
        alpha: float = 0.5,
        vmin: float = 0,
        vmax: float = 1,
        edgecolor: str = "none",
        central_hex_color: str = "#2f7cb9",
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a cartoon representation of the stimulus."""
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

    This class creates a moving edge stimulus by using a very wide bar.

    Args:
        offsets: First and last offset to the central column in half ommatidia.
        intensities: Intensity of the edge.
        speeds: Speed of the edge in half ommatidia per second.
        height: Height of the edge in half ommatidia.
        dt: Time step in seconds.
        device: Device to store the stimulus.
        post_pad_mode: Padding mode after the stimulus.
        t_pre: Time before the stimulus in seconds.
        t_post: Time after the stimulus in seconds.
        build_stim_on_init: Build the stimulus on initialization.
        shuffle_offsets: Shuffle the offsets to remove spatio-temporal correlation.
        seed: Seed for the random state.
        angles: List of angles in degrees.

    Note:
        This class uses a very wide bar (width=80) under the hood to render an
        edge stimulus.
    """

    def __init__(
        self,
        offsets: tuple[int, int] = (-10, 11),
        intensities: list[float] = [0, 1],
        speeds: list[float] = [2.4, 4.8, 9.7, 13, 19, 25],
        height: int = 9,
        dt: float = 1 / 200,
        device: str = flyvis.device,
        post_pad_mode: Literal["continue", "value", "reflect"] = "continue",
        t_pre: float = 1.0,
        t_post: float = 1.0,
        build_stim_on_init: bool = True,
        shuffle_offsets: bool = False,
        seed: int = 0,
        angles: list[int] = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
        **kwargs,
    ) -> None:
        super().__init__(
            widths=[80],
            offsets=offsets,
            intensities=intensities,
            speeds=speeds,
            height=height,
            dt=dt,
            device=device,
            bar_loc_horizontal=np.radians(0),
            post_pad_mode=post_pad_mode,
            t_pre=t_pre,
            t_post=t_post,
            build_stim_on_init=build_stim_on_init,
            shuffle_offsets=shuffle_offsets,
            seed=seed,
            angles=angles,
        )


def filter_post(resp: np.ndarray, t_post: float, dt: float) -> np.ndarray:
    """Remove post-stimulus time from responses.

    Args:
        resp: Response array of shape (n_samples, n_frames, ...).
        t_post: Post-stimulus time in seconds.
        dt: Time step in seconds.

    Returns:
        Filtered response array.

    Note:
        The post stimulus time is per speed in 'first_nan - int(t_post / dt):first_nan'.
        Assuming resp can be partitioned along the temporal dimension as:
        |pre stim|stimulus|post stimulus|nan padding|
        |0:t_pre:|t_pre + t_stim:|t_pre + t_stim + t_post:|t_pre + t_stim_max + t_post|
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


def filter_pre(resp: np.ndarray, t_pre: float, dt: float) -> np.ndarray:
    """Remove pre-stimulus time from responses.

    Args:
        resp: Response array of shape (n_samples, n_frames, ...).
        t_pre: Pre-stimulus time in seconds.
        dt: Time step in seconds.

    Returns:
        Filtered response array.
    """
    return resp[:, int(t_pre / dt) :]
