"""'Transduction' of cartesian pixels to hexals on a regular hexagonal lattice."""

from itertools import product
from typing import Iterator, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from torch import nn

import flyvis
from flyvis.analysis.visualization.plt_utils import init_plot, rm_spines

from .utils import (
    hex_center_coordinates,
    is_inside_hex,
    median,
    render_bars_cartesian,
    render_gratings_cartesian,
)

__all__ = ["BoxEye", "HexEye"]

# ----- BoxEye -----------------------------------------------------------------


class BoxEye:
    """BoxFilter to produce an array of hexals matching the photoreceptor array.

    Args:
        extent: Radius, in number of receptors, of the hexagonal array.
        kernel_size: Photon collection radius, in pixels.

    Attributes:
        extent (int): Radius, in number of receptors, of the hexagonal array.
        kernel_size (int): Photon collection radius, in pixels.
        receptor_centers (torch.Tensor): Tensor of shape (hexals, 2) containing the y, x
            coordinates of the hexal centers.
        hexals (int): Number of hexals in the array.
        min_frame_size (torch.Tensor): Minimum frame size to contain the hexal array.
        pad (Tuple[int, int, int, int]): Padding to apply to the frame before convolution.
        conv (nn.Conv2d): Convolutional box filter to apply to the frame.
    """

    def __init__(self, extent: int = 15, kernel_size: int = 13):
        self.extent = extent
        self.kernel_size = kernel_size
        self.receptor_centers = torch.tensor(
            [*self._receptor_centers()], dtype=torch.long
        )
        self.hexals = len(self.receptor_centers)
        # The rest of kernel_size distance from outer centers to the border
        # is taken care of by the padding of the convolution object.
        self.min_frame_size = (
            self.receptor_centers.max(dim=0).values
            - self.receptor_centers.min(dim=0).values
            + 1
        )
        self._set_filter()

        pad = (self.kernel_size - 1) / 2
        self.pad = (
            int(np.ceil(pad)),
            int(np.floor(pad)),
            int(np.ceil(pad)),
            int(np.floor(pad)),
        )

    def _receptor_centers(self) -> Iterator[Tuple[float, float]]:
        """Generate receptor center coordinates.

        Returns:
            Iterator[Tuple[float, float]]: Yields y, x coordinates of receptor centers.
        """
        n = self.extent
        d = self.kernel_size
        for u in range(-n, n + 1):
            v_min = max(-n, -n - u)
            v_max = min(n, n - u)
            for v in range(v_min, v_max + 1):
                # y = -d * v
                # x = 2 / np.sqrt(3) * d * (u + v/2)
                y = d * (
                    u + v / 2
                )  # - d * v # either must be negative or origin must be upper
                x = d * v  # 2 / np.sqrt(3) * d * (u + v / 2)
                yield y, x
                # xs.append()
                # ys.append()

    def _set_filter(self) -> None:
        """Set up the convolutional filter for the box kernel."""
        self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1, padding=0)
        self.conv.weight.data /= self.conv.weight.data
        self.conv.bias.data.fill_(0)  # if not self.requires_grad else None
        self.conv.weight.requires_grad = False  # self.requires_grad
        self.conv.bias.requires_grad = False  # self.requires_grad

    def __call__(
        self,
        sequence: torch.Tensor,
        ftype: Literal["mean", "sum", "median"] = "mean",
        hex_sample: bool = True,
    ) -> torch.Tensor:
        """Apply a box kernel to all frames in a sequence.

        Args:
            sequence: Cartesian movie sequences of shape (samples, frames, height, width).
            ftype: Filter type.
            hex_sample: If False, returns filtered cartesian sequences.

        Returns:
            torch.Tensor: Shape (samples, frames, 1, hexals) if hex_sample is True,
                otherwise (samples, frames, height, width).
        """
        samples, frames, height, width = sequence.shape

        if not isinstance(sequence, torch.Tensor):
            # auto-moving to GPU in case default tensor is cuda but passed
            # sequence is not, for convenience
            sequence = torch.tensor(sequence, dtype=torch.float32, device=flyvis.device)

        if (self.min_frame_size > torch.tensor([height, width])).any():
            # to rescale to the minimum frame size
            sequence = ttf.resize(sequence, self.min_frame_size.tolist())
            height, width = sequence.shape[2:]

        def _convolve():
            # convole each sample sequentially to avoid gpu memory issues
            def conv(x):
                return self.conv(x.unsqueeze(1))

            return torch.cat(
                tuple(map(conv, torch.unbind(F.pad(sequence, self.pad), dim=0))), dim=0
            )

        if ftype == "mean":
            out = _convolve() / self.kernel_size**2
        elif ftype == "sum":
            out = _convolve()
        elif ftype == "median":
            out = median(sequence, self.kernel_size)
        else:
            raise ValueError("ftype must be 'sum', 'mean', or 'median." f"Is {ftype}.")

        if hex_sample is True:
            return self.hex_render(out).reshape(samples, frames, 1, -1)

        return out.reshape(samples, frames, height, width)

    def hex_render(self, sequence: torch.Tensor) -> torch.Tensor:
        """Sample receptor locations from a sequence of cartesian frames.

        Args:
            sequence: Cartesian movie sequences of shape (samples, frames, height, width).

        Returns:
            torch.Tensor: Shape (samples, frames, 1, hexals).

        Note:
            Resizes the sequence to the minimum frame size if necessary.
        """
        h, w = sequence.shape[2:]
        if (self.min_frame_size > torch.tensor([h, w])).any():
            sequence = ttf.resize(sequence, self.min_frame_size.tolist())
            h, w = sequence.shape[2:]
        c = self.receptor_centers + torch.tensor([h // 2, w // 2])
        out = sequence[:, :, c[:, 0], c[:, 1]]
        return out.view(*sequence.shape[:2], 1, -1)

    def illustrate(self) -> plt.Figure:
        """Illustrate the receptive field centers and the hexagonal sampling.

        Returns:
            plt.Figure: Matplotlib figure object.
        """
        figsize = [2, 2]
        fontsize = 5
        y_hc, x_hc = np.array(list(self._receptor_centers())).T

        height, width = self.min_frame_size.cpu().numpy()
        x_img, y_img = np.array(
            list(
                product(
                    np.arange(-width / 2, width / 2),
                    np.arange(-height / 2, height / 2),
                )
            )
        ).T

        r = np.sqrt(2) * self.kernel_size / 2

        vertices = []
        angles = [45, 135, 225, 315, 405]
        for _y_c, _x_c in zip(y_hc, x_hc):
            _vertices = []
            for angle in angles:
                offset = r * np.exp(np.radians(angle) * 1j)
                _vertices.append([_y_c + offset.real, _x_c + offset.imag])
            vertices.append(_vertices)
        vertices = np.transpose(vertices, (1, 2, 0))

        fig, ax = init_plot(figsize=figsize, fontsize=fontsize)
        ax.scatter(x_hc, y_hc, color="#00008B", zorder=1, s=0.5)
        ax.scatter(x_img, y_img, color="#34ebd6", s=0.1, zorder=0)

        for h in range(len(x_hc)):
            for i in range(4):
                y1, x1 = vertices[i, :, h]  # x1, y1: (n_hexagons)
                y2, x2 = vertices[i + 1, :, h]
                ax.plot([x1, x2], [y1, y2], c="black", lw=0.25)

        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(-height / 2, height / 2)
        rm_spines(ax)
        # fig.tight_layout()
        return fig


# ----- HexEye (slower, more precise) ----------------------------------------


class HexEye:
    """Hexagonal eye model for more precise rendering.

    Args:
        n_ommatidia: Number of ommatidia in the eye. Must currently fill a regular
            hex grid.
        ppo: Pixels per ommatidium.
        monitor_height_px: Monitor height in pixels.
        monitor_width_px: Monitor width in pixels.
        device: Computation device.
        dtype: Data type for computations.

    Attributes:
        monitor_width_px (int): Monitor width in pixels.
        monitor_height_px (int): Monitor height in pixels.
        is_inside (torch.Tensor): Boolean mask for pixels inside hexagons.
        n_ommatidia (int): Number of ommatidia in the eye.
        omm_width_rad (float): Ommatidium width in radians.
        omm_height_rad (float): Ommatidium height in radians.
        ppo (int): Pixels per ommatidium.
        n_hex_circfer (float): Number of hexagons in the circumference.
        device (torch.device): Computation device.
        dtype (torch.dtype): Data type for computations.
    """

    def __init__(
        self,
        n_ommatidia: int = 721,
        ppo: int = 25,
        monitor_height_px: Optional[int] = None,
        monitor_width_px: Optional[int] = None,
        device: torch.device = flyvis.device,
        dtype: torch.dtype = torch.float16,
    ):
        n_hex_circfer = 2 * (-1 / 2 + np.sqrt(1 / 4 - ((1 - n_ommatidia) / 3))) + 1

        if n_hex_circfer % 1 != 0:
            raise ValueError(f"{n_ommatidia} does not fill a regular hex grid.")

        self.monitor_width_px = monitor_width_px or ppo * int(n_hex_circfer)
        self.monitor_height_px = monitor_height_px or ppo * int(n_hex_circfer)

        x_hc, y_hc, (dist_w, dist_h) = hex_center_coordinates(
            n_ommatidia, self.monitor_width_px, self.monitor_height_px
        )

        x_img, y_img = np.array(
            list(
                product(
                    np.arange(self.monitor_width_px),
                    np.arange(self.monitor_height_px),
                )
            )
        ).T

        dist_to_edge = (dist_w + dist_h) / 4

        _, self.is_inside = is_inside_hex(
            torch.tensor(y_img, dtype=dtype, device=device),
            torch.tensor(x_img, dtype=dtype, device=device),
            torch.tensor(x_hc, dtype=dtype, device=device),
            torch.tensor(y_hc, dtype=dtype, device=device),
            torch.tensor(dist_to_edge, dtype=dtype, device=device),
            torch.tensor(np.radians(0), dtype=dtype, device=device),
            device=device,
            dtype=dtype,
        )
        self.kernel_sum = self.is_inside.sum(dim=0)
        # Clean up excessive memory usage.
        if device != "cpu":
            torch.cuda.empty_cache()
        self.n_ommatidia = n_ommatidia
        self.omm_width_rad = np.radians(5.8)
        self.omm_height_rad = np.radians(5.8)
        self.ppo = ppo or int(
            (self.monitor_width_px + self.monitor_width_px) / (2 * n_hex_circfer)
        )
        self.n_hex_circfer = n_hex_circfer
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        stim: torch.Tensor,
        mode: Literal["mean", "median", "sum"] = "mean",
        n_chunks: int = 1,
    ) -> torch.Tensor:
        """Process stimulus through the hexagonal eye model.

        Args:
            stim: Input stimulus tensor (n_frames, height, width) or
                (n_frames, height * width). Height and width must correspond to the
                monitor size.
            mode: Processing mode.
            n_chunks: Number of chunks to process the stimulus in.

        Returns:
            torch.Tensor: Processed stimulus.
        """
        shape = stim.shape
        if mode not in ["mean", "median", "sum"]:
            raise ValueError

        if len(stim.shape) == 3:
            h, w = stim.shape[1:]
            # resize to monitor size if necessary
            if h < self.monitor_height_px or w < self.monitor_width_px:
                stim = ttf.resize(stim, [self.monitor_height_px, self.monitor_width_px])
            stim = stim.reshape(shape[0], -1)

        try:
            if mode == "median":
                n_pixels = shape[1]
                stim = median(
                    stim.view(-1, self.monitor_height_px, self.monitor_width_px)
                    .float()
                    .to(self.device),
                    int(np.sqrt(self.ppo)),
                ).view(-1, n_pixels)
            elif mode == "sum":
                return (stim[:, :, None] * self.is_inside).sum(dim=1)
            elif mode == "mean":
                return (stim[:, :, None] * self.is_inside).sum(dim=1) / self.kernel_sum
            else:
                raise ValueError(f"Invalid mode: {mode}")

        except RuntimeError as e:
            if "memory" not in str(e):
                raise e
            if "CUDA" in str(e):
                torch.cuda.empty_cache()
            if n_chunks > shape[0]:
                raise ValueError from e

            chunks = torch.chunk(stim, max(n_chunks, 1), dim=0)

            def map_fn(chunk):
                return self(chunk, mode=mode, n_chunks=n_chunks + 1)

            return torch.cat(tuple(map(map_fn, chunks)), dim=0)

    # Rendering functions
    # TODO: move to separate file and make agnostic of the eye model

    def render_bar(
        self,
        bar_width_rad: float,
        bar_height_rad: float,
        bar_loc_theta: float,
        bar_loc_phi: float,
        n_bars: int,
        bar_intensity: float,
        bg_intensity: float,
        moving_angle: float,
        cartesian: bool = False,
        mode: Literal["mean", "median", "sum"] = "mean",
    ) -> Union[np.ndarray, torch.Tensor]:
        """Render a bar stimulus.

        Args:
            bar_width_rad: Width of bars in radians.
            bar_height_rad: Height of bars in radians.
            bar_loc_theta: Horizontal location of bars in radians.
            bar_loc_phi: Vertical location of bars in radians.
            n_bars: Number of bars.
            bar_intensity: Intensity of the bar.
            bg_intensity: Intensity of the background.
            moving_angle: Rotation angle in degrees.
            cartesian: If True, return cartesian coordinates.
            mode: Processing mode.

        Returns:
            Union[np.ndarray, torch.Tensor]: Generated bar stimulus.
        """
        bar_width_px = int(bar_width_rad / self.omm_width_rad * self.ppo)
        bar_height_px = int(bar_height_rad / self.omm_height_rad * self.ppo)
        bar_loc_horizontal_px = int(
            self.monitor_width_px * bar_loc_theta / np.radians(180)
        )
        bar_loc_vertical_px = int(self.monitor_height_px * bar_loc_phi / np.radians(180))

        bar = render_bars_cartesian(
            self.monitor_height_px,
            self.monitor_width_px,
            bar_width_px,
            bar_height_px,
            bar_loc_horizontal_px,
            bar_loc_vertical_px,
            n_bars,
            bar_intensity,
            bg_intensity,
            moving_angle,
        )
        if cartesian:
            return bar
        return self(torch.tensor(bar.flatten(), device=self.device)[None], mode)

    def render_grating(
        self,
        period_rad: float,
        phase_rad: float,
        intensity: float,
        bg_intensity: float,
        moving_angle: float,
        width_rad: Optional[float] = None,
        height_rad: Optional[float] = None,
        cartesian: bool = False,
        mode: Literal["mean", "median", "sum"] = "mean",
    ) -> Union[np.ndarray, torch.Tensor]:
        """Render a grating stimulus.

        Args:
            period_rad: Period of the grating in radians.
            phase_rad: Phase of the grating in radians.
            intensity: Intensity of the grating.
            bg_intensity: Intensity of the background.
            moving_angle: Rotation angle in degrees.
            width_rad: Width of the grating in radians.
            height_rad: Height of the grating in radians.
            cartesian: If True, return cartesian coordinates.
            mode: Processing mode.

        Returns:
            Union[np.ndarray, torch.Tensor]: Generated grating stimulus.
        """
        period_px = int(period_rad / self.omm_width_rad * self.ppo)
        phase_px = int(phase_rad / self.omm_width_rad * self.ppo)

        height_rad_px = None
        if height_rad:
            height_rad_px = int(height_rad / self.omm_height_rad * self.ppo)

        width_rad_px = None
        if width_rad:
            width_rad_px = int(width_rad / self.omm_width_rad * self.ppo)

        grating = render_gratings_cartesian(
            self.monitor_height_px,
            self.monitor_width_px,
            period_px,
            intensity,
            bg_intensity,
            grating_phase_px=phase_px,
            rotate=moving_angle,
            grating_height_px=height_rad_px,
            grating_width_px=width_rad_px,
        )
        if cartesian:
            return grating
        return self(torch.tensor(grating.flatten(), device=self.device)[None], mode)

    def render_grating_offsets(
        self,
        period_rad: float,
        intensity: float,
        bg_intensity: float,
        moving_angle: float,
        width_rad: Optional[float] = None,
        height_rad: Optional[float] = None,
        cartesian: bool = False,
        mode: Literal["mean", "median", "sum"] = "mean",
    ) -> Union[np.ndarray, torch.Tensor]:
        """Render grating stimuli with a range of offsets.

        Args:
            period_rad: Period of the grating in radians.
            intensity: Intensity of the grating.
            bg_intensity: Intensity of the background.
            moving_angle: Rotation angle in degrees.
            width_rad: Width of the grating in radians.
            height_rad: Height of the grating in radians.
            cartesian: If True, return cartesian coordinates.
            mode: Processing mode.

        Returns:
            Union[np.ndarray, torch.Tensor]: Generated grating stimuli with offsets.
        """
        dphase_px = np.radians(
            5.8 / 2
        )  # half ommatidia width - corresponds to led width of 2.25 degree
        n_offsets = np.ceil(period_rad / dphase_px).astype(int)
        gratings = []
        for offset in range(n_offsets):
            gratings.append(
                self.render_grating(
                    period_rad,
                    offset * dphase_px,
                    intensity,
                    bg_intensity,
                    moving_angle,
                    width_rad=width_rad,
                    height_rad=height_rad,
                    cartesian=cartesian,
                    mode=mode,
                )
            )
        if cartesian:
            return np.array(gratings)
        return torch.cat(gratings, dim=0)

    def render_offset_bars(
        self,
        bar_width_rad: float,
        bar_height_rad: float,
        n_bars: int,
        offsets: List[float],
        bar_intensity: float,
        bg_intensity: float,
        moving_angle: float,
        bar_loc_horizontal: float = np.radians(90),
        bar_loc_vertical: float = np.radians(90),
        mode: Literal["mean", "median", "sum"] = "mean",
    ) -> torch.Tensor:
        """Render bars with a range of offsets.

        Args:
            bar_width_rad: Width of bars in radians.
            bar_height_rad: Height of bars in radians.
            n_bars: Number of bars.
            offsets: Offsets of bars wrt. the center in radians.
            bar_intensity: Intensity of the bar.
            bg_intensity: Intensity of the background.
            moving_angle: Rotation angle in degrees.
            bar_loc_horizontal: Horizontal location of bars in radians.
            bar_loc_vertical: Vertical location of bars in radians.
            mode: Processing mode.

        Returns:
            torch.Tensor: Generated offset bars.
        """
        flashes = []
        for offset in offsets:
            flashes.append(
                self.render_bar(
                    bar_width_rad,
                    bar_height_rad,
                    bar_loc_horizontal + offset,
                    bar_loc_vertical,
                    n_bars,
                    bar_intensity,
                    bg_intensity,
                    moving_angle,
                    mode=mode,
                )
            )
        return torch.cat(flashes, dim=0)

    def render_bar_movie(
        self,
        t_stim: float,
        dt: float,
        bar_width_rad: float,
        bar_height_rad: float,
        n_bars: int,
        offsets: List[float],
        bar_intensity: float,
        bg_intensity: float,
        moving_angle: float,
        t_pre: float = 0.0,
        t_between: float = 0.0,
        t_post: float = 0.0,
        bar_loc_horizontal: float = np.radians(90),
        bar_loc_vertical: float = np.radians(90),
    ) -> torch.Tensor:
        """Render moving bars.

        Args:
            t_stim: Stimulus duration.
            dt: Temporal resolution.
            bar_width_rad: Width of bars in radians.
            bar_height_rad: Height of bars in radians.
            n_bars: Number of bars.
            offsets: Offsets of bars wrt. the center in radians.
            bar_intensity: Intensity of the bar.
            bg_intensity: Intensity of the background.
            moving_angle: Rotation angle in degrees.
            t_pre: Grey pre stimulus duration.
            t_between: Grey between offset stimulus duration.
            t_post: Grey post stimulus duration.
            bar_loc_horizontal: Horizontal location of bars in radians.
            bar_loc_vertical: Vertical location of bars in radians.

        Returns:
            torch.Tensor: Generated moving bars.
        """
        pre_frames = round(t_pre / dt)
        stim_frames = round(t_stim / (len(offsets) * dt))
        if stim_frames == 0:
            raise ValueError(
                f"stimulus time {t_stim}s not sufficient to sample {len(offsets)} "
                "offsets at {dt}s"
            )
        between_frames = round(t_between / dt)
        post_frames = round(t_post / dt)

        flashes = []
        if pre_frames:
            flashes.append(torch.ones([pre_frames, self.n_ommatidia]) * bg_intensity)

        for i, offset in enumerate(offsets):
            flash = self.render_bar(
                bar_width_rad,
                bar_height_rad,
                bar_loc_horizontal + offset,
                bar_loc_vertical,
                n_bars,
                bar_intensity,
                bg_intensity,
                moving_angle,
            )
            flashes.append(flash.repeat(stim_frames, 1))

            if between_frames and i < len(offsets) - 1:
                flashes.append(
                    torch.ones([between_frames, self.n_ommatidia]) * bg_intensity
                )
        if post_frames:
            flashes.append(torch.ones([post_frames, self.n_ommatidia]) * bg_intensity)
        return torch.cat(flashes, dim=0)

    def illustrate(
        self, figsize: List[int] = [5, 5], fontsize: int = 5
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Illustrate the hexagonal eye model.

        Args:
            figsize: Figure size.
            fontsize: Font size for the plot.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects.
        """
        x_hc, y_hc, (dist_w, dist_h) = hex_center_coordinates(
            self.n_ommatidia, self.monitor_width_px, self.monitor_height_px
        )

        x_img, y_img = np.array(
            list(
                product(
                    np.arange(self.monitor_width_px),
                    np.arange(
                        self.monitor_height_px,
                    ),
                )
            )
        ).T

        dist_to_edge = (dist_w + dist_h) / 4

        vertices, _ = is_inside_hex(
            torch.tensor(y_img, dtype=self.dtype),
            torch.tensor(x_img, dtype=self.dtype),
            torch.tensor(x_hc, dtype=self.dtype),
            torch.tensor(y_hc, dtype=self.dtype),
            torch.tensor(dist_to_edge, dtype=self.dtype),
            torch.tensor(np.radians(0), dtype=self.dtype),
        )
        vertices = vertices.cpu()
        fig, ax = init_plot(figsize=figsize, fontsize=fontsize)
        ax.scatter(x_hc, y_hc, color="#eb4034", zorder=1)
        ax.scatter(x_img, y_img, color="#34ebd6", s=0.5, zorder=0)

        for h in range(self.n_ommatidia):
            for i in range(6):
                x1, y1 = vertices[i, :, h]  # x1, y1: (n_hexagons)
                x2, y2 = vertices[i + 1, :, h]
                ax.plot([x1, x2], [y1, y2], c="black")

        ax.set_xlim(0, self.monitor_width_px)
        ax.set_ylim(0, self.monitor_height_px)
        rm_spines(ax)
        # fig.tight_layout()
        return fig, ax
