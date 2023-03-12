"""'Transduction' of cartesian pixels to hexals on a regular hexagonal lattice.
"""
from typing import Iterator, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from itertools import product

__all__ = ["BoxEye", "HexEye"]

# ----- BoxEye (faster, less precise) ------------------------------------------


class BoxEye:  # BoxEye
    """BoxFilter to produce an array of hexals matching the photoreceptor array.

    Args:
        extent: int
        "Radius, in number of receptors, of the hexagonal array"
        kernel_size: float
        "Photon collection radius, in pixels"

    """

    def __init__(self, extent: int, kernel_size: float):
        self.extent = extent
        self.kernel_size = kernel_size
        self.receptor_centers = (
            torch.Tensor([*self._receptor_centers()]).long().to(torch.tensor(0).device)
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
        """
        Generate receptive field centers for the population.

        Generates the hexagonal slice which scales with extent and pixels.
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

    def _set_filter(self):
        self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1, padding=0)
        self.conv.weight.data /= self.conv.weight.data
        self.conv.bias.data.fill_(0)  # if not self.requires_grad else None
        self.conv.weight.requires_grad = False  # self.requires_grad
        self.conv.bias.requires_grad = False  # self.requires_grad

    def __call__(self, sequence, ftype="mean", hex_sample=True):
        """Applies a box kernel to all frames in a sequence.

        Arguments:
            sequence (torch.Tensor): shape (samples, frames, height, width)
            ftype: filter type, 'mean' or 'median'.
        """
        samples, frames, height, width = sequence.shape

        if not isinstance(sequence, torch.cuda.FloatTensor):
            sequence = torch.Tensor(sequence)

        if (self.min_frame_size > torch.tensor([height, width])).any():
            sequence = ttf.resize(sequence, self.min_frame_size.tolist())

        if ftype == "mean":

            sequence = F.pad(sequence, self.pad)

            def conv(x):
                return self.conv(x.unsqueeze(1))

            out = torch.cat(tuple(map(conv, torch.unbind(sequence, dim=0))), dim=0)

            out /= self.kernel_size**2

        elif ftype == "sum":

            sequence = F.pad(sequence, self.pad)

            def conv(x):
                return self.conv(x.unsqueeze(1))

            out = torch.cat(tuple(map(conv, torch.unbind(sequence, dim=0))), dim=0)

        elif ftype == "median":
            out = median(sequence, self.kernel_size)

        else:
            raise ValueError("ftype must be 'sum', 'mean', or 'median." f"Is {ftype}.")

        if hex_sample is True:
            out = self.hex_sample(out)
            return out.reshape(samples, frames, -1)

        return out.reshape(samples, frames, height, width)

    def hex_sample(self, sequence):
        h, w = sequence.shape[-2:]
        c = self.receptor_centers + torch.tensor([h // 2, w // 2])
        out = sequence[:, :, c[:, 0], c[:, 1]]
        return out

    def sample(self, img, ftype="mean"):
        """Sample individual frames."""
        _type = np.asarray if isinstance(img, np.ndarray) else torch.Tensor
        _device = "cpu" if not isinstance(img, torch.cuda.FloatTensor) else "cuda"
        return _type(
            self(img[None, None, :, :], ftype=ftype, hex_sample=True)
            .squeeze()
            .to(_device)
        )

    def illustrate(self):
        return NotImplemented
        import matplotlib.pyplot as plt

        x_hc, y_hc, (dist_w, dist_h) = hex_center_coordinates(
            self.n_ommatidia, self.monitor_width_px, self.monitor_height_px
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

        vertices, _ = is_inside_hex(
            torch.tensor(y_img, dtype=self.dtype),
            torch.tensor(x_img, dtype=self.dtype),
            torch.tensor(x_hc, dtype=self.dtype),
            torch.tensor(y_hc, dtype=self.dtype),
            torch.tensor(dist_to_edge, dtype=self.dtype),
            torch.tensor(np.radians(0), dtype=self.dtype),
        )
        fig = plt.figure(
            figsize=[
                12
                * self.monitor_width_px
                / max(self.monitor_width_px, self.monitor_height_px),
                12
                * self.monitor_height_px
                / max(self.monitor_width_px, self.monitor_height_px),
            ]
        )
        plt.scatter(x_hc, y_hc, color="#eb4034", zorder=1)
        plt.scatter(x_img, y_img, color="#34ebd6", s=0.5, zorder=0)

        for h in range(self.n_ommatidia):
            for i in range(6):
                x1, y1 = vertices[i, :, h]  # x1, y1: (n_hexagons)
                x2, y2 = vertices[i + 1, :, h]
                plt.plot([x1, x2], [y1, y2], c="black")

        plt.xlim(0, self.monitor_width_px)
        plt.ylim(0, self.monitor_height_px)


# ----- HexEye (slower, more precise) ----------------------------------------


class HexEye:
    def __init__(
        self,
        n_ommatidia=721,
        ppo=25,
        monitor_height_px=None,
        monitor_width_px=None,
        device="cuda",
        dtype=torch.float16,
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

    def __call__(self, stim, mode="mean", n_chunks=1):
        """

        Args:
            stim ([type]): , (n_frames, pixels (monitor_height_px * monitor_width_px))
        """
        shape = stim.shape
        if mode not in ["mean", "median", "sum"]:
            raise ValueError
        if len(stim.shape) == 2:
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
            # mode is 'mean'
            return (stim[:, :, None] * self.is_inside).sum(dim=1) / self.is_inside.sum(
                dim=0
            )

        except RuntimeError:
            if n_chunks > shape[0]:
                raise ValueError

            if self.device != "cpu":
                torch.cuda.empty_cache()
            chunks = torch.chunk(stim, max(n_chunks, 1), dim=0)

            def map_fn(chunk):
                return self(chunk, mode=mode, n_chunks=n_chunks + 1)

            return torch.cat(tuple(map(map_fn, chunks)), dim=0)

    def bar(
        self,
        bar_width_rad,
        bar_height_rad,
        bar_loc_theta,
        bar_loc_phi,
        n_bars,
        bar_intensity,
        bg_intensity,
        moving_angle,  # rotation angle
        cartesian=False,
        mode="mean",
    ):

        bar_width_px = int(bar_width_rad / self.omm_width_rad * self.ppo)
        bar_height_px = int(bar_height_rad / self.omm_height_rad * self.ppo)
        bar_loc_horizontal_px = int(
            self.monitor_width_px * bar_loc_theta / np.radians(180)
        )
        bar_loc_vertical_px = int(
            self.monitor_height_px * bar_loc_phi / np.radians(180)
        )

        bar = cartesian_bars(
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

    def grating(
        self,
        period_rad,
        phase_rad,
        intensity,
        bg_intensity,
        moving_angle,
        width_rad=None,
        height_rad=None,
        cartesian=False,
        mode="mean",
    ):
        """period = 1/ spatial frequency"""
        period_px = int(period_rad / self.omm_width_rad * self.ppo)
        phase_px = int(phase_rad / self.omm_width_rad * self.ppo)

        height_rad_px = None
        if height_rad:
            height_rad_px = int(height_rad / self.omm_height_rad * self.ppo)

        width_rad_px = None
        if width_rad:
            width_rad_px = int(width_rad / self.omm_width_rad * self.ppo)

        grating = cartesian_gratings(
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

    def grating_offsets(
        self,
        period_rad,
        intensity,
        bg_intensity,
        moving_angle,
        width_rad=None,
        height_rad=None,
        cartesian=False,
        mode="mean",
    ):

        dphase_px = np.radians(
            5.8 / 2
        )  # half ommatidia width - corresponds to led width of 2.25 degree
        n_offsets = np.ceil(period_rad / dphase_px).astype(int)
        gratings = []
        for offset in range(n_offsets):
            gratings.append(
                self.grating(
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

    def wn_bars(self, frames=1, moving_angle=0, cartesian=False, mode="mean"):

        bars = []
        for frame in range(frames):
            bars.append(
                cartesian_wn_bars(
                    self.monitor_height_px,
                    self.monitor_width_px,
                    rotate=moving_angle,
                )
            )
        bars = np.concatenate(bars)

        if cartesian:
            return bars
        return self(torch.tensor(bars.reshape(frames, -1), device=self.device), mode)

    def offset_bars(
        self,
        bar_width_rad,
        bar_height_rad,
        n_bars,
        offsets,
        bar_intensity,
        bg_intensity,
        moving_angle,
        bar_loc_horizontal=np.radians(90),
        bar_loc_vertical=np.radians(90),
        mode="mean",
    ):
        """Returns offset bars.

        Args:
            bar_width_rad (float): width of bars in radians,
                                   e.g. np.radians(2.25).
            bar_height_rad (float): height of bars in radians,
                                    e.g. np.radians(20.25).
            n_bars (int): number of bars (in regular distance).
            offsets (List[int]): offsets of bars wrt. the center in radians.
            bar_intensity (float): intensity of the bar.
            bg_intensity (float): intensity of the background.
            moving_angle (float): moving angle in degree [0, 360). Orientation
                                  is perpendicular to that.

        Returns:
            tensor: moving bars, (#offsets, hexals)
        """
        flashes = []
        for i, offset in enumerate(offsets):
            flashes.append(
                self.bar(
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
        # breakpoint()
        return torch.cat(flashes, dim=0)

    def bar_movie(
        self,
        t_stim,
        dt,
        bar_width_rad,
        bar_height_rad,
        n_bars,
        offsets,
        bar_intensity,
        bg_intensity,
        moving_angle,
        t_pre=0.0,
        t_between=0.0,
        t_post=0.0,
        bar_loc_horizontal=np.radians(90),
        bar_loc_vertical=np.radians(90),
    ):
        """Generates moving bars.

        Args:
            t_stim (float): stimulus duration.
            dt (float): temporal resolution.
            bar_width_rad (float): width of bars in radians,
                                   e.g. np.radians(2.25).
            bar_height_rad (float): height of bars in radians,
                                    e.g. np.radians(20.25).
            n_bars (int): number of bars (in regular distance).
            offsets (List[int]): offsets of bars wrt. the center in radians.
            bar_intensity (float): intensity of the bar.
            bg_intensity (float): intensity of the background.
            moving_angle (float): moving angle in degree [0, 360). Orientation
                                  is perpendicular to that.
            t_pre (float, optional): grey pre stimulus duration. Defaults to 0.
            t_between (float, optional): grey between offset stimulus duration.
                                         Defaults to 0.
            t_post (float, optional): grey post stimulus duration.
                                      Defaults to 0.

        Returns:
            tensor: moving bars, (timesteps, hexals)
        """
        pre_frames = round(t_pre / dt)
        stim_frames = round(t_stim / (len(offsets) * dt))
        if stim_frames == 0:
            raise ValueError(
                f"stimulus time {t_stim}s not sufficient to sample {len(offsets)} offsets at {dt}s"
            )
        between_frames = round(t_between / dt)
        post_frames = round(t_post / dt)

        flashes = []
        if pre_frames:
            flashes.append(torch.ones([pre_frames, self.n_ommatidia]) * bg_intensity)

        for i, offset in enumerate(offsets):
            flash = self.bar(
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

    def illustrate(self):
        import matplotlib.pyplot as plt

        x_hc, y_hc, (dist_w, dist_h) = hex_center_coordinates(
            self.n_ommatidia, self.monitor_width_px, self.monitor_height_px
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

        vertices, _ = is_inside_hex(
            torch.tensor(y_img, dtype=self.dtype),
            torch.tensor(x_img, dtype=self.dtype),
            torch.tensor(x_hc, dtype=self.dtype),
            torch.tensor(y_hc, dtype=self.dtype),
            torch.tensor(dist_to_edge, dtype=self.dtype),
            torch.tensor(np.radians(0), dtype=self.dtype),
        )
        fig = plt.figure(
            figsize=[
                12
                * self.monitor_width_px
                / max(self.monitor_width_px, self.monitor_height_px),
                12
                * self.monitor_height_px
                / max(self.monitor_width_px, self.monitor_height_px),
            ]
        )
        plt.scatter(x_hc, y_hc, color="#eb4034", zorder=1)
        plt.scatter(x_img, y_img, color="#34ebd6", s=0.5, zorder=0)

        for h in range(self.n_ommatidia):
            for i in range(6):
                x1, y1 = vertices[i, :, h]  # x1, y1: (n_hexagons)
                x2, y2 = vertices[i + 1, :, h]
                plt.plot([x1, x2], [y1, y2], c="black")

        plt.xlim(0, self.monitor_width_px)
        plt.ylim(0, self.monitor_height_px)


# ----- Supporting definitions -------------------------------------------------


def median(x, kernel_size, stride=1, n_chunks=10):
    """Median image filter with reflected padding.

    x (array or tensor): (#samples, #frames, height, width).
                        First and second dimension are optional.
    kernel_size (int): size of the boxes.
    n_chunks (int): chunksize over #samples, #frames to process the data on the
        gpu if it runs out of memory.

    Note: On the gpu it creates a tensor of kernel_size ** 2 * prod(x.shape)
        elements which consumes a lot of memory.
        ~ 14 GB for 50 frames of extent 436, 1024 with kernel_size 13.
        In case of a RuntimeError due to memory the methods processes the data
        in chunks.
    """
    # Get padding so that the resulting tensor is of the same shape.
    p = max(kernel_size - 1, 0)
    p_floor = p // 2
    p_ceil = p - p_floor
    padding = (p_floor, p_ceil, p_floor, p_ceil)

    shape = x.shape

    if not isinstance(x, torch.cuda.FloatTensor):
        # to save time at library import
        from scipy.ndimage import median_filter

        # Process on cpu using scipy.ndimage.median_filter.
        _type = np.array
        if isinstance(x, torch.FloatTensor):
            x = x.numpy()
            _type = torch.FloatTensor
        x = x.reshape(-1, *x.shape[-2:])

        def map_fn(z):
            return median_filter(z, size=kernel_size)

        x = np.concatenate(tuple(map(map_fn, x)), axis=0)
        return _type(x.reshape(shape))

    # Process on gpu.
    try:
        with torch.no_grad():
            x.unsqueeze_(0).unsqueeze_(0) if len(shape) == 2 else x.unsqueeze_(
                0
            ) if len(shape) == 3 else None
            assert len(x.shape) == 4
            _x = F.pad(x, padding, mode="reflect")
            _x = _x.unfold(dimension=2, size=kernel_size, step=stride).unfold(
                dimension=3, size=kernel_size, step=stride
            )
            _x = _x.contiguous().view(shape[:4] + (-1,)).median(dim=-1)[0]
            return _x.view(shape)
    except RuntimeError:
        torch.cuda.empty_cache()
        _x = x.reshape(-1, *x.shape[-2:])
        chunks = torch.chunk(_x, max(n_chunks, 1), dim=0)

        def map_fn(z):
            return median(z, kernel_size, n_chunks=n_chunks - 1)

        _x = torch.cat(tuple(map(map_fn, chunks)), dim=0)
        return _x.view(shape)


def hex_center_coordinates(n_hex_area, img_width, img_height, center=True):

    # Horizontal extent of the grid
    n = np.floor(np.sqrt(n_hex_area / 3)).astype("int")

    dist_h = img_height / (2 * n + 1)
    dist_w = img_width / (2 * n + 1)

    xs = []
    ys = []
    for q in range(-n, n + 1):
        for r in range(max(-n, -n - q), min(n, n - q) + 1):
            xs.append(dist_w * r)
            ys.append(
                dist_h * (q + r / 2)
            )  # either must be negative or origin must be upper
    xs, ys = np.array(xs), np.array(ys)
    if center:
        xs += img_width // 2
        ys += img_height // 2
    return xs, ys, (dist_w, dist_h)


def is_inside_hex(
    x,
    y,
    x_centers,
    y_centers,
    dist_to_edge,
    tilt,
    device="cuda",
    dtype=torch.float16,
):
    """Finds whether the given points are inside the given hexagons.

    Args:
        x, y (tensors): Cartesian coordinates of the points, (n_points).
        x_centers, y_centers (tensors): Cartesian centers of the hexagon, (n_hexagons).
        dist_to_edge (float): Euclidian distance from center to edge of the hexagon.
        tilt (float or tensor): Angle of hexagon counter-clockwise tilt, radians.
    Returns:
        vertices (tensor): Cartesian coordinates of the hexagons' vertices,
            (7, 2, n_hexagons).
        is_inside (boolean tensor): Whether points inside or not,
            (n_points, n_hexagons).
    """

    if not isinstance(tilt, torch.Tensor):
        tilt = torch.tensor(tilt, device=device)

    R = torch.tensor(
        [
            [torch.cos(tilt), -torch.sin(tilt)],
            [torch.sin(tilt), torch.cos(tilt)],
        ],
        dtype=dtype,
        device=device,
    )  # rotation matrix
    pi = torch.tensor(np.pi, device=device, dtype=dtype)
    R60 = torch.tensor(
        [
            [torch.cos(pi / 3), -torch.sin(pi / 3)],
            [torch.sin(pi / 3), torch.cos(pi / 3)],
        ],
        dtype=dtype,
        device=device,
    )  # rotation matrix

    # Generate hexagon vertices
    dist_to_vertex = 2 / np.sqrt(3) * dist_to_edge
    vertices = torch.zeros(7, 2, dtype=dtype, device=device)
    vertices[0, :] = torch.matmul(
        R, torch.tensor([dist_to_vertex, 0], dtype=dtype, device=device)
    )
    for i in range(1, 7):
        vertices[i] = torch.matmul(R60, vertices[i - 1])
    vertices = vertices[:, :, None]
    vertices = torch.cat(
        (
            vertices[:, 0:1, :] + x_centers[None, None, :],
            vertices[:, 1:2, :] + y_centers[None, None, :],
        ),
        dim=1,
    )  # (7, 2, n_hexagons)

    # Generate is_inside output
    is_inside = torch.ones(len(x), len(x_centers), dtype=torch.bool, device=device)
    for i in range(6):
        x1, y1 = vertices[i, :, :]  # x1, y1: (n_hexagons)
        x2, y2 = vertices[i + 1, :, :]
        slope = (y2 - y1) / (x2 - x1)  # (n_hexagons)
        f_center = y1 + slope * (x_centers - x1) - y_centers  # (n_hexagons)
        f_points = (
            y1[None, :] + slope[None, :] * (x[:, None] - x1[None, :]) - y[:, None]
        )  # (n_points, n_hexagons)
        is_inside = torch.logical_and(is_inside, f_center.sign() == f_points.sign())

    return vertices, is_inside  # (7, 2, n_hexagons), (n_points, n_hexagons)


def vsplit(array, out_width, n_splits, center_crop_fraction=0.7):
    """Splits an array into n_splits overlapping splits along the last dimension.

    Args:
        array (np.ndarray, torch.Tensor): array of shape (..., width).
        out_width (int): width of output.
        n_splits (int): number of splits.
        center_crop_fraction (float): array will be cropped centrally in width to
            the fraction center_crop_fraction before being partitioned to capture
            more central content, because some sequences have mostly still
            background on the left and right, this avoids creating motion still
            sequences.

    Returns:
        tuple of n_splits arrays: ((..., out_width), ..., (..., out_width))
    """
    if center_crop_fraction is not None:
        return vsplit(
            vcenter_crop(array, center_crop_fraction),
            out_width,
            n_splits,
            center_crop_fraction=None,
        )

    actual_width = array.shape[-1]
    out_width = int(out_width)

    def take(arr, start, stop):
        if isinstance(arr, np.ndarray):
            return np.take(arr, np.arange(start, stop), axis=-1)[None]
        elif isinstance(arr, torch.Tensor):
            return torch.index_select(arr, dim=-1, index=torch.arange(start, stop))[
                None
            ]

    if n_splits == 1:
        out = (array[None, :],)
    elif n_splits > 1:
        _div = int(out_width / n_splits)
        out = ()
        out_width = max(out_width, int(actual_width / n_splits))
        overlap = np.ceil(
            (out_width * n_splits - actual_width) / (n_splits - 1)
        ).astype(int)
        for i in range(n_splits):
            start = i * out_width - i * overlap
            stop = (i + 1) * out_width - i * overlap
            out += (take(array, start, stop),)
    elif n_splits is None or n_splits == 0:
        return array
    else:
        raise ValueError
    if isinstance(array, np.ndarray):
        return np.concatenate(out, axis=0)
    elif isinstance(array, torch.Tensor):
        return torch.cat(out, dim=0)


def vcenter_crop(array, out_width_ratio):
    def take(arr, start, stop):
        if isinstance(arr, np.ndarray):
            return np.take(arr, np.arange(start, stop), axis=-1)
        elif isinstance(arr, torch.Tensor):
            return torch.index_select(arr, dim=-1, index=torch.arange(start, stop))

    width = array.shape[-1]
    out_width = int(out_width_ratio * width)
    return take(array, (width - out_width) // 2, (width + out_width) // 2)


def cartesian_bars(
    img_height_px,
    img_width_px,
    bar_width_px,
    bar_height_px,
    bar_loc_horizontal_px,
    bar_loc_vertical_px,
    n_bars,
    bar_intensity,
    bg_intensity,
    rotate=0,
):
    """
    All parameters in units of pixels.
    """

    bar_spacing = int(img_width_px / n_bars - bar_width_px)

    height_slice = slice(
        int(bar_loc_vertical_px - bar_height_px / 2),
        int(bar_loc_vertical_px + bar_height_px / 2) + 1,
    )

    img = np.ones([img_height_px, img_width_px]) * bg_intensity

    loc_w = int(bar_loc_horizontal_px - bar_width_px / 2)
    for i in range(n_bars):
        #  Fill background with bars.
        start = max(loc_w + i * bar_width_px + i * bar_spacing, 0)
        width_slice = slice(start, loc_w + (i + 1) * bar_width_px + i * bar_spacing + 1)
        img[height_slice, width_slice] = bar_intensity

    if rotate % 360 != 0:
        img = rotate_image(img, angle=rotate)

    return img


def cartesian_gratings(
    img_height_px,
    img_width_px,
    spatial_period_px,
    grating_intensity,
    bg_intensity,
    grating_height_px=None,
    grating_width_px=None,
    grating_phase_px=0,
    rotate=0,
):
    """
    All parameters in units of pixels.
    """
    # to save time at library import
    from scipy.signal import square

    t = (
        2
        * np.pi
        / (spatial_period_px / img_width_px)
        * (
            np.linspace(-1 / 2, 1 / 2, int(img_width_px))
            - grating_phase_px / img_width_px
        )
    )

    gratings = np.tile(square(t), img_height_px).reshape(img_height_px, img_width_px)
    gratings[gratings == -1] = bg_intensity
    gratings[gratings == 1] = grating_intensity

    if grating_height_px:
        mask = np.ones_like(gratings).astype(bool)

        height_slice = slice(
            int(img_height_px // 2 - grating_height_px / 2),
            int(img_height_px // 2 + grating_height_px / 2) + 1,
        )
        mask[height_slice] = False
        gratings[mask] = 0.5

    if grating_width_px:
        mask = np.ones_like(gratings).astype(bool)

        width_slice = slice(
            int(img_width_px // 2 - grating_width_px / 2),
            int(img_width_px // 2 + grating_width_px / 2) + 1,
        )
        mask[:, width_slice] = False
        gratings[mask] = 0.5

    if rotate % 360 != 0:
        gratings = rotate_image(gratings, angle=rotate)

    return gratings


def cartesian_wn_bars(img_height, img_width, rotate=0):
    """
    All parameters in units of pixels.
    """
    seq = np.zeros([img_height, img_width])

    seq[np.arange(img_height)] = np.random.normal(0.5, 0.25, size=[img_width])[None]

    if rotate % 360 != 0:
        seq = rotate_image(seq, angle=rotate)

    return seq


def rotate_image(img, angle=0):

    h, w = img.shape

    diagonal = int(np.sqrt(h**2 + w**2))

    pad_in_height = (diagonal - h) // 2
    pad_in_width = (diagonal - w) // 2

    img = np.pad(
        img,
        ((pad_in_height, pad_in_height), (pad_in_width, pad_in_width)),
        mode="edge",
    )

    img = Image.fromarray((255 * img).astype("uint8")).rotate(
        angle, Image.BILINEAR, False, None
    )
    img = np.array(img, dtype=float) / 255.0

    padded_h, padded_w = img.shape
    return img[
        pad_in_height : padded_h - pad_in_height,
        pad_in_width : padded_w - pad_in_width,
    ]


def resample(stims, t_stim, dt, dim=0, device="cuda"):
    """Resamples set of stims for given stimulus duration and dt.

    Args:
        stims (tensor): stims of shape (#conditions, #hexals).
        t_stim (float): stimulus duration in seconds.
        dt (float): integration time constant in seconds.

    Returns:
        tensor: stims of shape (#frames, #hexals).
    """
    n_offsets = stims.shape[dim]
    repeats = torch.linspace(0, n_offsets - 1, int(t_stim / dt), device=device).long()
    return torch.index_select(stims, dim, repeats)


def shuffle(stims, randomstate=None):
    """To randomly shuffle stims along the frame dimension.

    Args:
        stims (tensor): of shape (N (optional), #frames, #hexals)
    """
    if len(stims.shape) == 3:
        # assume (smples frames hexals)
        def _shuffle(x):
            return shuffle(x, randomstate)

        return torch.stack(list(map(_shuffle, stims)), dim=0)
    perms = (
        randomstate.permutation(stims.shape[0])
        if randomstate is not None
        else np.random.permutation(stims.shape[0])
    )
    return stims[perms]


def resample_grating(grating, t_stim, dt, temporal_frequency):
    n_frames = int(t_stim / dt)
    t_period = 1 / temporal_frequency
    _grating = resample(grating, t_period, dt)
    _grating = _grating.repeat(np.ceil(n_frames / _grating.shape[0]).astype(int), 1)
    return _grating[:n_frames]


def pad(stim, t_stim, dt, fill=0, mode="end", pad_mode="value"):
    """Pads second to last dimension.

    Args:
        stim (tensor): stimulus (..., n_frames, n_hexals)
    """
    diff = int(t_stim / dt) - stim.shape[-2]
    if diff <= 0:
        return stim

    if mode == "end":
        pad = (0, 0, 0, diff)
    elif mode == "start":
        pad = (0, 0, diff, 0)

    if pad_mode == "value":
        return torch.nn.functional.pad(stim, pad=pad, mode="constant", value=fill)
    elif pad_mode == "continue":
        return repeat_last(stim, -2, diff)
    else:
        return torch.nn.functional.pad(stim, pad=pad, mode=pad_mode)


def repeat_last(stim, dim, n_repeats):
    last = stim.index_select(
        dim, torch.tensor([stim.size(dim) - 1], device=stim.device)
    )
    stim = torch.cat((stim, last.repeat_interleave(n_repeats, dim=dim)), dim=dim)
    return stim
