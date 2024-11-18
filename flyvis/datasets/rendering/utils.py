"""Rendering utils"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import flyvis


def median(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    n_chunks: int = 10,
) -> torch.Tensor:
    """
    Apply median image filter with reflected padding.

    Args:
        x: Input array or tensor of shape (n_samples, n_frames, height, width).
           First and second dimensions are optional.
        kernel_size: Size of the filter kernel.
        stride: Stride for the filter operation.
        n_chunks: Number of chunks to process the data if memory is limited.
            Recursively increases the chunk size until the data fits in memory.

    Returns:
        Filtered array or tensor of the same shape as input.

    Note:
        On GPU, this creates a tensor of kernel_size ** 2 * prod(x.shape) elements,
        consuming significant memory (e.g., ~14 GB for 50 frames of 436x1024 with
        kernel_size 13). In case of a RuntimeError due to memory, the method
        processes the data in chunks.
    """
    # Get padding so that the resulting tensor is of the same shape.
    p = max(kernel_size - 1, 0)
    p_floor = p // 2
    p_ceil = p - p_floor
    padding = (p_floor, p_ceil, p_floor, p_ceil)

    shape = x.shape

    try:
        with torch.no_grad():
            if len(shape) == 2:
                x.unsqueeze_(0).unsqueeze_(0)
            elif len(shape) == 3:
                x.unsqueeze_(0)
            elif len(shape) == 4:
                pass
            else:
                raise ValueError(f"Invalid shape: {shape}")
            assert len(x.shape) == 4
            _x = F.pad(x, padding, mode="reflect")
            _x = _x.unfold(dimension=2, size=kernel_size, step=stride).unfold(
                dimension=3, size=kernel_size, step=stride
            )
            _x = _x.contiguous().view(shape[:4] + (-1,)).median(dim=-1)[0]
            return _x.view(shape)
    except RuntimeError as e:
        if "memory" not in str(e):
            raise e
        if "CUDA" in str(e):
            torch.cuda.empty_cache()
        _x = x.reshape(-1, *x.shape[-2:])
        chunks = torch.chunk(_x, max(n_chunks, 1), dim=0)

        def map_fn(z):
            return median(z, kernel_size, n_chunks=n_chunks - 1)

        _x = torch.cat(tuple(map(map_fn, chunks)), dim=0)
        return _x.view(shape)


def split(
    array: Union[np.ndarray, torch.Tensor],
    out_nelements: int,
    n_splits: int,
    center_crop_fraction: Optional[float] = 0.7,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Split an array into overlapping segments along the last dimension.

    Args:
        array: Input array of shape (..., nelements).
        out_nelements: Number of elements in each output split.
        n_splits: Number of splits to create.
        center_crop_fraction: If not None, the array is centrally cropped in the
            last dimension to this fraction before splitting.

    Returns:
        A new array of shape (n_splits, ..., out_nelements) containing the splits.

    Raises:
        ValueError: If n_splits is less than 0.
        TypeError: If the input array is neither a numpy array nor a torch tensor.

    Note:
        - If n_splits is 1, the entire array is returned (with an added dimension).
        - If n_splits is None or 0, the original array is returned unchanged.
        - Splits may overlap if out_nelements * n_splits > array.shape[-1].
    """
    assert isinstance(array, (np.ndarray, torch.Tensor))
    if center_crop_fraction is not None:
        return split(
            center_crop(array, center_crop_fraction),
            out_nelements,
            n_splits,
            center_crop_fraction=None,
        )

    actual_nelements = array.shape[-1]
    out_nelements = int(out_nelements)

    def take(
        arr: Union[np.ndarray, torch.Tensor], start: int, stop: int
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(arr, np.ndarray):
            return np.take(arr, np.arange(start, stop), axis=-1)[None]
        elif isinstance(arr, torch.Tensor):
            return torch.index_select(arr, dim=-1, index=torch.arange(start, stop))[None]

    if n_splits == 1:
        out = (array[None, :],)
    elif n_splits > 1:
        out = ()
        out_nelements = max(out_nelements, int(actual_nelements / n_splits))
        overlap = np.ceil(
            (out_nelements * n_splits - actual_nelements) / (n_splits - 1)
        ).astype(int)
        for i in range(n_splits):
            start = i * out_nelements - i * overlap
            stop = (i + 1) * out_nelements - i * overlap
            out += (take(array, start, stop),)
    elif n_splits is None or n_splits == 0:
        return array
    else:
        raise ValueError("n_splits must be a non-negative integer or None")

    if isinstance(array, np.ndarray):
        return np.concatenate(out, axis=0)
    elif isinstance(array, torch.Tensor):
        return torch.cat(out, dim=0)


def center_crop(
    array: Union[np.ndarray, torch.Tensor], out_nelements_ratio: float
) -> Union[np.ndarray, torch.Tensor]:
    """
    Centrally crop an array along the last dimension with given ratio.

    Args:
        array: Array of shape (..., nelements).
        out_nelements_ratio: Ratio of output elements to input elements.

    Returns:
        Cropped array of shape (..., out_nelements).
    """

    def take(arr, start, stop):
        if isinstance(arr, np.ndarray):
            return np.take(arr, np.arange(start, stop), axis=-1)
        elif isinstance(arr, torch.Tensor):
            return torch.index_select(arr, dim=-1, index=torch.arange(start, stop))

    nelements = array.shape[-1]
    out_nelements = int(out_nelements_ratio * nelements)
    return take(array, (nelements - out_nelements) // 2, (nelements + out_nelements) // 2)


def hex_center_coordinates(
    n_hex_area: int, img_width: int, img_height: int, center: bool = True
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Calculate hexagon center coordinates for a given area.

    Args:
        n_hex_area: Number of hexagons in the area.
        img_width: Width of the image.
        img_height: Height of the image.
        center: If True, center the hexagon grid in the image.

    Returns:
        Tuple containing x coordinates, y coordinates, and (dist_w, dist_h).
    """
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
    x: torch.Tensor,
    y: torch.Tensor,
    x_centers: torch.Tensor,
    y_centers: torch.Tensor,
    dist_to_edge: float,
    tilt: Union[float, torch.Tensor],
    device: torch.device = flyvis.device,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find whether given points are inside the given hexagons.

    Args:
        x: Cartesian x-coordinates of the points (n_points).
        y: Cartesian y-coordinates of the points (n_points).
        x_centers: Cartesian x-centers of the hexagons (n_hexagons).
        y_centers: Cartesian y-centers of the hexagons (n_hexagons).
        dist_to_edge: Euclidean distance from center to edge of the hexagon.
        tilt: Angle of hexagon counter-clockwise tilt in radians.
        device: Torch device to use for computations.
        dtype: Data type for torch tensors.

    Returns:
        Tuple containing:
        - vertices: Cartesian coordinates of the hexagons' vertices (7, 2, n_hexagons).
        - is_inside: Boolean tensor indicating whether points are inside
            (n_points, n_hexagons).

    Info: Credits
        Adapted from Roman Vaxenburg's original implementation.
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


def render_bars_cartesian(
    img_height_px: int,
    img_width_px: int,
    bar_width_px: int,
    bar_height_px: int,
    bar_loc_horizontal_px: int,
    bar_loc_vertical_px: int,
    n_bars: int,
    bar_intensity: float,
    bg_intensity: float,
    rotate: float = 0,
) -> np.ndarray:
    """
    Render bars in a cartesian coordinate system.

    Args:
        img_height_px: Height of the image in pixels.
        img_width_px: Width of the image in pixels.
        bar_width_px: Width of each bar in pixels.
        bar_height_px: Height of each bar in pixels.
        bar_loc_horizontal_px: Horizontal location of the bars in pixels.
        bar_loc_vertical_px: Vertical location of the bars in pixels.
        n_bars: Number of bars to generate.
        bar_intensity: Intensity of the bars.
        bg_intensity: Intensity of the background.
        rotate: Rotation angle in degrees.

    Returns:
        Generated image as a numpy array.
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


def render_gratings_cartesian(
    img_height_px: int,
    img_width_px: int,
    spatial_period_px: float,
    grating_intensity: float,
    bg_intensity: float,
    grating_height_px: Optional[int] = None,
    grating_width_px: Optional[int] = None,
    grating_phase_px: float = 0,
    rotate: float = 0,
) -> np.ndarray:
    """
    Render gratings in a cartesian coordinate system.

    Args:
        img_height_px: Height of the image in pixels.
        img_width_px: Width of the image in pixels.
        spatial_period_px: Spatial period of the gratings in pixels.
        grating_intensity: Intensity of the gratings.
        bg_intensity: Intensity of the background.
        grating_height_px: Height of the grating area in pixels.
        grating_width_px: Width of the grating area in pixels.
        grating_phase_px: Phase of the gratings in pixels.
        rotate: Rotation angle in degrees.

    Returns:
        Generated image as a numpy array.
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


def rotate_image(img: np.ndarray, angle: float = 0) -> np.ndarray:
    """
    Rotate an image by a given angle.

    Args:
        img: Input image as a numpy array.
        angle: Rotation angle in degrees.

    Returns:
        Rotated image as a numpy array.
    """
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


def resample(
    stims: torch.Tensor,
    t_stim: float,
    dt: float,
    dim: int = 0,
    device: torch.device = flyvis.device,
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Resample a set of stimuli for a given stimulus duration and time step.

    Args:
        stims: Stimuli tensor of shape (#conditions, #hexals).
        t_stim: Stimulus duration in seconds.
        dt: Integration time constant in seconds.
        dim: Dimension along which to resample.
        device: Torch device to use for computations.
        return_indices: If True, return the indices used for resampling.

    Returns:
        Resampled stimuli tensor of shape (#frames, #hexals), or a tuple of
        (resampled stimuli, indices) if return_indices is True.
    """
    n_offsets = stims.shape[dim]
    # round to nearest integer
    # this results in unequal counts of each frame usually by +-1
    indices = torch.linspace(0, n_offsets - 1, int(t_stim / dt), device=device).long()
    if not return_indices:
        return torch.index_select(stims, dim, indices)
    return torch.index_select(stims, dim, indices), indices


def shuffle(
    stims: torch.Tensor, randomstate: Optional[np.random.RandomState] = None
) -> torch.Tensor:
    """
    Randomly shuffle stimuli along the frame dimension.

    Args:
        stims: Stimuli tensor of shape (N (optional), #frames, #hexals).
        randomstate: Random state for reproducibility.

    Returns:
        Shuffled stimuli tensor.
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


def resample_grating(
    grating: torch.Tensor, t_stim: float, dt: float, temporal_frequency: float
) -> torch.Tensor:
    """
    Resample a grating stimulus for a given duration and temporal frequency.

    Args:
        grating: Input grating tensor.
        t_stim: Stimulus duration in seconds.
        dt: Time step in seconds.
        temporal_frequency: Temporal frequency of the grating.

    Returns:
        Resampled grating tensor.
    """
    n_frames = int(t_stim / dt)
    t_period = 1 / temporal_frequency
    _grating = resample(grating, t_period, dt)
    _grating = _grating.repeat(np.ceil(n_frames / _grating.shape[0]).astype(int), 1)
    return _grating[:n_frames]


def pad(
    stim: torch.Tensor,
    t_stim: float,
    dt: float,
    fill: float = 0,
    mode: Literal["end", "start"] = "end",
    pad_mode: Literal["value", "continue", "reflect"] = "value",
) -> torch.Tensor:
    """
    Pad the second to last dimension of a stimulus tensor.

    Args:
        stim: Stimulus tensor of shape (..., n_frames, n_hexals).
        t_stim: Target stimulus duration in seconds.
        dt: Integration time constant in seconds.
        fill: Value to fill with if pad_mode is "value".
        mode: Padding mode, either "end" or "start".
        pad_mode: Padding type, either "value", "continue", or "reflect".

    Returns:
        Padded stimulus tensor.
    """
    diff = int(t_stim / dt) - stim.shape[-2]
    if diff <= 0:
        return stim

    # Pad the second-to-last dimension (n_frames)
    # Format: (pad_last_dim_left, pad_last_dim_right,
    #          pad_second_to_last_dim_before, pad_second_to_last_dim_after)
    if mode == "end":
        pad = (0, 0, 0, diff)  # Pad after the existing frames
    elif mode == "start":
        pad = (0, 0, diff, 0)  # Pad before the existing frames

    if pad_mode == "value":
        return torch.nn.functional.pad(stim, pad=pad, mode="constant", value=fill)
    elif pad_mode == "continue":
        return repeat_last(stim, -2, diff)
    else:
        return torch.nn.functional.pad(stim, pad=pad, mode=pad_mode)


def repeat_last(stim: torch.Tensor, dim: int, n_repeats: int) -> torch.Tensor:
    """
    Repeat the last frame of a stimulus tensor along a specified dimension.

    Args:
        stim: Input stimulus tensor.
        dim: Dimension along which to repeat.
        n_repeats: Number of times to repeat the last frame.

    Returns:
        Stimulus tensor with the last frame repeated.
    """
    last = stim.index_select(dim, torch.tensor([stim.size(dim) - 1], device=stim.device))
    stim = torch.cat((stim, last.repeat_interleave(n_repeats, dim=dim)), dim=dim)
    return stim
