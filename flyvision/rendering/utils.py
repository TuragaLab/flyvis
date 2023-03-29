"""Rendering utils"""
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F


# ----- Supporting definitions -------------------------------------------------


def median(x, kernel_size, stride=1, n_chunks=10):
    """Median image filter with reflected padding.

    x (array or tensor): (n_samples, n_frames, height, width).
                        First and second dimension are optional.
    kernel_size (int): size of the boxes.
    n_chunks (int): chunksize over n_samples, n_frames to process the data on the
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
