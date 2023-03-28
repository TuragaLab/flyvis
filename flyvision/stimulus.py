from typing import Dict, Union
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor

from flyvision.connectome import ConnectomeDir


class Stimulus:
    """Interface to control the cell-specific stimulus for the network.

    Creates a buffer and e.g. maps standard video input to the photoreceptors
    but can map input to any other cell as well, e.g. to do perturbation
    experiments.

    Args:
        connectome: connectome directory to retrieve indexes for the stimulus
            buffer at the respective cell positions.
        n_samples: optional number of samples to initialize the buffer with.
            Defaults to 1. Else call `zero()` to resize the buffer.
        n_frames: optional number of frames to initialize the buffer with.
            Defaults to 1. Else call `zero()` to resize the buffer.
        _init: if False, do not initialize the stimulus buffer.

    Returns:
        Tensor: stimulus of shape (#samples, n_frames, #cells)

    Example:
        stim = Stimulus(network.connectome, *x.shape[:2])
        stim.add_input(x)
        response = network(stim(), dt)
    """

    layer_index: Dict[str, np.ndarray]
    central_cells_index: Dict[str, int]
    input_index: np.ndarray
    n_frames: int
    n_samples: int
    n_nodes: int
    buffer: Tensor

    def __init__(
        self,
        connectome: ConnectomeDir,
        n_samples: int = 1,
        n_frames: int = 1,
        _init=True,
    ):
        self.layer_index = {
            cell_type: index[:]
            for cell_type, index in connectome.nodes.layer_index.items()
        }
        self.central_cells_index = dict(
            zip(
                connectome.unique_cell_types[:].astype(str),
                connectome.central_cells_index[:],
            )
        )
        self.input_index = np.array(
            [
                self.layer_index[cell_type.decode()]
                for cell_type in connectome.input_cell_types[:]
            ]
        )
        self.n_samples, self.n_frames, self.n_nodes = (
            n_samples,
            n_frames,
            len(connectome.nodes.type),
        )
        if _init:
            self.zero()

    def zero(
        self,
        n_samples=None,
        n_frames=None,
    ):
        """Resets the stimulus buffer to zero.

        Args:
            n_samples: optional number of samples. If provided, the
                buffer will be resized.
            n_frames: optional number of frames. If provided, the
                buffer will be resized.
        """
        self.n_samples = n_samples or self.n_samples
        self.n_frames = n_frames or self.n_frames
        self.buffer = torch.zeros((self.n_samples, self.n_frames, self.n_nodes))

    def add_input(self, x: torch.Tensor, start=None, stop=None):
        """Adds input to the input/photoreceptor cells.

        Args:
            x: an input video of shape (#samples, n_frames, 1, n_hexals).
            start: optional temporal start index of the stimulus.
            stop: optional temporal stop index of the stimulus.
        """
        shape = x.shape
        if len(shape) != 4:
            raise ValueError(
                f"input has shape {x.shape} but must have (n_samples, n_frames, 1, n_hexals)"
            )

        n_samples, n_frames = shape[:2]
        if start is not None and stop is not None:
            n_frames = stop - start
        elif start is not None and stop is None:
            n_frames = self.n_frames - start
        elif start is None and stop is not None:
            n_frames = stop
        else:
            if n_frames != self.n_frames or n_samples != self.n_samples:
                self.zero(n_samples, n_frames)

        self.buffer[:, slice(start, stop), self.input_index] += x.to(
            self.buffer.device
        ).view(n_samples, n_frames, 1, x.shape[-1])

    def add_pre_stim(self, x: torch.Tensor, start: int = None, stop: int = None):
        """Adds a constant or sequence of constants to the input/photoreceptor cells.

        Args:
            x: grey value(s). If Tensor, must have length `n_frames` or `stop - start`.
            start: start index in time. Defaults to None.
            stop: stop index in time. Defaults to None.
        """
        if isinstance(x, torch.Tensor) and x.ndim != 0:
            self.buffer[:, slice(start, stop), self.input_index] += x.view(
                1, len(x), 1, 1
            )
        else:
            self.buffer[:, slice(start, stop), self.input_index] += x

    def add_layer_stim(self, cell_type: str, x: torch.Tensor):
        """Adds a stimulus to cells of a specified cell type.

        Args:
            cell_type: a cell type, e.g. "T4a"
            x: an input sequence of shape (#samples, n_frames, 1, n_hexals).
        """
        self.buffer[:, :, self.layer_index[cell_type]] += x

    def add_central_stim(self, cell_type: str, x: torch.Tensor):
        """Adds a stimulus to a central cell of a specified cell type.

        Args:
            cell_type: a cell type, e.g. "T4a"
            x: an input sequence of shape (#samples, n_frames).
        """
        self.buffer[:, :, self.central_cells_index[cell_type]] += x

    def add_layer_noise(self, cell_type: str, mean: float, std: float):
        """Adds gaussian noise to cells of a specified cell type.

        Args:
            cell_type: a cell type, e.g. "T4a"
            mean: mean of the gaussian noise.
            std: standard deviation of the gaussian noise.
        """
        noise = (
            torch.randn_like(self.buffer[:, :, self.layer_index[cell_type]]) * std
            + mean
        )
        self.buffer[:, :, self.layer_index[cell_type]] += noise

    def add_central_noise(self, cell_type: str, mean: float, std: float):
        """Adds gaussian noise to the central cell of a specified cell type.

        Args:
            cell_type: a cell type, e.g. "T4a"
            mean: mean of the gaussian noise.
            std: standard deviation of the gaussian noise.
        """
        noise = (
            torch.randn_like(self.buffer[:, :, self.central_cells_index[cell_type]])
            * std
            + mean
        )
        self.buffer[:, :, self.central_cells_index[cell_type]] += noise

    def suppress_layer(self, cell_type: str, value: float = -1e3):
        """Adds large negative inputs to cells of a specified cell type.

        Args:
            cell_type: a cell type, e.g. "T4a"
            value: negative input.
        """
        self.buffer[:, :, self.layer_index[cell_type]] += value

    def suppress_center(self, cell_type: str, value: float = -1e3):
        """Adds large negative inputs to the central cell of a specified cell type.

        Args:
            cell_type: a cell type, e.g. "T4a"
            value: negative input.
        """
        self.buffer[:, :, self.central_cells_index[cell_type]] += value

    def __call__(self) -> torch.Tensor:
        """Returns the buffer tensor."""
        return self.buffer

    @contextmanager
    def memory_friendly(self):
        """To remove the buffer temporarily to save GPU memory."""
        try:
            yield
        finally:
            delattr(self, "buffer")
