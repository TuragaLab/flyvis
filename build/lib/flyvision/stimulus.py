"""Interface to control the cell-specific stimulus for the network."""
from typing import Dict, Union
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray

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

    Attributes:
        layer_index: dictionary of cell type to index array.
        central_cells_index: dictionary of cell type to central cell index.
        input_index: index array of photoreceptors.
        n_frames: number of frames in the stimulus buffer.
        n_samples: number of samples in the stimulus buffer.
        n_nodes: number of nodes in the stimulus buffer.
        n_input_elements: number of input elements.
        buffer: stimulus buffer of shape (n_samples, n_frames, n_cells).

    Returns:
        Tensor: stimulus of shape (n_samples, n_frames, n_cells)

    Example:
        stim = Stimulus(network.connectome, *x.shape[:2])
        stim.add_input(x)
        response = network(stim(), dt)
    """

    layer_index: Dict[str, NDArray]
    central_cells_index: Dict[str, int]
    input_index: NDArray
    n_frames: int
    n_samples: int
    n_nodes: int
    n_input_elements: int
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
        self.n_input_elements = self.input_index.shape[1]
        self.n_samples, self.n_frames, self.n_nodes = (
            n_samples,
            n_frames,
            len(connectome.nodes.type),
        )
        self.connectome = connectome
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
        if hasattr(self, "buffer") and self.buffer.shape[:2] == (
            self.n_samples,
            self.n_frames,
        ):
            self.buffer.zero_()
            return
        self.buffer = torch.zeros((self.n_samples, self.n_frames, self.n_nodes))
        self._nonzero = False

    @property
    def nonzero(self):
        """Returns True if elements have been added to the stimulus buffer.

        Note, even if those elements were all zero.
        """
        return self._nonzero

    def add_input(
        self,
        x: torch.Tensor,
        start=None,
        stop=None,
        n_frames_buffer=None,
        cumulate=False,
    ):
        """Adds input to the input/photoreceptor cells.

        Args:
            x: an input video of shape (n_samples, n_frames, 1, n_input_elements).
            start: optional temporal start index of the stimulus.
            stop: optional temporal stop index of the stimulus.
            n_frames_buffer: optional number of frames to resize the buffer to.
            cumulate: if True, add input to the existing buffer.
        """
        shape = x.shape
        if len(shape) != 4:
            raise ValueError(
                f"input has shape {x.shape} but must have "
                "(n_samples, n_frames, 1, n_input_elements)"
            )
        n_samples, n_frames_input = shape[:2]

        if not hasattr(self, "buffer"):
            self.zero(n_samples, n_frames_buffer or n_frames_input)
        elif not cumulate and self.nonzero:
            self.zero(n_samples, n_frames_buffer or n_frames_input)

        try:
            # add input to buffer
            self.buffer[:, slice(start, stop), self.input_index] += x.to(
                self.buffer.device
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"input has shape {x.shape} but buffer has shape {self.buffer.shape}"
            ) from e
        self._nonzero = True

    def add_pre_stim(
        self, x: torch.Tensor, start: int = None, stop: int = None, n_frames_buffer=None
    ):
        """Adds a constant or sequence of constants to the input/photoreceptor cells.

        Args:
            x: grey value(s). If Tensor, must have length `n_frames` or `stop - start`.
            start: start index in time. Defaults to None.
            stop: stop index in time. Defaults to None.
        """

        if not hasattr(self, "buffer"):
            self.zero(None, n_frames_buffer)
        elif self.nonzero:
            self.zero(None, n_frames_buffer)

        try:
            if isinstance(x, torch.Tensor) and x.ndim == 1:
                self.buffer[:, slice(start, stop), self.input_index] += x.view(
                    1, len(x), 1, 1
                )
            else:
                self.buffer[:, slice(start, stop), self.input_index] += x
        except RuntimeError as e:
            raise RuntimeError(
                f"input has shape {x.shape} but buffer has shape {self.buffer.shape}"
            ) from e
        self._nonzero = True

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
