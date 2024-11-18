"""Interface to control the cell-specific stimulus buffer for the network."""

from typing import Any, Callable, Dict, Optional, Protocol, Type, Union, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from flyvis.connectome import ConnectomeFromAvgFilters

__all__ = ["Stimulus"]


@runtime_checkable
class StimulusProtocol(Protocol):
    """Protocol for the Stimulus class."""

    def __call__(self) -> Tensor: ...
    def add_input(self, x: Tensor, **kwargs) -> None: ...
    def add_pre_stim(self, x: Tensor, **kwargs) -> None: ...
    def zero(self, **kwargs) -> None: ...
    def nonzero(self) -> bool: ...


AVAILABLE_STIMULI: Dict[str, Type[StimulusProtocol]] = {}


def register_stimulus(
    cls: Optional[Type[StimulusProtocol]] = None,
) -> Union[
    Callable[[Type[StimulusProtocol]], Type[StimulusProtocol]], Type[StimulusProtocol]
]:
    """Register a stimulus class.

    Args:
        cls: The stimulus class to register (optional when used as a decorator).

    Returns:
        Registered class or decorator function.

    Example:
        As a standalone function:
        ```python
        register_stimulus(CustomStimulus)
        ```

        As a decorator:
        ```python
        @register_stimulus
        class CustomStimulus(StimulusProtocol): ...
        ```
    """

    def decorator(cls: Type[StimulusProtocol]) -> Type[StimulusProtocol]:
        AVAILABLE_STIMULI[cls.__name__] = cls
        return cls

    if cls is None:
        return decorator
    else:
        return decorator(cls)


@register_stimulus
class Stimulus:
    """Interface to control the cell-specific stimulus buffer for the network.

    Creates a buffer and maps standard video input to the photoreceptors
    but can map input to any other cell as well, e.g. to do perturbation
    experiments.

    Args:
        connectome: Connectome directory to retrieve indexes for the stimulus
            buffer at the respective cell positions.
        n_samples: Number of samples to initialize the buffer with.
        n_frames: Number of frames to initialize the buffer with.
        init_buffer: If False, do not initialize the stimulus buffer.

    Attributes:
        layer_index (Dict[str, NDArray]): Dictionary of cell type to index array.
        central_cells_index (Dict[str, int]): Dictionary of cell type to
            central cell index.
        input_index (NDArray): Index array of photoreceptors.
        n_frames (int): Number of frames in the stimulus buffer.
        n_samples (int): Number of samples in the stimulus buffer.
        n_nodes (int): Number of nodes in the stimulus buffer.
        n_input_elements (int): Number of input elements.
        buffer (Tensor): Stimulus buffer of shape (n_samples, n_frames, n_cells).

    Returns:
        Tensor: Stimulus of shape (n_samples, n_frames, n_cells)

    Example:
        ```python
        stim = Stimulus(network.connectome, *x.shape[:2])
        stim.add_input(x)
        response = network(stim(), dt)
        ```
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
        connectome: ConnectomeFromAvgFilters,
        n_samples: int = 1,
        n_frames: int = 1,
        init_buffer: bool = True,
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
        self.input_index = np.array([
            self.layer_index[cell_type.decode()]
            for cell_type in connectome.input_cell_types[:]
        ])
        self.n_input_elements = self.input_index.shape[1]
        self.n_samples, self.n_frames, self.n_nodes = (
            n_samples,
            n_frames,
            len(connectome.nodes.type),
        )
        self.connectome = connectome
        if init_buffer:
            self.zero()

    def zero(
        self,
        n_samples: Optional[int] = None,
        n_frames: Optional[int] = None,
    ) -> None:
        """Reset the stimulus buffer to zero.

        Args:
            n_samples: Number of samples. If provided, the buffer will be resized.
            n_frames: Number of frames. If provided, the buffer will be resized.
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
    def nonzero(self) -> bool:
        """Check if elements have been added to the stimulus buffer.

        Returns:
            bool: True if elements have been added, even if those elements were all zero.
        """
        return self._nonzero

    def add_input(
        self,
        x: torch.Tensor,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        n_frames_buffer: Optional[int] = None,
        cumulate: bool = False,
    ) -> None:
        """Add input to the input/photoreceptor cells.

        Args:
            x: Input video of shape (n_samples, n_frames, 1, n_input_elements).
            start: Temporal start index of the stimulus.
            stop: Temporal stop index of the stimulus.
            n_frames_buffer: Number of frames to resize the buffer to.
            cumulate: If True, add input to the existing buffer.

        Raises:
            ValueError: If input shape is incorrect.
            RuntimeError: If input shape doesn't match buffer shape.
        """
        shape = x.shape
        if len(shape) != 4:
            raise ValueError(
                f"input has shape {x.shape} but must have "
                "(n_samples, n_frames, 1, n_input_elements)"
            )
        n_samples, n_frames_input = shape[:2]

        if not hasattr(self, "buffer") or not cumulate and self.nonzero:
            self.zero(n_samples, n_frames_buffer or n_frames_input)

        try:
            self.buffer[:, slice(start, stop), self.input_index] += x.to(
                self.buffer.device
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"input has shape {x.shape} but buffer has shape {self.buffer.shape}"
            ) from e
        self._nonzero = True

    def add_pre_stim(
        self,
        x: torch.Tensor,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        n_frames_buffer: Optional[int] = None,
    ) -> None:
        """Add a constant or sequence of constants to the input/photoreceptor cells.

        Args:
            x: Grey value(s). If Tensor, must have length `n_frames` or `stop - start`.
            start: Start index in time.
            stop: Stop index in time.
            n_frames_buffer: Number of frames to resize the buffer to.

        Raises:
            RuntimeError: If input shape doesn't match buffer shape.
        """
        if not hasattr(self, "buffer") or self.nonzero:
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
        """Return the stimulus tensor.

        Returns:
            torch.Tensor: The stimulus buffer.
        """
        return self.buffer


def is_stimulus_protocol(obj: Any) -> bool:
    return isinstance(obj, StimulusProtocol)


def init_stimulus(connectome: ConnectomeFromAvgFilters, **kwargs) -> StimulusProtocol:
    if "type" not in kwargs:
        return None
    stimulus_class = AVAILABLE_STIMULI[kwargs.pop("type")]
    return stimulus_class(connectome, **kwargs)
