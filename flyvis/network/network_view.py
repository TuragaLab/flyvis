# pylint: disable=dangerous-default-value
"""
Deep mechanistic network module.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import wraps
from os import PathLike
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Union

import torch.nn as nn
import xarray as xr
from cachetools import FIFOCache
from datamate import Directory, set_root_context
from joblib import Memory

import flyvis
from flyvis.analysis import (
    optimal_stimuli,
    stimulus_responses,
    stimulus_responses_currents,
)
from flyvis.connectome import (
    ConnectomeView,
    get_avgfilt_connectome,
)
from flyvis.task.tasks import init_decoder
from flyvis.utils.cache_utils import context_aware_cache, make_hashable
from flyvis.utils.chkpt_utils import (
    best_checkpoint_default_fn,
    recover_decoder,
    recover_network,
    resolve_checkpoints,
)

from .directories import NetworkDir
from .network import Network

logger = logging.getLogger(__name__)

__all__ = ["NetworkView", "CheckpointedNetwork"]


class NetworkView:
    """IO interface for network.

    Args:
        network_dir: Directory of the network.
        network_class: Network class. Defaults to Network.
        root_dir: Root directory. Defaults to flyvis.results_dir.
        connectome_getter: Function to get the connectome.
            Defaults to flyvision_connectome.
        checkpoint_mapper: Function to map checkpoints. Defaults to resolve_checkpoints.
        best_checkpoint_fn: Function to get the best checkpoint. Defaults to
            best_checkpoint_default_fn.
        best_checkpoint_fn_kwargs: Keyword arguments for best_checkpoint_fn. Defaults to
            {"validation_subdir": "validation", "loss_file_name": "loss"}.
        recover_fn: Function to recover the network. Defaults to recover_network.

    Attributes:
        network_class (nn.Module): Network class.
        dir (Directory): Network directory.
        name (str): Network name.
        root_dir (PathLike): Root directory.
        connectome_getter (Callable): Function to get the connectome.
        checkpoint_mapper (Callable): Function to map checkpoints.
        connectome_view (ConnectomeView): Connectome view.
        connectome (Directory): Connectome directory.
        checkpoints: Mapped checkpoints.
        memory (Memory): Joblib memory cache.
        best_checkpoint_fn (Callable): Function to get the best checkpoint.
        best_checkpoint_fn_kwargs (dict): Keyword arguments for best_checkpoint_fn.
        recover_fn (Callable): Function to recover the network.
        _network (CheckpointedNetwork): Checkpointed network instance.
        decoder: Decoder instance.
        _initialized (dict): Initialization status for network and decoder.
        cache (FIFOCache): Cache for storing results.
    """

    def __init__(
        self,
        network_dir: Union[str, PathLike, NetworkDir],
        network_class: nn.Module = Network,
        root_dir: PathLike = flyvis.results_dir,
        connectome_getter: Callable = get_avgfilt_connectome,
        checkpoint_mapper: Callable = resolve_checkpoints,
        best_checkpoint_fn: Callable = best_checkpoint_default_fn,
        best_checkpoint_fn_kwargs: dict = {
            "validation_subdir": "validation",
            "loss_file_name": "epe",
        },
        recover_fn: Callable = recover_network,
    ):
        self.network_class = network_class
        self.dir, self.name = self._resolve_dir(network_dir, root_dir)
        self.root_dir = root_dir
        self.connectome_getter = connectome_getter
        self.checkpoint_mapper = checkpoint_mapper
        self.connectome_view: ConnectomeView = connectome_getter(
            self.dir.config.network.connectome
        )
        self.connectome = self.connectome_view.dir
        self.checkpoints = checkpoint_mapper(self.dir)
        self.memory = Memory(
            location=self.dir.path / "__cache__",
            backend="xarray_dataset_h5",
            verbose=0,
            # verbose=11,
        )
        self.best_checkpoint_fn = best_checkpoint_fn
        self.best_checkpoint_fn_kwargs = best_checkpoint_fn_kwargs
        self.recover_fn = recover_fn
        self._network_instance = None
        self.decoder = None
        self._initialized = {"network": None, "decoder": None}
        self.cache = FIFOCache(maxsize=3)
        logging.info("Initialized network view at %s", str(self.dir.path))

    @property
    def _network(self) -> CheckpointedNetwork:
        """Lazy init of CheckpointedNetwork because get_checkpoint can be slow."""
        if self._network_instance is None:
            self._network_instance = CheckpointedNetwork(
                self.network_class,
                self.dir.config.network.to_dict(),
                self.name,
                self.get_checkpoint("best"),
                self.recover_fn,
                network=None,
            )
        return self._network_instance

    @_network.setter
    def _network(self, value):
        """Setter for _network property."""
        self._network_instance = value

    def _clear_cache(self):
        """Clear the FIFO cache."""
        self.cache = self.cache.__class__(maxsize=self.cache.maxsize)

    def _clear_memory(self):
        """Clear the joblib memory cache."""
        self.memory.clear()

    # --- ConnectomeView API for static code analysis
    # pylint: disable=missing-function-docstring
    @wraps(ConnectomeView.connectivity_matrix)
    def connectivity_matrix(self, *args, **kwargs):
        return self.connectome_view.connectivity_matrix(*args, **kwargs)

    connectivity_matrix.__doc__ = ConnectomeView.connectivity_matrix.__doc__

    @wraps(ConnectomeView.network_layout)
    def network_layout(self, *args, **kwargs):
        return self.connectome_view.network_layout(*args, **kwargs)

    network_layout.__doc__ = ConnectomeView.network_layout.__doc__

    @wraps(ConnectomeView.hex_layout)
    def hex_layout(self, *args, **kwargs):
        return self.connectome_view.hex_layout(*args, **kwargs)

    hex_layout.__doc__ = ConnectomeView.hex_layout.__doc__

    @wraps(ConnectomeView.hex_layout_all)
    def hex_layout_all(self, *args, **kwargs):
        return self.connectome_view.hex_layout_all(*args, **kwargs)

    hex_layout_all.__doc__ = ConnectomeView.hex_layout_all.__doc__

    @wraps(ConnectomeView.get_uv)
    def get_uv(self, *args, **kwargs):
        return self.connectome_view.get_uv(*args, **kwargs)

    get_uv.__doc__ = ConnectomeView.get_uv.__doc__

    @wraps(ConnectomeView.sources_list)
    def sources_list(self, *args, **kwargs):
        return self.connectome_view.sources_list(*args, **kwargs)

    sources_list.__doc__ = ConnectomeView.sources_list.__doc__

    @wraps(ConnectomeView.targets_list)
    def targets_list(self, *args, **kwargs):
        return self.connectome_view.targets_list(*args, **kwargs)

    targets_list.__doc__ = ConnectomeView.targets_list.__doc__

    @wraps(ConnectomeView.receptive_field)
    def receptive_field(self, *args, **kwargs):
        return self.connectome_view.receptive_field(*args, **kwargs)

    receptive_field.__doc__ = ConnectomeView.receptive_field.__doc__

    @wraps(ConnectomeView.receptive_fields_grid)
    def receptive_fields_grid(self, *args, **kwargs):
        return self.connectome_view.receptive_fields_grid(*args, **kwargs)

    receptive_fields_grid.__doc__ = ConnectomeView.receptive_fields_grid.__doc__

    @wraps(ConnectomeView.projective_field)
    def projective_field(self, *args, **kwargs):
        return self.connectome_view.projective_field(*args, **kwargs)

    projective_field.__doc__ = ConnectomeView.projective_field.__doc__

    @wraps(ConnectomeView.projective_fields_grid)
    def projective_fields_grid(self, *args, **kwargs):
        return self.connectome_view.projective_fields_grid(*args, **kwargs)

    projective_fields_grid.__doc__ = ConnectomeView.projective_fields_grid.__doc__

    @wraps(ConnectomeView.receptive_fields_df)
    def receptive_fields_df(self, *args, **kwargs):
        return self.connectome_view.receptive_fields_df(*args, **kwargs)

    receptive_fields_df.__doc__ = ConnectomeView.receptive_fields_df.__doc__

    @wraps(ConnectomeView.receptive_fields_sum)
    def receptive_fields_sum(self, *args, **kwargs):
        return self.connectome_view.receptive_fields_sum(*args, **kwargs)

    receptive_fields_sum.__doc__ = ConnectomeView.receptive_fields_sum.__doc__

    @wraps(ConnectomeView.projective_fields_df)
    def projective_fields_df(self, *args, **kwargs):
        return self.connectome_view.projective_fields_df(*args, **kwargs)

    projective_fields_df.__doc__ = ConnectomeView.projective_fields_df.__doc__

    @wraps(ConnectomeView.projective_fields_sum)
    def projective_fields_sum(self, *args, **kwargs):
        return self.connectome_view.projective_fields_sum(*args, **kwargs)

    projective_fields_sum.__doc__ = ConnectomeView.projective_fields_sum.__doc__

    # --- own API

    def get_checkpoint(self, checkpoint="best"):
        """Return the best checkpoint index.

        Args:
            checkpoint: Checkpoint identifier. Defaults to "best".

        Returns:
            str: Path to the checkpoint.
        """
        try:
            if checkpoint == "best":
                return self.best_checkpoint_fn(
                    self.dir.path,
                    **self.best_checkpoint_fn_kwargs,
                )
            return self.checkpoints.paths[checkpoint]
        except FileNotFoundError:
            logger.warning("Checkpoint %s not found at %s", checkpoint, self.dir.path)
            return None

    def network(
        self, checkpoint="best", network: Optional[Any] = None, lazy=False
    ) -> CheckpointedNetwork:
        """Lazy loading of network instance.

        Args:
            checkpoint: Checkpoint identifier. Defaults to "best".
            network: Existing network instance to use. Defaults to None.
            lazy: If True, don't recover the network immediately. Defaults to False.

        Returns:
            CheckpointedNetwork: Checkpointed network instance.
        """
        self._network = CheckpointedNetwork(
            self.network_class,
            self.dir.config.network.to_dict(),
            self.name,
            self.get_checkpoint(checkpoint),
            self.recover_fn,
            network=network or self._network.network,
        )
        if self._network.network is not None and not lazy:
            self._network.recover()
        return self._network

    def init_network(self, checkpoint="best", network: Optional[Any] = None) -> Network:
        """Initialize the network.

        Args:
            checkpoint: Checkpoint identifier. Defaults to "best".
            network: Existing network instance to use. Defaults to None.

        Returns:
            Network: Initialized network instance.
        """
        checkpointed_network = self.network(checkpoint=checkpoint, network=network)

        if checkpointed_network.network is not None:
            return checkpointed_network.network
        checkpointed_network.init()
        return checkpointed_network.recover()

    def init_decoder(self, checkpoint="best", decoder=None):
        """Initialize the decoder.

        Args:
            checkpoint: Checkpoint identifier. Defaults to "best".
            decoder: Existing decoder instance to use. Defaults to None.

        Returns:
            Decoder: Initialized decoder instance.
        """
        checkpointed_network = self.network(checkpoint=checkpoint, lazy=True)
        if (
            self._initialized["decoder"] == checkpointed_network.checkpoint
            and decoder is None
        ):
            return self.decoder
        self.decoder = decoder or init_decoder(
            self.dir.config.task.decoder, self.connectome
        )
        recover_decoder(self.decoder, checkpointed_network.checkpoint)
        self._initialized["decoder"] = checkpointed_network.checkpoint
        return self.decoder

    def _resolve_dir(self, network_dir, root_dir):
        """Resolve the network directory.

        Args:
            network_dir: Network directory path or Directory instance.
            root_dir: Root directory path.

        Returns:
            tuple: (Directory, str) - Network directory and name.

        Raises:
            ValueError: If the directory is not a NetworkDir.
        """
        if isinstance(network_dir, (PathLike, str)):
            with set_root_context(root_dir):
                network_dir = Directory(network_dir)
        if not network_dir.config.type == "NetworkDir":
            raise ValueError(f"NetworkDir not found at {network_dir.path}.")
        name = os.path.sep.join(network_dir.path.parts[-3:])
        return network_dir, name

    # --- stimulus responses

    @wraps(stimulus_responses.flash_responses)
    @context_aware_cache
    def flash_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate flash responses."""
        return stimulus_responses.flash_responses(self, *args, **kwargs)

    @wraps(stimulus_responses.moving_edge_responses)
    @context_aware_cache
    def moving_edge_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate moving edge responses."""
        return stimulus_responses.moving_edge_responses(self, *args, **kwargs)

    @wraps(stimulus_responses_currents.moving_edge_currents)
    @context_aware_cache
    def moving_edge_currents(
        self, *args, **kwargs
    ) -> List[stimulus_responses_currents.ExperimentData]:
        """Generate moving edge currents."""
        return stimulus_responses_currents.moving_edge_currents(self, *args, **kwargs)

    @wraps(stimulus_responses.moving_bar_responses)
    @context_aware_cache
    def moving_bar_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate moving bar responses."""
        return stimulus_responses.moving_bar_responses(self, *args, **kwargs)

    @wraps(stimulus_responses.naturalistic_stimuli_responses)
    @context_aware_cache
    def naturalistic_stimuli_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate naturalistic stimuli responses."""
        return stimulus_responses.naturalistic_stimuli_responses(self, *args, **kwargs)

    @wraps(stimulus_responses.central_impulses_responses)
    @context_aware_cache
    def central_impulses_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate central ommatidium impulses responses."""
        return stimulus_responses.central_impulses_responses(self, *args, **kwargs)

    @wraps(stimulus_responses.spatial_impulses_responses)
    @context_aware_cache
    def spatial_impulses_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate spatial ommatidium impulses responses."""
        return stimulus_responses.spatial_impulses_responses(self, *args, **kwargs)

    @wraps(stimulus_responses.optimal_stimulus_responses)
    @context_aware_cache
    def optimal_stimulus_responses(
        self, cell_type, *args, **kwargs
    ) -> optimal_stimuli.RegularizedOptimalStimulus:
        """Generate optimal stimuli responses."""
        return stimulus_responses.optimal_stimulus_responses(
            self, cell_type, *args, **kwargs
        )


@dataclass
class CheckpointedNetwork:
    """A network representation with checkpoint that can be pickled.

    Attributes:
        network_class: Network class (e.g., flyvis.Network).
        config: Configuration for the network.
        name: Name of the network.
        checkpoint: Checkpoint path.
        recover_fn: Function to recover the network.
        network: Network instance to avoid reinitialization.
    """

    network_class: Any
    config: Dict
    name: str
    checkpoint: PathLike
    recover_fn: Any = recover_network
    network: Optional[Network] = None

    def init(self, eval: bool = True) -> Network:
        """Initialize the network.

        Args:
            eval: Whether to set the network in evaluation mode.

        Returns:
            The initialized network.
        """
        if self.network is None:
            self.network = self.network_class(**self.config)
        if eval:
            self.network.eval()
        return self.network

    def recover(self, checkpoint: Optional[PathLike] = None) -> Network:
        """Recover the network from the checkpoint.

        Args:
            checkpoint: Path to the checkpoint. If None, uses the default checkpoint.

        Returns:
            The recovered network.

        Note:
            Initializes the network if it hasn't been initialized yet.
        """
        if self.network is None:
            self.init()
        return self.recover_fn(self.network, checkpoint or self.checkpoint)

    def __repr__(self):
        return (
            f"CheckpointedNetwork(\n"
            f"    network_class={self.network_class.__name__},\n"
            f"    name='{self.name}',\n"
            f"    config={pformat(self.config, indent=4)},\n"
            f"    checkpoint='{os.path.basename(str(self.checkpoint))}'\n"
            f"    recover_fn={self.recover_fn.__name__}\n"
            f")"
        )

    def _hash_key(self):
        return (
            self.network_class,
            self.name,
            make_hashable(self.config),
            self.checkpoint,
            self.recover_fn.__name__,
        )

    def __hash__(self):
        return hash(self._hash_key())

    # Equality check based on hashable elements.
    def __eq__(self, other):
        if not isinstance(other, CheckpointedNetwork):
            return False
        return (
            self.network_class == other.network_class
            and self.name == other.name
            and make_hashable(self.config) == make_hashable(other.config)
            and self.checkpoint == other.checkpoint
            and self.recover_fn.__name__ == other.recover_fn.__name__
        )

    # Custom reduce method to make the object compatible with joblib's pickling.
    # This ensures the 'network' attribute is never pickled.
    # Return a tuple containing:
    # 1. A callable that will recreate the object (here, the class itself)
    # 2. The arguments required to recreate the object (excluding the network)
    # 3. The state, excluding the 'network' attribute
    def __reduce__(self):
        state = self.__dict__.copy()
        state["network"] = None  # Exclude the complex network from being pickled

        return (
            self.__class__,
            (
                self.network_class,
                self.config,
                self.checkpoint,
                self.recover_fn.__name__,
                None,
            ),
            state,
        )

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.network = None
