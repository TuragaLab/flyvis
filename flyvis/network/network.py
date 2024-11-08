# pylint: disable=dangerous-default-value
"""
Deep mechanistic network module.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datamate import Namespace, namespacify
from toolz import valmap
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from flyvis.connectome import (
    init_connectome,
)
from flyvis.datasets.datasets import SequenceDataset
from flyvis.utils.activity_utils import LayerActivity
from flyvis.utils.class_utils import forward_subclass
from flyvis.utils.dataset_utils import IndexSampler
from flyvis.utils.nn_utils import n_params, simulation
from flyvis.utils.tensor_utils import AutoDeref, RefTensor

from .dynamics import NetworkDynamics
from .initialization import Parameter
from .stimulus import init_stimulus

logger = logging.getLogger(__name__)

__all__ = ["Network"]


class Network(nn.Module):
    """A connectome-constrained network with nodes, edges, and dynamics.

    Args:
        connectome: Connectome configuration.
        dynamics: Network dynamics configuration.
        node_config: Node parameter configuration.
        edge_config: Edge parameter configuration.

    Attributes:
        connectome (Connectome): Connectome directory.
        dynamics (NetworkDynamics): Network dynamics.
        node_params (Namespace): Node parameters.
        edge_params (Namespace): Edge parameters.
        n_nodes (int): Number of nodes.
        n_edges (int): Number of edges.
        num_parameters (int): Number of parameters.
        config (Namespace): Config namespace.
        _source_indices (Tensor): Source indices.
        _target_indices (Tensor): Target indices.
        symmetry_config (Namespace): Symmetry config.
        clamp_config (Namespace): Clamp config.
        stimulus (Stimulus): Stimulus object.
        _state_hooks (tuple): State hooks.
    """

    def __init__(
        self,
        connectome: Dict[str, Any] = Namespace(
            type="ConnectomeFromAvgFilters",
            file="fib25-fib19_v2.2.json",
            extent=15,
            n_syn_fill=1,
        ),
        dynamics: Dict[str, Any] = Namespace(
            type="PPNeuronIGRSynapses", activation=Namespace(type="relu")
        ),
        node_config: Dict[str, Any] = Namespace(
            bias=Namespace(
                type="RestingPotential",
                groupby=["type"],
                initial_dist="Normal",
                mode="sample",
                requires_grad=True,
                mean=0.5,
                std=0.05,
                penalize=Namespace(activity=True),
                seed=0,
            ),
            time_const=Namespace(
                type="TimeConstant",
                groupby=["type"],
                initial_dist="Value",
                value=0.05,
                requires_grad=True,
            ),
        ),
        edge_config: Dict[str, Any] = Namespace(
            sign=Namespace(
                type="SynapseSign",
                initial_dist="Value",
                requires_grad=False,
                groupby=["source_type", "target_type"],
            ),
            syn_count=Namespace(
                type="SynapseCount",
                initial_dist="Lognormal",
                mode="mean",
                requires_grad=False,
                std=1.0,
                groupby=["source_type", "target_type", "dv", "du"],
            ),
            syn_strength=Namespace(
                type="SynapseCountScaling",
                initial_dist="Value",
                requires_grad=True,
                scale=0.01,
                clamp="non_negative",
                groupby=["source_type", "target_type"],
            ),
        ),
        stimulus_config: Dict[str, Any] = Namespace(type="Stimulus", init_buffer=False),
    ):
        super().__init__()

        # Prepare configs.
        connectome, dynamics, node_config, edge_config, stimulus_config, self.config = (
            self.prepare_configs(
                connectome, dynamics, node_config, edge_config, stimulus_config
            )
        )

        # Store the connectome, dynamics, and parameters.
        self.connectome = init_connectome(**connectome)
        self.dynamics = forward_subclass(NetworkDynamics, dynamics)

        # Load constant indices into memory.
        # Store source/target indices.
        self._source_indices = torch.tensor(self.connectome.edges.source_index[:])
        self._target_indices = torch.tensor(self.connectome.edges.target_index[:])

        self.n_nodes = len(self.connectome.nodes.index)
        self.n_edges = len(self.connectome.edges.source_index)

        # Optional way of parameter sharing is averaging at every call across
        # precomputed masks. This can be useful for e.g. symmetric electrical
        # compartments.
        # These masks are collected from Parameters into this namespace.
        self.symmetry_config = Namespace()  # type: Dict[str, List[torch.Tensor]]
        # Clamp configuration is collected from Parameter into this Namespace
        # for projected gradient descent.
        self.clamp_config = Namespace()

        # Construct node parameter sets.
        self.node_params = Namespace()
        for param_name, param_config in node_config.items():
            param = forward_subclass(
                Parameter,
                config={
                    "type": param_config.type,
                    "param_config": param_config,
                    "connectome": self.connectome,
                },
            )

            # register parameter to module
            self.register_parameter(f"nodes_{param_name}", param.raw_values)

            # creating index to map shared parameters onto all nodes,
            # sources, or targets
            param.readers = dict(
                nodes=param.indices,
                sources=param.indices[self._source_indices],
                targets=param.indices[self._target_indices],
            )
            self.node_params[param_name] = param

            # additional map to optional boolean masks to constrain
            # parameters (called in self.clamp)
            self.symmetry_config[f"nodes_{param_name}"] = getattr(
                param, "symmetry_masks", []
            )

            # additional map to optional clamp configuration to constrain
            # parameters (called in self.clamp)
            self.clamp_config[f"nodes_{param_name}"] = getattr(
                param_config, "clamp", None
            )

        # Construct edge parameter sets.
        self.edge_params = Namespace()
        for param_name, param_config in edge_config.items():
            param = forward_subclass(
                Parameter,
                config={
                    "type": param_config.type,
                    "param_config": param_config,
                    "connectome": self.connectome,
                },
            )

            self.register_parameter(f"edges_{param_name}", param.raw_values)

            # creating index to map shared parameters onto all edges
            param.readers = dict(edges=param.indices)

            self.edge_params[param_name] = param

            self.symmetry_config[f"edges_{param_name}"] = getattr(
                param, "symmetry_masks", []
            )

            self.clamp_config[f"edges_{param_name}"] = getattr(
                param_config, "clamp", None
            )

        self.num_parameters = n_params(self)
        self._state_hooks = tuple()

        self.stimulus = init_stimulus(self.connectome, **stimulus_config)

        logger.info("Initialized network with %s parameters.", self.num_parameters)

    def __repr__(self):
        return self.config.__repr__().replace("Namespace", "Network", 1)

    def prepare_configs(
        self, connectome, dynamics, node_config, edge_config, stimulus_config
    ):
        """Prepare configs for network initialization."""
        connectome = namespacify(connectome).deepcopy()
        dynamics = namespacify(dynamics).deepcopy()
        node_config = namespacify(node_config).deepcopy()
        edge_config = namespacify(edge_config).deepcopy()
        stimulus_config = namespacify(stimulus_config).deepcopy()
        config = Namespace(
            connectome=connectome,
            dynamics=dynamics,
            node_config=node_config,
            edge_config=edge_config,
            stimulus_config=stimulus_config,
        ).deepcopy()
        return connectome, dynamics, node_config, edge_config, stimulus_config, config

    def param_api(self) -> Dict[str, Dict[str, Tensor]]:
        """Param api for inspection.

        Returns:
            Parameter namespace for inspection.

        Note:
            This is not the same as the parameter api passed to the dynamics. This is a
            convenience function to inspect the parameters, but does not write derived
            parameters or sources and targets states.
        """
        # Construct the base parameter namespace.
        params = Namespace(
            nodes=Namespace(),
            edges=Namespace(),
            sources=Namespace(),
            targets=Namespace(),
        )
        for param_name, parameter in {
            **self.node_params,
            **self.edge_params,
        }.items():
            values = parameter.semantic_values
            for route, indices in parameter.readers.items():
                # route one of ("nodes", "sources", "target", "edges")
                params[route][param_name] = Namespace(parameter=values, indices=indices)
        return params

    def _param_api(self) -> AutoDeref[str, AutoDeref[str, RefTensor]]:
        """Returns params object passed to `dynamics`.

        Returns:
            Parameter namespace for dynamics.
        """
        # Construct the base parameter namespace.
        params = AutoDeref(
            nodes=AutoDeref(),
            edges=AutoDeref(),
            sources=AutoDeref(),
            targets=AutoDeref(),
        )
        for param_name, parameter in {
            **self.node_params,
            **self.edge_params,
        }.items():
            values = parameter.semantic_values
            for route, indices in parameter.readers.items():
                # route one of ("nodes", "sources", "target", "edges")
                params[route][param_name] = RefTensor(values, indices)
        # Add derived parameters.
        self.dynamics.write_derived_params(params)
        for k, v in params.nodes.items():
            if k not in params.sources:
                params.sources[k] = self._source_gather(v)
                params.targets[k] = self._target_gather(v)

        return params

    def _source_gather(self, x: Tensor) -> RefTensor:
        """Gathers source node states across edges.

        Args:
            x: Node-level activation, e.g., voltages. Shape is (n_nodes).

        Returns:
            Edge-level representation. Shape is (n_edges).

        Note:
            For edge-level access to target node states for elementwise operations.
            Called in _param_api and _state_api.
        """
        return RefTensor(x, self._source_indices)

    def _target_gather(self, x: Tensor) -> RefTensor:
        """Gathers target node states across edges.

        Args:
            x: Node-level activation, e.g., voltages. Shape is (n_nodes).

        Returns:
            Edge-level representation. Shape is (n_edges).

        Note:
            For edge-level access to target node states for elementwise operations.
            Called in _param_api and _state_api.
        """
        return RefTensor(x, self._target_indices)

    def target_sum(self, x: Tensor) -> Tensor:
        """Scatter sum operation creating target node states from inputs.

        Args:
            x: Edge inputs to targets, e.g., currents. Shape is (batch_size, n_edges).

        Returns:
            Node-level input. Shape is (batch_size, n_nodes).
        """
        result = torch.zeros((*x.shape[:-1], self.n_nodes))
        # signature: tensor.scatter_add_(dim, index, other)
        result.scatter_add_(
            -1,  # nodes dim
            self._target_indices.expand(  # view of index expanded over dims of x
                *x.shape
            ),
            x,
        )
        return result

    def _initial_state(
        self, params: AutoDeref[str, AutoDeref[str, RefTensor]], batch_size: int
    ) -> AutoDeref[str, AutoDeref[str, Union[Tensor, RefTensor]]]:
        """Compute the initial state, given the parameters and batch size.

        Args:
            params: Parameter namespace.
            batch_size: Batch size.

        Returns:
            Initial state namespace of node, edge, source, and target states.
        """
        # Initialize the network.
        state = AutoDeref(nodes=AutoDeref(), edges=AutoDeref())
        self.dynamics.write_initial_state(state, params)

        # Expand over batch dimension.
        for k, v in state.nodes.items():
            state.nodes[k] = v.expand(batch_size, *v.shape)
        for k, v in state.edges.items():
            state.edges[k] = v.expand(batch_size, *v.shape)

        return self._state_api(state)

    def _next_state(
        self,
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
        state: AutoDeref[str, AutoDeref[str, Union[Tensor, RefTensor]]],
        x_t: Tensor,
        dt: float,
    ) -> AutoDeref[str, AutoDeref[str, Union[Tensor, RefTensor]]]:
        """Compute the next state, given the current `state` and stimulus `x_t`.

        Args:
            params: Parameters.
            state: Current state.
            x_t: Stimulus at time t. Shape is (batch_size, n_nodes).
            dt: Time step.

        Returns:
            Next state namespace of node, edge, source, and target states.

        Note:
            Uses simple, elementwise Euler integration.
        """
        vel = AutoDeref(nodes=AutoDeref(), edges=AutoDeref())

        self.dynamics.write_state_velocity(
            vel, state, params, self.target_sum, x_t, dt=dt
        )

        next_state = AutoDeref(
            nodes=AutoDeref(**{
                k: state.nodes[k] + vel.nodes[k] * dt for k in state.nodes
            }),
            edges=AutoDeref(**{
                k: state.edges[k] + vel.edges[k] * dt for k in state.edges
            }),
        )

        return self._state_api(next_state)

    def _state_api(
        self, state: AutoDeref[str, AutoDeref[str, Union[Tensor, RefTensor]]]
    ) -> AutoDeref[str, AutoDeref[str, Union[Tensor, RefTensor]]]:
        """Populate sources and targets states from nodes states.

        Args:
            state: Current state.

        Returns:
            Updated state with populated sources and targets.

        Note:
            Optional state hooks are called here (in order of registration).
            This is returned by _initial_state and _next_state.
        """
        for hook in self._state_hooks:
            _state = hook(state)
            if _state is not None:
                state = _state

        state = AutoDeref(
            nodes=state.nodes,
            edges=state.edges,
            sources=AutoDeref(**valmap(self._source_gather, state.nodes)),
            targets=AutoDeref(**valmap(self._target_gather, state.nodes)),
        )

        return state

    def register_state_hook(self, state_hook: Callable, **kwargs) -> None:
        """Register a state hook to retrieve or modify the state.

        Args:
            state_hook: Callable to be used as a hook.
            **kwargs: Keyword arguments to pass to the callable.

        Raises:
            ValueError: If state_hook is not callable.

        Note:
            The hook is called in _state_api. Useful for targeted perturbations.
        """

        class StateHook:
            def __init__(self, hook, **kwargs):
                self.hook = hook
                self.kwargs = kwargs or {}

            def __call__(self, state):
                return self.hook(state, **self.kwargs)

        if not isinstance(state_hook, Callable):
            raise ValueError("state_hook must be callable")

        self._state_hooks += (StateHook(state_hook, **kwargs),)

    def clear_state_hooks(self, clear: bool = True):
        """Clear all state hooks.

        Args:
            clear: If True, clear all state hooks.
        """
        if clear:
            self._state_hooks = tuple()

    def clamp(self):
        """Clamp free parameters to their range specified in their config.

        Valid configs are `non_negative` to clamp at zero and tuple of the form
        (min, max) to clamp to an arbitrary range.

        Note:
            This function also enforces symmetry constraints.
        """
        # clamp parameters
        for param_name, mode in self.clamp_config.items():
            param = getattr(self, param_name)
            if param.requires_grad:
                if mode is None:
                    pass
                elif mode == "non_negative":
                    param.data.clamp_(0)
                elif isinstance(mode, Iterable) and len(mode) == 2:
                    param.data.clamp_(*mode)
                else:
                    raise NotImplementedError(f"Clamping mode {mode} not implemented.")

        # enforce symmetry constraints
        for param_name, masks in self.symmetry_config.items():
            param = getattr(self, param_name)
            if param.requires_grad:
                for symmetry in masks:
                    param.data[symmetry] = param.data[symmetry].mean()

    def forward(
        self, x: Tensor, dt: float, state: AutoDeref = None, as_states: bool = False
    ) -> Union[torch.Tensor, AutoDeref]:
        """Forward pass of the network.

        Args:
            x: Whole-network stimulus of shape (batch_size, n_frames, n_cells).
            dt: Integration time constant.
            state: Initial state of the network. If not given, computed from
                NetworksDynamics.write_initial_state. initial_state and fade_in_state
                are convenience functions to compute initial steady states.
            as_states: If True, returns the states as List[AutoDeref], else concatenates
                the activity of the nodes and returns a tensor.

        Returns:
            Network activity or states.
        """
        # To keep the parameters within their valid domain, they get clamped.
        self.clamp()
        # Construct the parameter API.
        params = self._param_api()

        # Initialize the network state.
        if state is None:
            state = self._initial_state(params, x.shape[0])

        def handle(state):
            # loop over the temporal dimension for integration of dynamics
            for i in range(x.shape[1]):
                state = self._next_state(params, state, x[:, i], dt)
                if as_states is False:
                    yield state.nodes.activity
                else:
                    yield state

        if as_states is True:
            return list(handle(state))
        return torch.stack(list(handle(state)), dim=1)

    def steady_state(
        self,
        t_pre: float,
        dt: float,
        batch_size: int,
        value: float = 0.5,
        state: Optional[AutoDeref] = None,
        grad: bool = False,
        return_last: bool = True,
    ) -> AutoDeref:
        """Compute state after grey-scale stimulus.

        Args:
            t_pre: Time of the grey-scale stimulus.
            dt: Integration time constant.
            batch_size: Batch size.
            value: Value of the grey-scale stimulus.
            state: Initial state of the network. If not given, computed from
                NetworksDynamics.write_initial_state. initial_state and fade_in_state
                are convenience functions to compute initial steady states.
            grad: If True, the state is computed with gradient.
            return_last: If True, return only the last state.

        Returns:
            Steady state of the network after a grey-scale stimulus.
        """
        if t_pre is None or t_pre <= 0.0:
            return state

        if value is None:
            return state

        self.stimulus.zero(batch_size, int(t_pre / dt))
        self.stimulus.add_pre_stim(value)

        with self.enable_grad(grad):
            if return_last:
                return self(self.stimulus(), dt, as_states=True, state=state)[-1]
            return self(self.stimulus(), dt, as_states=True, state=state)

    def fade_in_state(
        self,
        t_fade_in: float,
        dt: float,
        initial_frames: Tensor,
        state: Optional[AutoDeref] = None,
        grad: bool = False,
    ) -> AutoDeref:
        """Compute state after fade-in stimulus of initial_frames.

        Args:
            t_fade_in: Time of the fade-in stimulus.
            dt: Integration time constant.
            initial_frames: Tensor of shape (batch_size, 1, n_input_elements).
            state: Initial state of the network. If not given, computed from
                NetworksDynamics.write_initial_state. initial_state and fade_in_state
                are convenience functions to compute initial steady states.
            grad: If True, the state is computed with gradient.

        Returns:
            State after fade-in stimulus.
        """
        if t_fade_in is None or t_fade_in <= 0.0:
            return state

        batch_size = initial_frames.shape[0]

        # replicate initial frame over int(t_fade_in/dt) frames and fade in
        # by ramping up the contrast
        self.stimulus.zero(batch_size, int(t_fade_in / dt))

        initial_frames = (
            torch.linspace(0, 1, int(t_fade_in / dt))[None, :, None]
            * (initial_frames.repeat(1, int(t_fade_in / dt), 1) - 0.5)
            + 0.5
        )
        self.stimulus.add_input(initial_frames[:, :, None])
        with self.enable_grad(grad):
            return self(self.stimulus(), dt, as_states=True, state=state)[-1]

    def simulate(
        self,
        movie_input: torch.Tensor,
        dt: float,
        initial_state: Union[AutoDeref, None, Literal["auto"]] = "auto",
        as_states: bool = False,
        as_layer_activity: bool = False,
    ) -> Union[torch.Tensor, AutoDeref, LayerActivity]:
        """Simulate the network activity from movie input.

        Args:
            movie_input: Tensor of shape (batch_size, n_frames, 1, hexals).
            dt: Integration time constant. Warns if dt > 1/50.
            initial_state: Network activity at the beginning of the simulation.
                Use fade_in_state or steady_state to compute the initial state from grey
                input or from ramping up the contrast of the first movie frame.
                Defaults to "auto", which uses the steady_state after 1s of grey input.
            as_states: If True, return the states as AutoDeref dictionary instead of
                a tensor. Defaults to False.
            as_layer_activity: If True, return a LayerActivity object. Defaults to False.
                Currently only supported for ConnectomeFromAvgFilters.

        Returns:
            Activity tensor of shape (batch_size, n_frames, #neurons),
            or AutoDeref dictionary if `as_states` is True,
            or LayerActivity object if `as_layer_activity` is True.

        Raises:
            ValueError: If the movie_input is not four-dimensional.
            ValueError: If the integration time step is bigger than 1/50.
            ValueError: If the network is not in evaluation mode or any
                parameters require grad.
        """
        if len(movie_input.shape) != 4:
            raise ValueError("requires shape (sample, frame, 1, hexals)")

        if (
            as_layer_activity
            and not self.connectome.__class__.__name__ == "ConnectomeFromAvgFilters"
        ):
            raise ValueError(
                "as_layer_activity is currently only supported for "
                "ConnectomeFromAvgFilters"
            )

        if dt > 1 / 50:
            warnings.warn(
                f"dt={dt} is very large for integration. "
                "Better choose a smaller dt (<= 1/50 to avoid this warning)",
                IntegrationWarning,
                stacklevel=2,
            )

        batch_size, n_frames = movie_input.shape[:2]
        if initial_state == "auto":
            initial_state = self.steady_state(1.0, dt, batch_size)
        with simulation(self):
            assert self.training is False and all(
                not p.requires_grad for p in self.parameters()
            )
            self.stimulus.zero(batch_size, n_frames)
            self.stimulus.add_input(movie_input)
            if as_layer_activity:
                return LayerActivity(
                    self.forward(self.stimulus(), dt, initial_state, as_states).cpu(),
                    self.connectome,
                    keepref=True,
                )
            return self.forward(self.stimulus(), dt, initial_state, as_states)

    @contextmanager
    def enable_grad(self, grad: bool = True):
        """Context manager to enable or disable gradient computation.

        Args:
            grad: If True, enable gradient computation.
        """
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(grad)
        try:
            yield
        finally:
            torch.set_grad_enabled(prev)

    def stimulus_response(
        self,
        stim_dataset: SequenceDataset,
        dt: float,
        indices: Optional[Iterable[int]] = None,
        t_pre: float = 1.0,
        t_fade_in: float = 0.0,
        grad: bool = False,
        default_stim_key: Any = "lum",
        batch_size: int = 1,
    ):
        """Compute stimulus responses for a given stimulus dataset.

        Args:
            stim_dataset: Stimulus dataset.
            dt: Integration time constant.
            indices: Indices of the stimuli to compute the response for.
                If not given, all stimuli responses are computed.
            t_pre: Time of the grey-scale stimulus.
            t_fade_in: Time of the fade-in stimulus (slow).
            grad: If True, the state is computed with gradient.
            default_stim_key: Key of the stimulus in the dataset if it returns
                a dictionary.
            batch_size: Batch size for processing.

        Note:
            Per default, applies a grey-scale stimulus for 1 second, no
            fade-in stimulus.

        Yields:
            Tuple of (stimulus, response) as numpy arrays.
        """
        stim_dataset.dt = dt
        if indices is None:
            indices = np.arange(len(stim_dataset))
        stim_loader = DataLoader(
            stim_dataset, batch_size=batch_size, sampler=IndexSampler(indices)
        )

        stimulus = self.stimulus

        # compute initial state
        initial_state = self.steady_state(t_pre, dt, batch_size=1, value=0.5)

        with self.enable_grad(grad):
            logger.info("Computing %s stimulus responses.", len(indices))
            for stim in tqdm(
                stim_loader, desc="Batch", total=len(stim_loader), leave=False
            ):
                # when datasets return dictionaries, we assume that the stimulus
                # is stored under the key `default_stim_key`
                if isinstance(stim, dict):
                    stim = stim[default_stim_key]  # (batch, frames, 1, hexals)
                else:
                    stim = stim.unsqueeze(-2)  # (batch, frames, 1, hexals)

                # fade in stimulus
                fade_in_state = self.fade_in_state(
                    t_fade_in=t_fade_in,
                    dt=dt,
                    initial_frames=stim[:, 0],
                    state=initial_state,
                )

                def handle_stim(stim, fade_in_state):
                    # reset stimulus
                    batch_size, n_frames = stim.shape[:2]
                    stimulus.zero(batch_size, n_frames)

                    # add stimulus
                    stimulus.add_input(stim)

                    # compute response
                    if grad is False:
                        return (
                            stim.cpu().numpy(),
                            self(stimulus(), dt, state=fade_in_state)
                            .detach()
                            .cpu()
                            .numpy(),
                        )
                    elif grad is True:
                        return (
                            stim.cpu().numpy(),
                            self(stimulus(), dt, state=fade_in_state),
                        )

                yield handle_stim(stim, fade_in_state)

    def current_response(
        self,
        stim_dataset: SequenceDataset,
        dt: float,
        indices: Optional[Iterable[int]] = None,
        t_pre: float = 1.0,
        t_fade_in: float = 0,
        default_stim_key: Any = "lum",
    ):
        """Compute stimulus currents and responses for a given stimulus dataset.

        Note:
            Requires Dynamics to implement `currents`.

        Args:
            stim_dataset: Stimulus dataset.
            dt: Integration time constant.
            indices: Indices of the stimuli to compute the response for.
                If not given, all stimuli responses are computed.
            t_pre: Time of the grey-scale stimulus.
            t_fade_in: Time of the fade-in stimulus (slow).
            default_stim_key: Key of the stimulus in the dataset if it returns
                a dictionary.

        Yields:
            Tuple of (stimulus, activity, currents) as numpy arrays.
        """
        self.clamp()
        # Construct the parameter API.
        params = self._param_api()

        stim_dataset.dt = dt
        if indices is None:
            indices = np.arange(len(stim_dataset))
        stim_loader = DataLoader(
            stim_dataset, batch_size=1, sampler=IndexSampler(indices)
        )

        stimulus = self.stimulus
        initial_state = self.steady_state(t_pre, dt, batch_size=1, value=0.5)
        with torch.no_grad():
            logger.info("Computing %d stimulus responses.", len(indices))
            for stim in stim_loader:
                if isinstance(stim, dict):
                    stim = stim[default_stim_key].squeeze(-2)

                fade_in_state = self.fade_in_state(
                    t_fade_in=t_fade_in,
                    dt=dt,
                    initial_frames=stim[:, 0].unsqueeze(1),
                    state=initial_state,
                )

                def handle_stim(stim, fade_in_state):
                    # reset stimulus
                    batch_size, n_frames, _ = stim.shape
                    stimulus.zero(batch_size, n_frames)

                    # add stimulus
                    stimulus.add_input(stim.unsqueeze(2))

                    # compute response
                    states = self(stimulus(), dt, state=fade_in_state, as_states=True)
                    return (
                        stim.cpu().numpy().squeeze(),
                        torch.stack(
                            [s.nodes.activity.cpu() for s in states],
                            dim=1,
                        )
                        .numpy()
                        .squeeze(),
                        torch.stack(
                            [self.dynamics.currents(s, params).cpu() for s in states],
                            dim=1,
                        )
                        .numpy()
                        .squeeze(),
                    )

                # stim, activity, currents
                yield handle_stim(stim, fade_in_state)


class IntegrationWarning(Warning):
    """Warning for integration-related issues."""
