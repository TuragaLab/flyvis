"""
Deep mechanistic network module.
"""
from numbers import Number
from os import PathLike
from typing import Any, Dict, Iterable, List, Optional, Union, Callable
from contextlib import contextmanager

import warnings
import numpy as np
from toolz import valmap
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from datamate import Namespace, Directory

from flyvision.connectome import ConnectomeDir, ConnectomeView
import flyvision

# from flyvision.decoder import init_decoder
from flyvision.stimulus import Stimulus
from flyvision.initialization import Parameter
from flyvision.dynamics import NetworkDynamics
from flyvision.utils.activity_utils import LayerActivity
from flyvision.utils.nn_utils import n_params, simulation
from flyvision.utils.dataset_utils import IndexSampler
from flyvision.utils.tensor_utils import RefTensor, AutoDeref
from flyvision.datasets.base import SequenceDataset
import logging

logging = logging.getLogger()


class Network(nn.Module):
    """A connectome-constrained network with nodes, edges, and dynamics.

    Args:
        connectome: Connectome config.
        dynamics: Dynamics config.
        node_config: Node config.
        edge_config: Edge config.

    Attributes:
        connectome (ConnectomeDir): Connectome.
        dynamics (NetworkDynamics): Dynamics.
        node_params (Dict[str, Parameter]): Node parameters.
        edge_params (Dict[str, Parameter]): Edge parameters.
        n_nodes (int): Number of nodes.
        n_edges (int): Number of edges.
        num_parameters (int): Number of parameters.
        config (Namespace): Config namespace.
        input_indices (np.ndarray): Input indices.
        output_indices (np.ndarray): Output indices.
        _source_indices (np.ndarray): Source indices.
        _target_indices (np.ndarray): Target indices.
        symmetry_config (Dict[str, Dict[str, Tensor]]): Symmetry config.
        clamp_config (Dict[str, Dict[str, Tensor]]): Clamp config.
        stimulus (Stimulus): Stimulus.
        state_hooks (Namespace): State hook.
    """

    def __init__(
        self,
        connectome: Namespace = Namespace(
            file="fib25-fib19_v2.2.json", extent=15, n_syn_fill=1
        ),
        dynamics: Namespace = Namespace(
            type="PPNeuronIGRSynapses", activation=Namespace(type="relu")
        ),
        node_config: Namespace = Namespace(
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
        edge_config: Namespace = Namespace(
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
                scale_elec=0.01,
                scale_chem=0.01,
                clamp="non_negative",
                groupby=["source_type", "target_type"],
            ),
        ),
    ):
        super().__init__()

        # Call deecopy to alter passed configs without upstream effects
        connectome = connectome.deepcopy()
        dynamics = dynamics.deepcopy()
        node_config = node_config.deepcopy()
        edge_config = edge_config.deepcopy()
        self.config = Namespace(
            connectome=connectome,
            dynamics=dynamics,
            node_config=node_config,
            edge_config=edge_config,
        ).deepcopy()

        # Store the connectome, dynamics, and parameters.
        self.connectome = ConnectomeDir(connectome)
        self.cell_types = self.connectome.unique_cell_types[:].astype(str)
        self.dynamics = NetworkDynamics(dynamics)

        # Load constant indices into memory.
        # Store source/target indices.
        self._source_indices = torch.tensor(self.connectome.edges.source_index[:])
        self._target_indices = torch.tensor(self.connectome.edges.target_index[:])

        self.n_nodes = len(self.connectome.nodes.type)
        self.n_edges = len(self.connectome.edges.edge_type)

        # Optional way of parameter sharing is averaging at every call across
        # precomputed masks. This can be useful for e.g. symmetric electrical
        # compartments.
        # Theses masks are collected from Parameters into this namespace.
        self.symmetry_config = Namespace()  # type: Dict[str, List[torch.Tensor]]
        # Clamp configuration is collected from Parameter into this Namespace
        # for projected gradient descent.
        self.clamp_config = Namespace()

        # Construct node parameter sets.
        self.node_params = Namespace()
        for param_name, param_config in node_config.items():
            param = Parameter(param_config, self.connectome)

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
            # parameters (called in self._clamp)
            self.symmetry_config[f"nodes_{param_name}"] = getattr(
                param, "symmetry_masks", []
            )

            # additional map to optional clamp configuration to constrain
            # parameters (called in self._clamp)
            self.clamp_config[f"nodes_{param_name}"] = getattr(
                param_config, "clamp", None
            )

        # Construct edge parameter sets.
        self.edge_params = Namespace()
        for param_name, param_config in edge_config.items():
            param = Parameter(param_config, self.connectome)

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

        # Store chem/elec indices for electrical compartments specified by
        # the connectome.
        self._elec_indices = torch.tensor(
            np.nonzero(self.connectome.edges.edge_type[:] == b"elec")[0]
        ).long()
        self._chem_indices = torch.tensor(
            np.nonzero(self.connectome.edges.edge_type[:] == b"chem")[0]
        ).long()

        self.num_parameters = n_params(self)
        self._state_hooks = tuple()

        self.stimulus = Stimulus(self.connectome, _init=False)

        logging.info(f"Initialized network with {self.num_parameters} parameters.")

    def __repr__(self):
        return self.config.__repr__().replace("Namespace", "Network", 1)

    def param_api(self) -> Dict[str, Dict[str, Tensor]]:
        """Param api for inspection.

        Note, that this is not the same as the parameter api passed to the
        dynamics. This is a convenience function to inspect the parameters,
        but does not write derived parameters or sources and targets states.
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
        """Returns params object passed to `dynamics`."""
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
        self.dynamics.write_derived_params(
            params, chem_indices=self._chem_indices, elec_indices=self._elec_indices
        )
        for k, v in params.nodes.items():
            if k not in params.sources:
                params.sources[k] = self._source_gather(v)
                params.targets[k] = self._target_gather(v)

        return params

    # -- Scatter/gather operations -------------------------

    def _source_gather(self, x: Tensor) -> RefTensor:
        """Gathers source node states across edges.

        Args:
            x: abstractly node-level activation, e.g. voltages,
                Shape is (n_nodes).

        Returns:
            RefTensor of edge-level representation.
              Shape is (n_edges).

        Note, for edge-level access to target node states for elementwise
        operations.

        Called in _param_api and _state_api.
        """
        return RefTensor(x, self._source_indices)

    def _target_gather(self, x: Tensor) -> RefTensor:
        """Gathers target node states across edges.

        Args:
            x: abstractly node-level activation, e.g. voltages.
                Shape is (n_nodes).

        Returns:
            RefTensor of edge-level representation.
              Shape is (n_edges).

         Note, for edge-level access to target node states for elementwise
         operations.

        Called in _param_api and _state_api.
        """
        return RefTensor(x, self._target_indices)

    def target_sum(self, x: Tensor) -> Tensor:
        """Scatter sum operation creating target node states from inputs.

        Args:
            x: abstractly, edge inputs to targets, e.g. currents.
                Shape is (batch_size, n_edges).

        Returns:
            RefTensor of node-level input. Shape is (batch_size, n_nodes).
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

    # ------------------------------------------------------

    def _initial_state(
        self, params: AutoDeref[str, AutoDeref[str, RefTensor]], batch_size: int
    ) -> AutoDeref[str, AutoDeref[str, Union[Tensor, RefTensor]]]:
        """Compute the initial state, given the parameters and batch size.

        Args:
            params: parameter namespace.
            batch_size: batch size.

        Returns:
            initial_state: namespace of node, edge, source, and target states.
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
            params: parameters
            state: current state
            x_t: stimulus at time t. Shape is (batch_size, n_nodes).
            dt: time step.

        Returns:
            next_state: namespace of node, edge, source, and target states.

        Note: simple, elementwise Euler integration.
        """
        vel = AutoDeref(nodes=AutoDeref(), edges=AutoDeref())

        self.dynamics.write_state_velocity(
            vel, state, params, self.target_sum, x_t, dt=dt
        )

        next_state = AutoDeref(
            nodes=AutoDeref(
                **{k: state.nodes[k] + vel.nodes[k] * dt for k in state.nodes}
            ),
            edges=AutoDeref(
                **{k: state.edges[k] + vel.edges[k] * dt for k in state.edges}
            ),
        )

        return self._state_api(next_state)

    def _state_api(
        self, state: AutoDeref[str, AutoDeref[str, Union[Tensor, RefTensor]]]
    ) -> AutoDeref[str, AutoDeref[str, Union[Tensor, RefTensor]]]:
        """Populate sources and targets states from nodes states.

        Note, optional state hooks are called here (in order of registration).
        Note, this is returned by _initial_state and _next_state.
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

        E.g. for a targeted perturbation.

        Args:
            state_hook: provides the callable.
            kwargs: keyword arguments to pass to the callable.

        Note: the hook is called in _state_api.
        """

        class StateHook:
            def __init__(self, hook, **kwargs):
                self.hook = hook
                self.kwargs = kwargs or {}

            def __call__(self, state):
                return self.hook(state, **self.kwargs)

        if not isinstance(state_hook, Callable):
            raise ValueError

        self._state_hooks += (StateHook(state_hook, **kwargs),)

    def clear_state_hooks(self, clear=True):
        """Clear all state hooks."""
        if clear:
            self._state_hooks = tuple()

    def simulate(
        self,
        movie_input: torch.Tensor,
        dt: float,
        initial_state: Union[AutoDeref, None] = "auto",
        as_states: bool = False,
    ) -> Union[torch.Tensor, AutoDeref]:
        """Simulate the network activity from movie input.

        Args:
            movie_input: tensor requiring shape (batch_size, n_frames, 1, hexals)
            dt: integration time constant. Warns if dt > 1/50.
            initial_state: network activity at the beginning of the simulation.
                Either use fade_in_state or steady_state, to compute the
                initial state from grey input or from ramping up the contrast of
                the first movie frame. Defaults to "auto", which uses the
                steady_state after 1s of grey input.
            as_states: can return the states as AutoDeref dictionary instead of
                a tensor. Defaults to False.

        Returns:
            activity tensor of shape (batch_size, n_frames, #neurons)

        Raises:
            ValueError if the movie_input is not four-dimensional.
            ValueError if the integration time step is bigger than 1/50.
            ValueError if the network is not in evaluation mode or any
                parameters require grad.
        """

        if len(movie_input.shape) != 4:
            raise ValueError("requires shape (sample, frame, 1, hexals)")

        if dt > 1 / 50:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    (
                        f"dt={dt} is very large for integration."
                        " better choose a smaller dt (<= 1/50 to avoid this warning)"
                    ),
                    IntegrationWarning,
                    stacklevel=2,
                )

        batch_size, n_frames = movie_input.shape[:2]
        if initial_state == "auto":
            initial_state = self.steady_state(1.0, dt, batch_size)
        with simulation(self):
            assert self.training == False and all(
                not p.requires_grad for p in self.parameters()
            )
            self.stimulus.zero(batch_size, n_frames)
            self.stimulus.add_input(movie_input)
            return self.forward(self.stimulus(), dt, initial_state, as_states)

    def forward(
        self, x: Tensor, dt: float, state: AutoDeref = None, as_states: bool = False
    ) -> Union[torch.Tensor, AutoDeref]:
        """Forward pass of the network.

        Args:
            x: whole-network stimulus of shape (batch_size, n_frames, n_cells).
            dt: integration time constant.
            state: initial state of the network. If not given,
                computed from NetworksDynamics.write_initial_state.
                initial_state and fade_in_state are convenience functions to
                compute initial steady states.
            as_states: if True, returns the states as List[AutoDeref],
                else concatenates the activity of the nodes and returns a tensor.
        """
        # To keep the parameters within their valid domain, they get clamped.
        self._clamp()
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
        return_last=True,
    ) -> AutoDeref:
        """State after grey-scale stimulus.

        Args:
            t_pre: time of the grey-scale stimulus.
            dt: integration time constant.
            batch_size: batch size.
            value: value of the grey-scale stimulus.
            state: initial state of the network. If not given,
                computed from NetworksDynamics.write_initial_state.
                initial_state and fade_in_state are convenience functions to
                compute initial steady states.
            grad: if True, the state is computed with gradient.

        Returns:
            steady state of the network after a grey-scale stimulus.
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
        initial_frames,
        state=None,
        grad=False,
    ) -> AutoDeref:
        """State after fade-in stimulus of initial_frames.

        Args:
            t_fade_in: time of the fade-in stimulus.
            dt: integration time constant.
            initial_frames: tensor of shape (batch_size, 1, n_input_elements)
            state: initial state of the network. If not given,
                computed from NetworksDynamics.write_initial_state.
                initial_state and fade_in_state are convenience functions to
                compute initial steady states.
            grad: if True, the state is computed with gradient.


        initial_frames of shape (batch_size, 1, n_input_elements)
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

    def _clamp(self):
        """Clamp free parameters to their range specifid in their config.

        Valid configs are `non_negative` to clamp at zero and tuple of the form
        (min, max) to clamp to an arbitrary range.

        Note, this function also enforces symmetry constraints.
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

    @contextmanager
    def enable_grad(self, grad=True):
        """Context manager to enable or disable gradient computation."""
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
        indices: Iterable[int] = None,
        t_pre: float = 1.0,
        t_fade_in: float = 0.0,
        grad: bool = False,
        default_stim_key: Any = "lum",
    ):
        """Compute stimulus responses for a given stimulus dataset.

        Args:
            stim_dataset: stimulus dataset.
            dt: integration time constant.
            indices: indices of the stimuli to compute the response for.
                If not given, all stimuli responses are computed.
            t_pre: time of the grey-scale stimulus.
            t_fade_in: time of the fade-in stimulus (slow).
            grad: if True, the state is computed with gradient.
            default_stim_key: key of the stimulus in the dataset if it returns
                a dictionary.

        Note: per default, applies a grey-scale stimulus for 1 second, no
            fade-in stimulus.

        Returns:
            iterator over stimuli and respective responses as numpy
            arrays.
        """
        stim_dataset.dt = dt
        if indices is None:
            indices = np.arange(len(stim_dataset))
        stim_loader = DataLoader(
            stim_dataset, batch_size=1, sampler=IndexSampler(indices)
        )

        stimulus = self.stimulus

        # compute initial state
        initial_state = self.steady_state(t_pre, dt, batch_size=1, value=0.5)

        with self.enable_grad(grad):
            logging.info(f"Computing {len(indices)} stimulus responses.")
            for i, stim in enumerate(stim_loader):
                # when datasets return dictionaries, we assume that the stimulus
                # is stored under the key `default_stim_key`
                if isinstance(stim, dict):
                    stim = stim[default_stim_key].squeeze(-2)

                # fade in stimulus
                fade_in_state = self.fade_in_state(
                    t_fade_in=t_fade_in,
                    dt=dt,
                    initial_frames=stim[:, 0].unsqueeze(1),
                    state=initial_state,
                )

                def handle_stim():
                    # reset stimulus
                    batch_size, n_frames = stim.shape[:2]
                    stimulus.zero(batch_size, n_frames)

                    # add stimulus
                    stimulus.add_input(stim)

                    # compute response
                    with stimulus.memory_friendly():
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

                yield handle_stim()

    def current_response(
        self,
        stim_dataset: SequenceDataset,
        dt: float,
        indices: Iterable[int] = None,
        t_pre: float = 1.0,
        t_fade_in: float = 0,
        default_stim_key: Any = "lum",
    ):
        """Compute stimulus currents and responses for a given stimulus dataset.

        Note, requires Dynamics to implement `currents`.

        Args:
            stim_dataset: stimulus dataset.
            dt: integration time constant.
            indices: indices of the stimuli to compute the response for.
                If not given, all stimuli responses are computed.
            t_pre: time of the grey-scale stimulus.
            t_fade_in: time of the fade-in stimulus (slow).
            grad: if True, the state is computed with gradient.
            default_stim_key: key of the stimulus in the dataset if it returns
                a dictionary.

        Returns:
            iterator over stimuli, currents and respective responses as numpy
            arrays.
        """
        self._clamp()
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
            logging.info(f"Computing {len(indices)} stimulus responses.")
            for i, stim in enumerate(stim_loader):
                if isinstance(stim, dict):
                    stim = stim[default_stim_key].squeeze(-2)

                fade_in_state = self.fade_in_state(
                    t_fade_in=t_fade_in,
                    dt=dt,
                    initial_frames=stim[:, 0].unsqueeze(1),
                    state=initial_state,
                )

                def handle_stim():
                    # reset stimulus
                    batch_size, n_frames, _ = stim.shape
                    stimulus.zero(batch_size, n_frames)

                    # add stimulus
                    stimulus.add_input(stim.unsqueeze(2))

                    # compute response
                    with stimulus.memory_friendly():
                        states = self(
                            stimulus(), dt, state=fade_in_state, as_states=True
                        )
                        return (
                            stim.cpu().numpy().squeeze(),
                            torch.stack(
                                [s.nodes.activity.cpu() for s in states],
                                dim=1,
                            )
                            .numpy()
                            .squeeze(),
                            torch.stack(
                                [
                                    self.dynamics.currents(s, params).cpu()
                                    for s in states
                                ],
                                dim=1,
                            )
                            .numpy()
                            .squeeze(),
                        )

                yield handle_stim()


class NetworkDir(Directory):
    """Directory for a network."""

    pass


class NetworkView(ConnectomeView):
    """Views and convenience methods for trained networks.

    Args:
        network_dir: directory of the network.

    Attributes:
        dir: directory of the network.
        connectome: connectome directory.
        network: network instance. Requires to call init_network first.
    """

    def __init__(self, network_dir: Union[PathLike, NetworkDir]):
        if isinstance(network_dir, PathLike):
            network_dir = NetworkDir(network_dir)
        self.dir = network_dir
        self.connectome = ConnectomeDir(self.dir.config.network.connectome)
        super().__init__(self.connectome)
        self._initialized = dict(network=False, decoder=False)

    def reset_init(self, key):
        """Reset initialization of a component."""
        self._initialized[key] = False

    def init_network(
        self, chkpt="best_chkpt", network: Optional[Network] = None
    ) -> Network:
        """Initialize the network.

        Args:
            chkpt: checkpoint to load.
            network: network instance to initialize.

        Returns:
            network instance.
        """
        if self._initialized["network"] and network is None:
            return self.network
        self.network = network or Network(**self.dir.config.network)
        state_dict = torch.load(self.dir / chkpt, map_location=flyvision.device)
        self.network.load_state_dict(state_dict["network"])
        self._initialized["network"] = True
        return self.network

    def init_decoder(self, chkpt="best_chkpt", decoder=None):
        """Initialize the decoder.

        Args:
            chkpt: checkpoint to load.
            decoder: decoder instance to initialize.

        Returns:
            decoder instance.
        """
        raise NotImplementedError("Decoder initialization not implemented yet.")
        if self._initialized["decoder"] and decoder is None:
            return self.decoder
        self.decoder = decoder or init_decoder(
            self.dir.config.task_decoder, self.connectome
        )
        state_dict = torch.load(self.dir / chkpt, map_location=flyvision.device)
        self.decoder.load_state_dict(state_dict["decoder"]["flow"])
        self._initialized["decoder"] = True
        return self.decoder

    def __call__(self, movie_input, dt, initial_state=None, as_states=False):
        """Convenience method to simulate the network activity from movie input.

        Note, the nn.Module itself provides more flexibility.

        Args:
            movie_input: tensor requiring shape (batch_size, n_frames, 1, hexals)
            dt: integration time constant. Must be 1/50 or less.
            initial_state: network activity at the beginning of the simulation.
                Either use fade_in_state or steady_state, to compute the
                initial state from grey input or from ramping up the contrast of
                the first movie frame.
            as_states: can return the states as AutoDeref dictionary instead of
                a tensor. Defaults to False.

        Returns:
            LayerActivity object.

        Raises:
            ValueError if the movie_input is not four-dimensional.
            ValueError if the integration time step is bigger than 1/50.
            ValueError if the network is not in evaluation mode or any
                parameters require grad.
        """
        if not self._initialized["network"]:
            self.init_network()
        return LayerActivity(
            self.network.simulate(movie_input, dt, initial_state, as_states).cpu(),
            self.network.connectome,
            keepref=True,
        )


class IntegrationWarning(Warning):
    pass
