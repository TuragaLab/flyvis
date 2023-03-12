"""
Deep mechanistic network module.
"""
from typing import Any, Dict, Iterable, List, Union, Callable
from contextlib import contextmanager

import numpy as np
from toolz import valmap
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from datamate import Namespace, Directory

from flyvision.connectome import Connectome
from flyvision.stimulus import Stimulus
from flyvision.initialization import Parameter, init_parameter
from flyvision.dynamics import NetworkDynamics
from flyvision.utils.activity_utils import LayerActivity
from flyvision.utils.nn_utils import n_params, simulation
from flyvision.utils.dataset_utils import IndexSampler
from flyvision.utils.tensor_utils import RefTensor, AutoDeref

import logging

logging = logging.getLogger()


class Network(nn.Module):
    """
    A connectome-constrained network with nodes, edges, and dynamics.
    Fixed Retina Neurons.

    Args:
        config (Namespace): configuration of connectome, dynamics and all parameters.

        Example configuration:

        Namespace(
        connectome = Namespace(
            type = 'Connectome',
            path = '/groups/turaga/home/lappalainenj/FlyVis/dvs-sim/data/fib25-fib19_v2.2.json',
            extent = 15,
            n_syn_fill = 1,
            rotate_filt_n60 = 0
        ),
        dynamics = Namespace(type='StaticSynapses', activation=Namespace(type='relu')),
        node_config = Namespace(
            bias = Namespace(
            type = 'NeuronBias',
            keys = ['type'],
            form = 'normal',
            mode = 'sample',
            requires_grad = True,
            mean = 0.5,
            std = 0.05,
            penalize = Namespace(activity=True),
            seed = 0,
            symmetric = []
            ),
            time_const = Namespace(
            type = 'NeuronTimeConst',
            keys = ['type'],
            form = 'value',
            value = 0.05,
            requires_grad = True,
            symmetric = []
            )
        ),
        edge_config = Namespace(
            sign = Namespace(type='EdgeSign', form='value', requires_grad=False),
            syn_count = Namespace(
            type = 'SynCountSymmetry',
            form = 'lognormal',
            mode = 'mean',
            requires_grad = False,
            std = 1.0,
            clamp = 'non_negative',
            penalize = Namespace(function='weight_decay', args=[0.0]),
            symmetric = []
            ),
            syn_strength = Namespace(
            type = 'NSynStrengthPRFSymmetry',
            form = 'value',
            requires_grad = True,
            scale_elec = 0.01,
            scale_chem = 0.01,
            clamp = 'non_negative',
            penalize = Namespace(function='weight_decay', args=[0.0]),
            symmetric = []
            )
        ),
        sigma = 0.0
        )
    """

    ctome: Connectome
    dynamics: NetworkDynamics
    node_params: Dict[str, Parameter]
    edge_params: Dict[str, Parameter]
    _source_indices: torch.Tensor
    _target_indices: torch.Tensor
    elec_indices: torch.Tensor
    chem_indices: torch.Tensor
    symmetry_mask: Dict
    n_nodes: int
    n_edges: int
    num_parameters: int
    config: Dict
    _state_hook: Dict

    def __init__(
        self,
        connectome: Namespace,
        dynamics: Namespace,
        node_config: Namespace,
        edge_config: Namespace,
        initial_state_mean=None,
        initial_state_std=None,
        sigma=0,
        **kwargs,
    ):

        super().__init__()

        # To be able to alter the passed configs without upstream effect.
        connectome = connectome.deepcopy()
        dynamics = dynamics.deepcopy()
        node_config = node_config.deepcopy()
        edge_config = edge_config.deepcopy()

        # Store the connectome, dynamics, and parameters.
        self.ctome = Connectome(connectome)
        self.dynamics = NetworkDynamics(dynamics)

        # Accessing the h5 datasets in every loop leads to network and read/write bandwidth issues.
        # Therefore, we load those indices into memory, as they are constant during training.

        _node_types = self.ctome.nodes.type[:]
        self.input_indices = np.array(
            [np.nonzero(_node_types == t)[0] for t in self.ctome.input_node_types]
        )
        self.output_indices = torch.tensor(
            np.array(
                [np.nonzero(_node_types == t)[0] for t in self.ctome.output_node_types]
            )
        )

        # Store source/target indices.
        self._source_indices = torch.tensor(self.ctome.edges.source_index[:])
        self._target_indices = torch.tensor(self.ctome.edges.target_index[:])

        # Store chem/elec indices.
        self.elec_indices = torch.tensor(
            np.nonzero(self.ctome.edges.edge_type[:] == b"elec")[0]
        ).long()
        self.chem_indices = torch.tensor(
            np.nonzero(self.ctome.edges.edge_type[:] == b"chem")[0]
        ).long()
        # self._source_indices_elec = self._source_indices[self.elec_indices]
        # self._source_indices_chem = self._source_indices[self.chem_indices]
        # self._target_indices_elec = self._target_indices[self.elec_indices]
        # self._target_indices_chem = self._target_indices[self.chem_indices]
        self.n_nodes = len(self.ctome.nodes.type)
        self.n_edges = len(self.ctome.edges.edge_type)

        # Another way of parameter sharing is averaging at every call with
        # precomputed masks. E.g. used for CT1 compartment model.
        self.symmetry_config = Namespace()  # type: Dict[str, List[torch.Tensor]]
        self.clamp_config = Namespace()

        # Construct node parameter sets.
        self.node_params = Namespace()
        for param_name, param_config in node_config.items():
            param = init_parameter(param_config, self.ctome.nodes)

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
            param = init_parameter(param_config, self.ctome.edges)

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
        self.config = Namespace(
            connectome=connectome,
            dynamics=dynamics,
            node_config=node_config,
            edge_config=edge_config,
        )
        self._state_hook = None  # Namespace(hook=_state_hook, args=[], kwargs={})

        self.initial_state_mean = initial_state_mean
        self.initial_state_std = initial_state_std
        self.sigma = torch.tensor(sigma)
        self.stimulus = Stimulus(1, 1, self.ctome, _init=False)

        logging.info(f"Initialized network with {self.num_parameters} parameters.")
        logging.info(f"Internal noise described by sigma={sigma}.")

    def __repr__(self):
        return f"Network({repr(self.config)})"

    def param_api(self, as_reftensor=True) -> Dict[str, Dict[str, Tensor]]:
        """
        Public parameter api. Returns shared parameter and gathering indices.

        Note, the full api may look like this:

        Namespace(
            nodes = Namespace(
                bias = Namespace(
                        values = Parameter containing:
                                 tensor([0.5563, 0.4876, 0.3637,  ..., 0.4837, 0.5469, 0.3965],
                                 requires_grad=True),
                        indices = tensor([ 0,  0,  0,  ..., 65, 65, 65])
                ),
                time_const = Namespace(
                        values = Parameter containing:
                                 tensor([0., 0., 0.,  ..., 0., 0., 0.], requires_grad=True),
                        indices = tensor([ 0,  0,  0,  ..., 65, 65, 65])
                )
            ),
            edges = Namespace(
                sign = Namespace(
                        values = Parameter containing:
                                 tensor([-1., -1., -1.,  ...,  1.,  1.,  1.]),
                        indices = tensor([  0,   0,   0,  ..., 604, 604, 604])
                [...]
                )
            )
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
                # route is either ("nodes", "sources", "target") or "edges" as specified in the readers
                # RefTensor stores parameters, e.g. "initial_input" for nodes alongside corresponding indices in the lattice
                params[route][param_name] = (
                    RefTensor(values, indices)
                    if as_reftensor
                    else Namespace(parameter=values, indices=indices)
                )
        return params

    def _param_api(self) -> Dict[str, Dict[str, Tensor]]:
        """
        Private parameter api, returns the "params" object passed to "self.dynamics".

        Note, the full api may look like this:

        AutoDeref("nodes": AutoDeref("initial_input": <dvs.networks.RefTensor object at 0x7f2847e53c18>,
                       "bias": <dvs.networks.RefTensor object at 0x7f2775f72278>,
                       "time_const": <dvs.networks.RefTensor object at 0x7f2775e94550>),
           "edges": AutoDeref("sign": <dvs.networks.RefTensor object at 0x7f2775e94908>,
                       "syn_count": <dvs.networks.RefTensor object at 0x7f2775e94ac8>,
                       "syn_strength": <dvs.networks.RefTensor object at 0x7f2775e94b00>),
           "sources": AutoDeref("initial_input": <dvs.networks.RefTensor object at 0x7f2775e943c8>,
                         "bias": <dvs.networks.RefTensor object at 0x7f2775e94588>,
                         "time_const": <dvs.networks.RefTensor object at 0x7f2775e94940>),
           "targets": AutoDeref("initial_input": <dvs.networks.RefTensor object at 0x7f2775e94860>,
                         "bias": <dvs.networks.RefTensor object at 0x7f286c4adf60>,
                         "time_const": <dvs.networks.RefTensor object at 0x7f2775e945c0>))

        TODO: with new param initialization creating RefTensors at beginning this can be faster.
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
            # E.g. for syn count semantic values is exp(syn_count)
            values = parameter.semantic_values
            for route, indices in parameter.readers.items():
                # route is either ("nodes", "sources", "target") or "edges" as specified in the readers
                # RefTensor stores parameters, e.g. "initial_input" for nodes alongside corresponding indices in the lattice
                params[route][param_name] = RefTensor(values, indices)
        # Add derived parameters.
        self.dynamics.write_derived_params(
            params,
            chem_indices=self.chem_indices,
            elec_indices=self.elec_indices,
        )
        for k, v in params.nodes.items():
            if k not in params.sources:  # ?
                params.sources[k] = self._source_gather(v)
                params.targets[k] = self._target_gather(v)

        return params

    # -- Scatter/gather operations -------------------------

    def _source_gather(self, x: Tensor) -> Tensor:
        """Gathers values from x at source indices, e.g. distributes source node activity over edges."""
        return RefTensor(x, self._source_indices)

    def _target_gather(self, x: Tensor) -> Tensor:
        """Gathers values from x at target indices, e.g. distributes target node activity over edges."""
        return RefTensor(x, self._target_indices)

    def target_sum(self, x: Tensor) -> Tensor:
        """Sums all values from x into result at the indices specified in the index tensor along a given axis dim.
        Args:
            x (tensor): Input "weight * source activity". Corresponds to synaptic currents. Must be aggregated over receptive fields.

        Node: If multiple indices reference the same location, their contributions add.
        Analog to pandas groupby("target_indices").sum() in multiple dimensions.
        """
        result = torch.zeros((*x.shape[:-1], self.n_nodes))  # (n_frames, n_nodes)
        # args = dim, index, other
        result.scatter_add_(
            -1,  # n_nodes dim
            self._target_indices.expand(
                *x.shape
            ),  # view of indexing vector expanded over dims of x
            x,
        )
        return result

    def target_sum(self, x: Tensor) -> Tensor:
        """Sums all values from x into result at the indices specified in the index tensor along a given axis dim.
        Args:
            x (tensor): Input "weight * source activity". Corresponds to synaptic currents. Must be aggregated over receptive fields.

        Node: If multiple indices reference the same location, their contributions add.
        Analog to pandas groupby("target_indices").sum() in multiple dimensions.
        """
        result = torch.zeros((*x.shape[:-1], self.n_nodes))  # (n_frames, n_nodes)
        # args = dim, index, other
        result.scatter_add_(
            -1,  # n_nodes dim
            self._target_indices.expand(
                *x.shape
            ),  # view of indexing vector expanded over dims of x
            x,
        )
        return result

    # ------------------------------------------------------

    def _initial_state(
        self, params: Dict[str, Dict[str, Tensor]], batch_size: int
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Compute the initial state given the stimulus "x".
        """
        # Initialize the network.
        state = AutoDeref(nodes=AutoDeref(), edges=AutoDeref())
        self.dynamics.write_initial_state(
            state,
            params,
            self._source_indices,
            self.initial_state_mean,
            self.initial_state_std,
        )

        # Expand over batch dimension.
        for k, v in state.nodes.items():
            state.nodes[k] = v.expand(batch_size, *v.shape)
        for k, v in state.edges.items():
            state.edges[k] = v.expand(batch_size, *v.shape)

        return self._state_api(state)

    def _next_state(
        self,
        params: Dict[str, Dict[str, Tensor]],
        state: Dict[str, Dict[str, Tensor]],
        x_t: Tensor,
        dt: float,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Compute the next state, given the current state ("s") and the stimulus
        ("x").
        """
        # Compute state velocities.
        vel = AutoDeref(nodes=AutoDeref(), edges=AutoDeref())

        self.dynamics.write_state_velocity(
            vel, state, params, self.target_sum, x_t, dt=dt, sigma=self.sigma
        )

        # Construct and return the next state.
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
        self, state: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Return the "state" object passed to "self.dynamics".

        Note, the full api may look like this:

        {"nodes": {"activity": tensor([[1.1998, 1.1596, 1.1748,  ..., 0.0000, 0.0000, 0.0000]], grad_fn=<AsStridedBackward>)},
         "edges": {"current": tensor([[-1.1377e-04, -1.1377e-04, -1.1377e-04,  ...,  8.5991e-05, 8.5991e-05,  8.5991e-05]], grad_fn=<ExpandBackward>)},
         "sources": {"activity": <dvs.networks.RefTensor object at 0x7f85e2b61940>},
         "targets": {"activity": <dvs.networks.RefTensor object at 0x7f85e2b619b0>}}
        """

        if self._state_hook is not None:
            _state = self._state_hook.hook(state, **self._state_hook.kwargs)
            if _state is not None:
                state = _state

        state = AutoDeref(
            nodes=state.nodes,
            edges=state.edges,
            sources=AutoDeref(**valmap(self._source_gather, state.nodes)),
            targets=AutoDeref(**valmap(self._target_gather, state.nodes)),
        )
        return state

    def register_state_hook(
        self, state_hook: Dict[str, Union[Callable, Dict[str, Any]]]
    ) -> None:
        """Register a state hook to retrieve or modify the state.

        Args:
            state_hook: provides the callable and key word arguments for it.

        Note: the hook is called in _state_api.
        """
        if self._state_hook is not None and self._state_hook != state_hook:
            raise ValueError(
                "call remove_state_hook to remove the existing state hook first."
            )
        if all((key in state_hook for key in ["hook", "kwargs"])):
            self._state_hook = state_hook
        else:
            raise ValueError("Violating state_hook api")

    def remove_state_hook(self):
        self._state_hook = None

    def init_from_other(self, other, params):
        for param_name in params:
            tensor = getattr(self, param_name)
            tensor.data = getattr(other, param_name).data

    def simulate(
        self,
        movie_input: torch.Tensor,
        dt: float,
        initial_state: Union[AutoDeref, None] = "auto",
        as_states: bool = False,
    ) -> Union[torch.Tensor, AutoDeref]:
        """Simulate the network activity from movie input.

        Args:
            movie_input: tensor requiring shape (#samples, #frames, 1, hexals)
            dt: integration time constant. Must be 1/50 or less.
            initial_state: network activity at the beginning of the simulation.
                Either use fade_in_state or steady_state, to compute the
                initial state from grey input or from ramping up the contrast of
                the first movie frame.
            as_states: can return the states as AutoDeref dictionary instead of
                a tensor. Defaults to False.

        Returns:
            activity tensor of shape (#samples, #frames, #neurons)

        Raises:
            ValueError if the movie_input is not four-dimensional.
            ValueError if the integration time step is bigger than 1/50.
            ValueError if the network is not in evaluation mode or any
                parameters require grad.
        """

        if len(movie_input.shape) != 4:
            raise ValueError("requires shape (sample, frame, 1, hexals)")

        if dt > 1 / 50:
            raise ValueError

        n_samples, n_frames = movie_input.shape[:2]
        if initial_state == "auto":
            initial_state = self.steady_state(1.0, dt, movie_input.shape[0])
        with simulation(self):
            assert self.training == False and all(
                not p.requires_grad for p in self.parameters()
            )
            self.stimulus.zero(n_samples, n_frames)
            self.stimulus.add_input(movie_input)
            return self.forward(self.stimulus(), dt, initial_state, as_states)

    def simulate_ablate(
        self,
        movie_input: torch.Tensor,
        dt: float,
        initial_state: Union[AutoDeref, None],
        ablate_cell_type: str,
        ablation_mode: "str",
    ):
        activity = self.simulate(movie_input, dt, initial_state, as_states=False)

        if ablation_mode == "mean":
            activity[:, :, self.stimulus.layer_index[ablate_cell_type]] = activity[
                :, :, self.stimulus.layer_index[ablate_cell_type]
            ].mean(dim=(-1, -2))
        elif ablation_mode == "zero":
            activity[:, :, self.stimulus.layer_index[ablate_cell_type]] = 0
        return activity

    def forward(self, x, dt, state=None, as_states=False):
        """
        Args:
            x (Tensor): whole-network stimulus of shape (#samples, #frames, #nodes).
            dt (float): integration time constant.
        """
        # To keep the parameters within their valid domain, they get clamped.
        self._clamp(dt)
        # Construct the parameter API.
        params = self._param_api()

        # Initialize the network state.
        if state is None:
            state = self._initial_state(params, x.shape[0])

        def handle(state):
            # For simulating the dynamics, we loop over the temporal dimension.
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
        t_pre,
        dt,
        batch_size,
        value=0.5,
        initial_frames=None,
        state=None,
        no_grad=True,
    ):
        """State after grey-scale or initial frame fade in stimulus.

        initial_frames of shape (#samples, 1, n_hexals)
        """
        if t_pre is None or t_pre <= 0.0:
            return state
        if value is not None and initial_frames is None:
            self.stimulus.zero(batch_size, int(t_pre / dt))
            self.stimulus.add_pre_stim(value)
        elif value is None and initial_frames is not None:
            raise ValueError("use fade_in_state method instead")
            # replicate initial frame over int(t_pre/dt) - fade in
            self.stimulus.zero(batch_size, int(t_pre / dt))
            initial_frames = (
                torch.linspace(0, 1, int(t_pre / dt))[None, :, None]
                * (initial_frames.repeat(1, int(t_pre / dt), 1) - 0.5)
                + 0.5
            )
            self.stimulus.add_input(initial_frames)
        else:
            return state
        if no_grad:
            with torch.no_grad():
                return self(self.stimulus(), dt, as_states=True, state=state)[-1]
        else:
            return self(self.stimulus(), dt, as_states=True, state=state)[-1]

    def fade_in_state(
        self,
        t_fade_in,
        dt,
        initial_frames,
        state=None,
        no_grad=True,
    ):
        """State after grey-scale or initial frame fade in stimulus.

        initial_frames of shape (#samples, 1, n_hexals)
        """
        if t_fade_in is None or t_fade_in <= 0.0:
            return state
        batch_size = initial_frames.shape[0]
        # replicate initial frame over int(t_fade_in/dt) - fade in
        self.stimulus.zero(batch_size, int(t_fade_in / dt))
        initial_frames = (
            torch.linspace(0, 1, int(t_fade_in / dt))[None, :, None]
            * (initial_frames.repeat(1, int(t_fade_in / dt), 1) - 0.5)
            + 0.5
        )
        self.stimulus.add_input(initial_frames[:, :, None])
        if no_grad:
            with torch.no_grad():
                return self(self.stimulus(), dt, as_states=True, state=state)[-1]
        else:
            return self(self.stimulus(), dt, as_states=True, state=state)[-1]

    def forward_single(self, x, dt, state=None, params=None):
        if params is None:
            params = self._param_api()
        if state is None:
            state = self._initial_state(params, x.shape[0])
        return self._next_state(params, state, x, dt)

    def _clamp(self, dt=None):
        """
        To clamp parameters to their range specifid in their config.
        Nodes and edges parameters are combined in clamp_config.
        Clamp modes can be:
            - non_negative to clamp at zero
            - tuple of the form (min, max) to clamp to an arbitrary range
            - time_const (deprecated has no effect since time constants are
                now clamped in the dynamics) to clamp to the current integration
                time step dt
        """
        for param_name, mode in self.clamp_config.items():
            param = getattr(self, param_name)
            if param.requires_grad:
                if mode is None:
                    pass
                elif mode == "non_negative":
                    param.data.clamp_(0)
                # elif mode == "time_const":
                #     pass
                # #     param.data.clamp_(dt)
                elif isinstance(mode, Iterable) and len(mode) == 2:
                    param.data.clamp_(*mode)
                else:
                    raise NotImplementedError(f"Clamping mode {mode} not implemented.")
        for param_name, masks in self.symmetry_config.items():
            param = getattr(self, param_name)
            if param.requires_grad:
                for symmetry in masks:
                    param.data[symmetry] = param.data[symmetry].mean()
        # for param_name, index in self.clamp_config_dt.items():
        #     param = getattr(self, param_name)
        #     if param.requires_grad:
        #         param.data[index] = param.data[index].mean()

    @contextmanager
    def enable_grad(self, grad=True):
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(grad)
        try:
            yield
        finally:
            torch.set_grad_enabled(prev)

    def stimulus_response(
        self,
        stim_dataset,
        dt,
        indices=None,
        t_pre=1.0,
        t_fade_in=0.0,
        grad=False,
        default_stim_key="lum",
    ):
        stim_dataset.dt = dt
        if indices is None:
            indices = np.arange(len(stim_dataset))
        stim_loader = DataLoader(
            stim_dataset, batch_size=1, sampler=IndexSampler(indices)
        )

        stimulus = self.stimulus
        initial_state = self.steady_state(t_pre, dt, batch_size=1, value=0.5)
        with self.enable_grad(grad):
            logging.info(f"Computing {len(indices)} stimulus responses.")
            for i, stim in enumerate(stim_loader):

                # to avoid having to write an only-lum version for task datasets
                if isinstance(stim, dict):
                    stim = stim[default_stim_key].squeeze(-2)

                # returns initial state if t_fade_in is 0
                fade_in_state = self.fade_in_state(
                    t_fade_in=t_fade_in,
                    dt=dt,
                    initial_frames=stim[:, 0].unsqueeze(1),
                    state=initial_state,
                )

                def handle_stim():

                    # Resets the stimulus buffer (#samples, #frames, #neurons).
                    n_samples, n_frames, _ = stim.shape
                    stimulus.zero(n_samples, n_frames)

                    # Add batch of hex-videos (#samples, #frames, #hexals) as
                    # photorecptor stimuli.
                    stimulus.add_input(stim.unsqueeze(2))

                    with stimulus.memory_friendly():
                        if grad is False:
                            return (
                                self(stimulus(), dt, state=fade_in_state)
                                .detach()
                                .cpu()
                                .numpy(),
                                stim.cpu().numpy(),
                            )
                        elif grad is True:
                            return (
                                self(stimulus(), dt, state=fade_in_state),
                                stim.cpu().numpy(),
                            )

                yield handle_stim()

    def stimulus_responses(
        self,
        stim_dataset,
        dt,
        indices=None,
        t_pre=1.0,
        t_fade_in=0.0,
        grad=False,
        default_stim_key="lum",
    ):
        stim_dataset.dt = dt
        if indices is None:
            indices = np.arange(len(stim_dataset))
        stim_loader = DataLoader(
            stim_dataset, batch_size=1, sampler=IndexSampler(indices)
        )

        stimulus = self.stimulus
        initial_state = self.steady_state(t_pre, dt, batch_size=1, value=0.5)
        with self.enable_grad(grad):
            logging.info(f"Computing {len(indices)} stimulus responses.")
            for i, stim in enumerate(stim_loader):

                # to avoid having to write an only-lum version for task datasets
                if isinstance(stim, dict):
                    stim = stim[default_stim_key].squeeze(-2)

                # returns initial state if t_fade_in is 0
                fade_in_state = self.fade_in_state(
                    t_fade_in=t_fade_in,
                    dt=dt,
                    initial_frames=stim[:, 0].unsqueeze(1),
                    state=initial_state,
                )

                def handle_stim():

                    # Resets the stimulus buffer (#samples, #frames, #neurons).
                    n_samples, n_frames, _ = stim.shape
                    print(n_frames, dt, stim_dataset.dt)
                    stimulus.zero(n_samples, n_frames)

                    # Add batch of hex-videos (#samples, #frames, #hexals) as
                    # photorecptor stimuli.
                    stimulus.add_input(stim.unsqueeze(2))

                    with stimulus.memory_friendly():
                        if grad is False:
                            return (
                                self(stimulus(), dt, state=fade_in_state)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        elif grad is True:
                            return self(stimulus(), dt, state=fade_in_state)

                yield handle_stim()

    def current_response(
        self,
        stim_dataset,
        dt,
        indices=None,
        t_pre=1.0,
        default_stim_key="lum",
        t_fade_in=0,
    ):
        """To return postsynaptic currents.

        Note, requires Dynamics to implement currents method and expects
        that this is weights * activation(presynaptic voltage).

        Returns:
            stims: the stimulus
            currents: postsynaptic currents per synapse (edge).
            target_currents: postsynaptic currents per postsynaptic cell (node).
            activities: full postsynaptic voltage.
        """
        self._clamp(dt)
        # Construct the parameter API.
        params = self._param_api()

        stim_dataset.dt = dt
        if indices is None:
            indices = np.arange(len(stim_dataset))
        stim_loader = DataLoader(
            stim_dataset, batch_size=1, sampler=IndexSampler(indices)
        )

        stimulus = Stimulus(1, 1, self.ctome)
        initial_state = self.steady_state(t_pre, dt, batch_size=1, value=0.5)
        stims = []
        currents = []
        activities = []
        with torch.no_grad():
            logging.info(f"Computing {len(indices)} stimulus responses.")
            for i, stim in enumerate(stim_loader):

                # to avoid having to write an only-lum version for task datasets
                if isinstance(stim, dict):
                    stim = stim[default_stim_key].squeeze(-2)

                # returns initial state if t_fade_in is 0
                fade_in_state = self.fade_in_state(
                    t_fade_in=t_fade_in,
                    dt=dt,
                    initial_frames=stim[:, 0].unsqueeze(1),
                    state=initial_state,
                )

                def handle_stim():

                    # Resets the stimulus buffer (#samples, #frames, #neurons).
                    n_samples, n_frames, _ = stim.shape
                    stimulus.zero(n_samples, n_frames)

                    # Add batch of hex-videos (#samples, #frames, #hexals) as
                    # photorecptor stimuli.
                    stimulus.add_input(stim.unsqueeze(2))

                    with stimulus.memory_friendly():
                        states = self(
                            stimulus(), dt, state=fade_in_state, as_states=True
                        )
                        return (
                            torch.stack(
                                [s.nodes.activity.cpu() for s in states],
                                dim=1,
                            ).numpy(),
                            torch.stack(
                                [
                                    self.dynamics.currents(s, params).cpu()
                                    for s in states
                                ],
                                dim=1,
                            ).numpy(),
                        )

                activity, current = handle_stim()
                stims.append(stim.cpu().numpy().squeeze())
                activities.append(activity.squeeze())
                currents.append(current.squeeze())
        return np.array(stims), np.array(currents), np.array(activities)


class NetworkDir(Directory):
    pass


class NetworkView:
    def __init__(self, network_dir: NetworkDir):
        self.dir = network_dir
        self._initialized = dict(network=False)

    def reset_init(self, key):
        self._initialized[key] = False

    def init_network(self, chkpt="best_chkpt", network=None):
        if self._initialized["network"] and network is None:
            return self.network
        self.network = network or Network(**self.dir.config.network)
        state_dict = torch.load(self.dir / chkpt)
        self.network.load_state_dict(state_dict["network"])
        self._initialized["network"] = True
        return self.network

    def __call__(self, movie_input, dt, initial_state=None, as_states=False):
        """Simulate the network activity from movie input.

        Args:
            movie_input: tensor requiring shape (#samples, #frames, 1, hexals)
            dt: integration time constant. Must be 1/50 or less.
            initial_state: network activity at the beginning of the simulation.
                Either use fade_in_state or steady_state, to compute the
                initial state from grey input or from ramping up the contrast of
                the first movie frame.
            as_states: can return the states as AutoDeref dictionary instead of
                a tensor. Defaults to False.

        Returns:
            activity tensor of shape (#samples, #frames, #neurons)

        Raises:
            ValueError if the movie_input is not four-dimensional.
            ValueError if the integration time step is bigger than 1/50.
            ValueError if the network is not in evaluation mode or any
                parameters require grad.
        """
        if not self._initialized["network"]:
            self._init_network()
        return LayerActivity(
            self.network.simulate(movie_input, dt, initial_state, as_states),
            self.network.ctome,
            keepref=True,
        )


# -- State hook ---------------------------------------------------------------
def _state_hook(x, *args, **kwargs):
    """Default state hook returning identity."""
    return x
