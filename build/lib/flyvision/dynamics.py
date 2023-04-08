"""Classes defining the voltage initialization, voltage and current dynamics."""

from typing import Callable
import torch
from torch import nn

from datamate import namespacify
from flyvision.utils.class_utils import forward_subclass
from flyvision.utils.tensor_utils import AutoDeref, RefTensor

__all__ = ["NetworkDynamics", "PPNeuronIGRSynapses"]

activation_fns = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "softplus": nn.Softplus,
    "leakyrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

# -- The `NetworkDynamics` interface -------------------------------------------


class NetworkDynamics:
    """Defines the initialization and behavior of a Network during simulation.

    Extension point: define a subclass of `NetworkDynamics` to implement a
        custom network dynamics model. The subclass must implement the following
        methods:
            write_derived_params
            write_initial_state
            write_state_velocity
        It can implement an __init__ method, to store additional attributes like
        the activation function.
    """

    class Config:
        type: str = "NetworkDynamics"
        activation: str = "relu"

    def __new__(cls, config: Config = {}):
        return forward_subclass(cls, config)

    def write_derived_params(
        self, params: AutoDeref[str, AutoDeref[str, RefTensor]], **kwargs
    ) -> None:
        """
        Augment `params`, called once at every forward pass.

        Parameters:
            params: A namespace containing two subnamespaces: `nodes` and
                `edges`, containing node and edges parameters, respectively.

        This is called at the beginning of a stimulus presentation, before
        `write_initial_state`. Parameter transformations can be moved here to
        avoid repeating them at every timestep.

        Note: called by Network._param_api.
        """
        pass

    def write_initial_state(
        self,
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
        **kwargs
    ) -> None:
        """
        Initialize a network's state variables from its network parameters.

        Parameters:
            state: A namespace containing two subnamspaces: `nodes` and
                `edges`. Write initial node and edge state variable values, as
                1D tensors, into them, respectively.
            params: A namespace containing four subnamespaces: `nodes`,
                `edges`, `sources`, and `targets`. `nodes` and `edges` contain
                node and edges parameters, respectively. `sources` and
                `targets` provide access to the node parameters associated with
                the source node and target node of each edge, respectively.

        Note: called by Network._initial_state.
        """
        pass

    def write_state_velocity(
        self,
        vel: AutoDeref[str, AutoDeref[str, RefTensor]],
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
        target_sum: Callable,
        **kwargs
    ) -> None:
        """
        Compute dx/dt for each state variable.

        Parameters:
            vel: A namespace containing two subnamspaces: `nodes` and
                `edges`. Write dx/dt for node and edge state variables
                into them, respectively.
            state: A namespace containing two subnamespaces: `nodes` and
                `edges`, containing node and edge state variable values,
                respectively.
            params: A namespace containing four subnamespaces: `nodes`,
                `edges`, `sources`, and `targets`. `nodes` and `edges` contain
                node and edges parameters, respectively. `sources` and
                `targets` provide access to the node parameters associated with
                the source node and target node of each edge, respectively.
            target_sum: Sums the entries in a `len(edges)` tensor corresponding
                to edges with the same target node, yielding a `len(nodes)`
                tensor.

        Note: called by Network._next_state.
        """
        pass

    def currents(
        self,
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
    ):
        """
        Compute the current flowing through each edge.

        Parameters:
            state: A namespace containing two subnamespaces: `nodes` and
                `edges`, containing node and edge state variable values,
                respectively.
            params: A namespace containing four subnamespaces: `nodes`,
                `edges`, `sources`, and `targets`. `nodes` and `edges` contain
                node and edges parameters, respectively. `sources` and
                `targets` provide access to the node parameters associated with
                the source node and target node of each edge, respectively.

        Returns:
            A tensor of currents flowing through each edge.

        Note: called by Network.current_response.
        """
        pass


# -- `NetworkDynamics` implementations -----------------------------------------


class PPNeuronIGRSynapses(NetworkDynamics):
    """Passive point neurons with instantaneous graded release synapses."""

    class Config:
        type: str = "PPNeuronIGRSynapses"
        activation: str = "relu"

    def __init__(self, config: Config):
        self.activation = activation_fns[config["activation"].pop("type")](
            **config["activation"]
        )

    def write_derived_params(self, params, **kwargs):
        """Weights are the product of the sign, synapse count, and strength."""
        params.edges.weight = (
            params.edges.sign * params.edges.syn_count * params.edges.syn_strength
        )

    def write_initial_state(self, state, params, **kwargs):
        """Initial state is the bias."""
        state.nodes.activity = params.nodes.bias

    def write_state_velocity(self, vel, state, params, target_sum, x_t, dt, **kwargs):
        """Velocity is the bias plus the sum of the weighted rectified inputs."""
        vel.nodes.activity = (
            1
            / torch.max(params.nodes.time_const, torch.tensor(dt).float())
            * (
                -state.nodes.activity
                + params.nodes.bias
                + target_sum(
                    params.edges.weight * self.activation(state.sources.activity)
                )  # internal chemical current
                + x_t
            )
        )

    def currents(self, state, params):
        """Return the internal chemical current."""
        return params.edges.weight * self.activation(state.sources.activity)
