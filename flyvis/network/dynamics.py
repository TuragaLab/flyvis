"""Classes defining the voltage initialization, voltage and current dynamics."""

from typing import Callable, Dict

import torch
from torch import nn

from flyvis.utils.tensor_utils import AutoDeref, RefTensor

__all__ = ["NetworkDynamics", "PPNeuronIGRSynapses"]

activation_fns: Dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "softplus": nn.Softplus,
    "leakyrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

# -- The `NetworkDynamics` interface -------------------------------------------


class NetworkDynamics:
    """
    Defines the initialization and behavior of a Network during simulation.

    This class serves as an extension point for implementing custom network dynamics
    models. Subclasses must implement the following methods:

    - write_derived_params
    - write_initial_state
    - write_state_velocity

    Attributes:
        activation (nn.Module): The activation function for the network.

    Args:
        activation (dict): A dictionary specifying the activation function type and
            its parameters.
    """

    def __init__(self, activation: Dict[str, str] = {"type": "relu"}):
        self.activation = activation_fns[activation.pop("type")](**activation)

    def write_derived_params(
        self, params: AutoDeref[str, AutoDeref[str, RefTensor]], **kwargs
    ) -> None:
        """
        Augment `params`, called once at every forward pass.

        Args:
            params: A directory containing two subdirectories: `nodes` and
                `edges`, containing node and edges parameters, respectively.
            **kwargs: Additional keyword arguments.

        Note:
            This is called once per forward pass at the beginning. It's required
            after parameters have been updated by an optimizer but not at every
            timestep. Called by Network._param_api.
        """
        pass

    def write_initial_state(
        self,
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
        **kwargs,
    ) -> None:
        """
        Initialize a network's state variables from its network parameters.

        Args:
            state: A directory containing two subdirectories: `nodes` and
                `edges`. Write initial node and edge state variable values, as
                1D tensors, into them, respectively.
            params: A directory containing four subdirectories: `nodes`,
                `edges`, `sources`, and `targets`. `nodes` and `edges` contain
                node and edges parameters, respectively. `sources` and
                `targets` provide access to the node parameters associated with
                the source node and target node of each edge, respectively.
            **kwargs: Additional keyword arguments.

        Note:
            Called by Network._initial_state.
        """
        pass

    def write_state_velocity(
        self,
        vel: AutoDeref[str, AutoDeref[str, RefTensor]],
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
        target_sum: Callable,
        **kwargs,
    ) -> None:
        """
        Compute dx/dt for each state variable.

        Args:
            vel: A directory containing two subdirectories: `nodes` and
                `edges`. Write dx/dt for node and edge state variables
                into them, respectively.
            state: A directory containing two subdirectories: `nodes` and
                `edges`, containing node and edge state variable values,
                respectively.
            params: A directory containing four subdirectories: `nodes`,
                `edges`, `sources`, and `targets`. `nodes` and `edges` contain
                node and edges parameters, respectively. `sources` and
                `targets` provide access to the node parameters associated with
                the source node and target node of each edge, respectively.
            target_sum: Sums the entries in a `len(edges)` tensor corresponding
                to edges with the same target node, yielding a `len(nodes)`
                tensor.
            **kwargs: Additional keyword arguments.

        Note:
            Called by Network._next_state.
        """
        pass

    def currents(
        self,
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
    ) -> torch.Tensor:
        """
        Compute the current flowing through each edge.

        Args:
            state: A directory containing two subdirectories: `nodes` and
                `edges`, containing node and edge state variable values,
                respectively.
            params: A directory containing four subdirectories: `nodes`,
                `edges`, `sources`, and `targets`. `nodes` and `edges` contain
                node and edges parameters, respectively. `sources` and
                `targets` provide access to the node parameters associated with
                the source node and target node of each edge, respectively.

        Returns:
            A tensor of currents flowing through each edge.

        Note:
            Called by Network.current_response.
        """
        pass


# -- `NetworkDynamics` implementations -----------------------------------------


class PPNeuronIGRSynapses(NetworkDynamics):
    """Passive point neurons with instantaneous graded release synapses."""

    def write_derived_params(
        self, params: AutoDeref[str, AutoDeref[str, RefTensor]], **kwargs
    ) -> None:
        """
        Calculate weights as the product of sign, synapse count, and strength.

        Args:
            params: A directory containing edge parameters.
            **kwargs: Additional keyword arguments.
        """
        params.edges.weight = (
            params.edges.sign * params.edges.syn_count * params.edges.syn_strength
        )

    def write_initial_state(
        self,
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
        **kwargs,
    ) -> None:
        """
        Set the initial state to the bias.

        Args:
            state: A directory to write the initial state.
            params: A directory containing node parameters.
            **kwargs: Additional keyword arguments.
        """
        state.nodes.activity = params.nodes.bias

    def write_state_velocity(
        self,
        vel: AutoDeref[str, AutoDeref[str, RefTensor]],
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
        target_sum: Callable,
        x_t: torch.Tensor,
        dt: float,
        **kwargs,
    ) -> None:
        """
        Calculate velocity as bias plus sum of weighted rectified inputs.

        Args:
            vel: A directory to write the calculated velocity.
            state: A directory containing current state values.
            params: A directory containing node and edge parameters.
            target_sum: Function to sum edge values for each target node.
            x_t: External input at time t.
            dt: Time step.
            **kwargs: Additional keyword arguments.
        """
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

    def currents(
        self,
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
    ) -> torch.Tensor:
        """
        Calculate the internal chemical current.

        Args:
            state: A directory containing current state values.
            params: A directory containing edge parameters.

        Returns:
            torch.Tensor: The calculated internal chemical current.
        """
        return params.edges.weight * self.activation(state.sources.activity)
