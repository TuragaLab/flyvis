from typing import Protocol, Union, Optional, Dict

import numpy as np
from torch import nn
import torch
from toolz import valfilter
from datamate import Directory, Namespace

from flyvision.network import Network, NetworkDir
from flyvision.tasks import Task

import logging

logging = logger = logging.getLogger(__name__)


class SolverProtocol(Protocol):
    name: str
    config: Optional[Union[dict, Namespace]]

    def __init__(
        self, name: str = "", config: Optional[Union[dict, Namespace]] = None
    ) -> None: ...

    # placeholders for the `task` and `conditions` properties for illustration purposes
    # actual type hints should be replaced with the correct types
    dir: Directory = None
    network: Network = None
    decoder: Dict[str, nn.Module] = None
    task: Task = None
    optimizer: object = None
    penalty: object = None
    scheduler: object = None

    def train(self) -> None: ...

    def checkpoint(self) -> None: ...

    def test(self) -> None: ...

    # hint at other methods mentioned like 'recover'
    def recover(self) -> None: ...


class MultiTaskSolver:
    tasks: Dict[str, Task]

    def __init__(
        self,
        name: str = "",
        config: Optional[Union[dict, Namespace]] = None,
        init_network: bool = True,
        init_decoder: bool = True,
        init_task: bool = True,
        init_optim: bool = True,
        init_penalties: bool = True,
        init_scheduler: bool = True,
        delete_if_exists: bool = False,
    ) -> None:
        self.dir = NetworkDir(name, config or {})

        self.path = self.dir.path

        self.config = self.dir.config

        self.iteration = 0
        self._last_chkpt_ind = -1
        self._curr_chkpt_ind = -1

        self._initialized = self._init_solver(
            init_network=init_network,
            init_decoder=init_decoder,
            init_task=init_task,
            init_optim=init_optim,
            init_penalties=init_penalties,
            init_scheduler=init_scheduler,
        )

        logging.info("Initialized solver.")
        logging.info(repr(self.config))

    def _init_solver(
        self,
        init_network=False,
        init_decoder=False,
        init_task=False,
        init_optim=False,
        init_penalties=False,
        init_scheduler=False,
    ):
        """Initialize solver."""
        initialized = []

        if init_network:
            self.network = Network(self.config.network)
            initialized.append("network")

        if init_task:
            self.task = Task(**self.config.task)
            # self.task_keys = self.task.dataset.tasks
            # self.task_weights = self.task.dataset.task_weights
            # self.n_iters = self.config.task.n_iters
            # if not (self.wrap.path / "n_train_samples.h5").exists():
            #     self.wrap.n_train_samples = len(self.task.train_data)
            # if not (self.wrap.path / "train_data_index.h5").exists():
            #     self.wrap.train_data_index = self.task.train_seq_index
            # if not (self.wrap.path / "val_data_index.h5").exists():
            #     self.wrap.val_data_index = self.task.val_seq_index
            initialized.append("task")

            if init_decoder:
                self.decoder = self.task.init_decoder(self.network.connectome)
                initialized.append("decoder")

        if init_optim:
            self.optimizer = self._init_optimizer(
                self.config.optim, self.network, self.decoder
            )
            initialized.append("optim")

        if init_penalties:
            self.penalizer = Penalty(self.config.penalizer, self.network)
            initialized.append("penalties")

        if init_scheduler:
            self.scheduler = HyperParamScheduler(
                self.config.scheduler,
                self.config.task.n_iters,
                self.network,
                self.task,
                self.optimizer,
                self.penalizer,
            )
            self.scheduler(self.iteration)
            initialized.append("scheduler")

        return initialized

    @staticmethod
    def _init_optimizer(
        optim: Namespace, network: Network, decoder: Optional[Dict[str, nn.Module]]
    ):
        """Initializes the optim of network and decoder."""

        def decoder_parameters(decoder: Dict[str, nn.Module]):
            """Returns decoder parameters."""
            params = []
            for task, nn_module in enumerate(decoder.values()):
                params.append(
                    dict(
                        params=[w for w in nn_module.parameters()],
                        **config.optim_dec,
                    )
                )
            return params

        config = optim.deepcopy()

        optim_type = config.pop("type", "Adam")
        optim = torch.optim.__dict__[optim_type]
        logging.info(f"Initializing {optim.__name__} for network and decoder.")

        param_groups = [dict(params=network.parameters(), **config.optim_net)]

        if decoder:
            param_groups.extend(decoder_parameters(decoder))

        return optim(param_groups)

    def train(self) -> None: ...

    def checkpoint(self) -> None: ...

    def test(self) -> None: ...

    def recover(self) -> None: ...


class Rectifier:
    def __init__(self, positive_weight=1, negative_weight=0.1):
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def __call__(self, difference):
        return self.positive_weight * nn.functional.relu(
            difference
        ) - self.negative_weight * nn.functional.relu(-difference)


class Penalty:
    """Penalties on specific parameters.

    Args:
        solver (MultiTaskSolver): the solver instance.

    Example configurations passed to the network object:
        # Example 1: Penalize the resting potential of all cell types.
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
            )
        # Example 2: add a weight decay penalty to all synapse strengths.
        syn_strength=Namespace(
                type="SynapseCountScaling",
                initial_dist="Value",
                requires_grad=True,
                scale_elec=0.01,
                scale_chem=0.01,
                clamp="non_negative",
                groupby=["source_type", "target_type", "edge_type"],
                penalize=Namespace(function="weight_decay", kwargs=dict(lambda=1e-3,)),
            )
    """

    solver: object
    central_nodes_index: np.ndarray
    parameter_config: Namespace
    activity_penalty: float
    activity_baseline: float
    activity_penalty_stop_iter: int
    parameter_optim: torch.optim.Optimizer = None
    activity_optim: torch.optim.Optimizer = None
    optimizers: Dict[str, torch.optim.Optimizer]
    rectifier: Rectifier
    default_optim: torch.optim.Optimizer = torch.optim.SGD

    def __init__(self, penalty: Namespace, network: Network):
        self.config = penalty
        self.network = network

        # collecting parameter penalty configuration
        self.parameter_config = self.get_configs()

        self.init_optim()
        self.init_rectifier()

    def __repr__(self):
        return (
            f"Penalty("
            f"parameter_config={self.parameter_config}, "
            f"activity_penalty={self.activity_penalty}, "
            f"activity_baseline={self.activity_baseline}, "
            f"activity_penalty_stop_iter={self.activity_penalty_stop_iter}, "
            f"below_baseline_penalty_weight={self.below_baseline_penalty_weight}, "
            f"above_baseline_penalty_weight={self.above_baseline_penalty_weight}"
            f")"
        )

    def __call__(self, activity, iteration):
        """Run all configured penalties."""
        if self.parameter_optim:
            self.param_penalty_step()
        if self.activity_optim:
            if (
                self.activity_penalty_stop_iter is None
                or iteration < self.activity_penalty_stop_iter
            ):
                self.activity_penalty_step(activity, retain_graph=False)
            else:
                self.activity_optim = None

    def init_optim(self):
        """Initialize the individual optimizer instances with the correct set of
        parameters."""
        self.optimizers = {}
        self.param_list_func_pen = []
        self.param_list_act_pen = []

        # collect the parameters that need to be penalized
        # either by a function or by activity
        for name, config in self.parameter_config.items():
            if "function" in config and any(list(config.kwargs.values())):
                self.param_list_func_pen.append(name)
            if getattr(config, "activity", False):
                self.param_list_act_pen.append(name)

        if self.param_list_func_pen:
            self.parameter_optim = self.default_optim(
                (getattr(self.network, param) for param in self.param_list_func_pen),
                lr=1e-3,
            )  # LR is overwritten by scheduler.
            self.optimizers.update(dict(parameter_optim=self.parameter_optim))

        if self.param_list_act_pen:
            self.activity_optim = self.default_optim(
                (getattr(self.network, param) for param in self.param_list_act_pen),
                lr=1e-3,
            )  # LR is overwritten by scheduler.
            self.optimizers.update(dict(activity_optim=self.activity_optim))

    def init_rectifier(self):
        # collecting activity penalty parameters
        (
            self.activity_penalty,
            self.activity_baseline,
            self.activity_penalty_stop_iter,
            self.below_baseline_penalty_weight,
            self.above_baseline_penalty_weight,
        ) = self.get_act_pen_hparams()

        if (
            not any(
                (
                    self.activity_penalty,
                    self.activity_baseline,
                    self.below_baseline_penalty_weight,
                    self.above_baseline_penalty_weight,
                )
            )
            and self.param_list_act_pen
        ):
            raise ValueError(
                "Activity penalty is enabled but no activity penalty parameters are "
                "set."
            )

        self.rectifier = Rectifier(
            self.below_baseline_penalty_weight,
            self.above_baseline_penalty_weight,
        )

    def get_configs(self) -> Namespace:
        node_config = Namespace(
            {
                "nodes_" + k: v.pop("penalize", None)
                for k, v in self.network.config.node_config.deepcopy().items()
            }
        )
        edge_config = Namespace(
            {
                "edges_" + k: v.pop("penalize", None)
                for k, v in self.network.config.edge_config.deepcopy().items()
            }
        )
        return valfilter(
            lambda v: v is not None,
            Namespace(**node_config, **edge_config),
            factory=Namespace,
        )

    def get_act_pen_hparams(
        self,
    ) -> tuple:
        config = self.config.get("activity_penalty", Namespace())
        return (
            config.get("activity_penalty", None),
            config.get("activity_baseline", None),
            config.get("stop_iter", None),
            config.get("below_baseline_penalty_weight", None),
            config.get("above_baseline_penalty_weight", None),
        )

    def _chkpt(self):
        """Returns a dictionary of all state dicts of all optimizer instances."""
        _chkpt = {}
        for key, optim in self.optimizers.items():
            if optim is not None:
                _chkpt[key] = optim.state_dict()
        return _chkpt

    def param_penalty_step(self):
        """Apply all the penalties on the individual parameters."""
        self.parameter_optim.zero_grad()
        penalty = 0
        for param, config in self.parameter_config.items():
            if getattr(config, "function", False):
                penalty += getattr(self, config.function)(param, config)
        penalty.backward()
        self.parameter_optim.step()
        self.network._clamp()

    def activity_penalty_step(self, activity, retain_graph=True):
        """Penalizes parameters tracked in activity_optim for too high or low acticity.

        Encourages the central nodes to have a higher temporal mean activity, remedying
        dead neurons.

        Args:
            activity (tensor): network activity of shape (n_samples, n_frames, n_nodes).
        """
        self.activity_optim.zero_grad()
        n_samples, n_frames, n_nodes = activity.shape
        # the temporal average activity of the central nodes after a couple of frames
        # to avoid the initial transient response
        activity_mean = activity[:, n_frames // 4 :, self.central_nodes_index].mean(
            dim=1
        )  # (n_samples, n_node_types)
        penalty = (
            self.activity_penalty
            * (self.rectifier(self.activity_baseline - activity_mean) ** 2).mean()
        )
        penalty.backward(retain_graph=retain_graph)
        self.activity_optim.step()
        self.network._clamp()

    # -- Penalty functions --

    def weight_decay(self, param, config):
        """Adds weight decay to the loss."""
        w = getattr(self.network, param)
        return config.kwargs["lambda"] * (w**2).sum()

    def prior(self, param, config):
        """L2 penalty towards initial values."""
        _key = "edge_config" if param.startswith("edges") else "node_config"
        # TODO: check that this stores the actual initial values. This might be a
        # convenient but suboptimal implementation if the initial values are cast
        # to tensors at each iteration
        prior = torch.tensor(
            getattr(self.network.config, _key)[
                param.replace("edges_", "").replace("nodes_", "")
            ].value
        )
        return (
            config.kwargs["lambda"]
            * ((getattr(self.network, param) - prior) ** 2).sum()
        )


class HyperParamScheduler:
    """Schedules hyperparameters per training iteration.

    Calling the scheduler instance updates the respective hyperparameters per training
    iteration.
    """

    def __init__(
        self,
        scheduler: Namespace,
        n_iters: int,
        network: Optional[Network],
        task: Optional[Task],
        optimizer: Optional[torch.optim.Optimizer],
        penalty: Optional[Penalty],
    ):
        self.config = scheduler.deepcopy()
        self.n_iters = n_iters
        self.stop_iter = scheduler.get("sched_stop_iter", self.n_iters)
        self.network = network
        self.task = task
        self.optimizer = optimizer
        self.penalty = penalty
        self._current_iteration = 0

        pop = []
        for key, param in self.config.items():
            _function = getattr(self, param.get("function"))
            if any((a is None for a in param.values())):
                # Hyperparameters for which one of start, stop, or steps is not
                # specified are not scheduled.
                pop.append(key)
            else:
                param.array = _function(
                    self.stop_iter,
                    self.n_iters,
                    param.start,
                    param.stop,
                    param.steps,
                )

        for key in pop:
            # These hyperparameters are removed from scheduling.
            self.config.pop(key)

    def __call__(self, iteration):
        self._current_iteration = iteration
        for key, param in self.config.items():
            try:
                setattr(self, key, param.array[iteration])
            except IndexError as e:
                raise IndexError(
                    (
                        f"{e} for parameter {key}. Array is of length "
                        f"{len(param.array)}."
                        f" Scheduler was called for iteration {iteration}."
                    )
                )
        logging.info(self)

    def __repr__(self):
        return "Scheduler. Iteration: {}/{}.\nCurrent values: {}.".format(
            self._current_iteration,
            self.n_iters,
            self._params(),
        )

    def _params(self):
        params = {}
        for key, param in self.config.items():
            value = getattr(self, key)
            params[key] = value
        return params

    # -------- Decaying parameters

    @property
    def dt(self):
        return self.task.dataset.dt

    @dt.setter
    def dt(self, value):
        self.task.dataset.dt = value

    @property
    def lr_net(self):
        if self.optimizer is None:
            return
        return self.optimizer.param_groups[0]["lr"]

    @lr_net.setter
    def lr_net(self, value):
        if self.optimizer is None:
            return
        self.optimizer.param_groups[0]["lr"] = value

    @property
    def lr_dec(self):
        if self.optimizer is None:
            return
        return [param_group["lr"] for param_group in self.optimizer.param_groups[1:]]

    @lr_dec.setter
    def lr_dec(self, value):
        if self.optimizer is None:
            return
        for param_group in self.optimizer.param_groups[1:]:
            param_group["lr"] = value

    @property
    def lr_pen(self):
        if self.penalty is None:
            return
        return [
            param_group["lr"]
            for optim in self.penalty.optimizers.values()
            for param_group in optim.param_groups
        ]

    @lr_pen.setter
    def lr_pen(self, value):
        if self.penalty is None:
            return
        for optim in self.penalty.optimizers.values():
            if optim is not None:
                for param_group in optim.param_groups:
                    param_group["lr"] = value

    @property
    def relu_leak(self):
        if self.network is None:
            return
        return getattr(self.network.dynamics.activation, "negative_slope", None)

    @relu_leak.setter
    def relu_leak(self, value):
        if self.network is None:
            return
        if hasattr(self.network.dynamics.activation, "negative_slope"):
            self.network.dynamics.activation.negative_slope = value

    @property
    def activity_penalty(self):
        if self.penalty is None:
            return
        return self.penalty.activity_penalty

    @activity_penalty.setter
    def activity_penalty(self, value):
        if self.penalty is None:
            return
        self.penalty.activity_penalty = value

    # -------- Decay Options

    @staticmethod
    def linear(stop_iter, n_iterations, start, stop, steps):
        f = np.linspace(start, stop, stop_iter)
        return np.pad(f, (0, n_iterations - len(f) + 1), constant_values=stop)

    @staticmethod
    def stepwise(stop_iter, n_iterations, start, stop, steps):
        f = np.linspace(start, stop, steps).repeat(stop_iter / steps)
        return np.pad(f, (0, n_iterations - len(f) + 1), constant_values=stop)

    @staticmethod
    def stepwise_2ndhalf(stop_iter, n_iterations, start, stop, steps):
        """Decays within half of the iterations and remains constant then."""
        f = np.linspace(start, stop, steps).repeat((stop_iter / 2) / steps)
        return np.pad(f, (n_iterations - len(f) + 1, 0), constant_values=start)

    @staticmethod
    def stepwise_half(stop_iter, n_iterations, start, stop, steps):
        """Decays within half of the iterations and remains constant then."""
        f = np.linspace(start, stop, steps).repeat((stop_iter / 2) / steps)
        return np.pad(f, (0, n_iterations - len(f) + 1), constant_values=stop)

    @staticmethod
    def steponential(stop_iter, n_iterations, start, stop, steps):
        x = (1 / stop) ** (1 / steps)
        values = start / x ** np.arange(steps)
        f = values.repeat(stop_iter / steps)
        return np.pad(f, (0, n_iterations - len(f) + 1), constant_values=values[-1])

    @staticmethod
    def steponential_inv(stop_iter, n_iterations, start, stop, steps):
        _start = steps
        _stop = 0
        x = 1 / _stop
        values = _start / x ** np.arange(steps)
        f = values.repeat(stop_iter / steps)
        return np.pad(f, (0, n_iterations - len(f) + 1), constant_values=values[-1])

    @staticmethod
    def exponential(stop_iter, n_iterations, start, stop, steps):
        tau = -stop_iter / (np.log(stop + 1e-15) - np.log(start))
        f = start * np.exp(-np.arange(stop_iter) / tau)
        return np.pad(f, (0, n_iterations - len(f) + 1), constant_values=stop)

    @staticmethod
    def exponential_half(stop_iter, n_iterations, start, stop, steps):
        tau = -int((stop_iter / 2)) / (np.log(stop) - np.log(start))
        f = start * np.exp(-np.arange(int(stop_iter / 2)) / tau)
        return np.pad(f, (0, n_iterations - len(f) + 1), constant_values=stop)
