from itertools import product
import traceback
import functools
import logging
from typing import Iterable
from dvs.analysis.strfs import patch_known_fastslow

logging = logging.getLogger("dvs")
from toolz import valmap

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import torch
from torch._C import dtype
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_hex, hex2color


from dvs import exp_path_context, exp_dir
from dvs import utils
from dvs.datasets.wraps.network_wrap import init_network_wrap
from dvs import solver
from dvs.datasets import (
    Flashes,
    Gratings,
    EyalDataSet,
    Movingbar,
    OrientedBar,
    Dots,
    AugmentedPaddedSintelLum,
)
from dvs.utils.activity_utils import CentralActivity, LayerActivity
from dvs import animations
from dvs.plots import plots, plt_utils, decoration
from dvs.analysis import ConnectomeViews, ResponseTimes
import dvs.analysis
from dvs.utils.datawrapper import Datawrap
from dvs import initialization_v2
from dvs.analysis.mei import NMEI, plot_stim_response
from dvs.utils import hex_rows, datawrapper
from dvs.utils.hex_utils import HexLattice, pad_to_regular_hex
from dvs.datasets.flicker import RectangularFlicker
from dvs.utils.wrap_utils import write_meta

# ------------------------ # Trained Network Views # ------------------------- #
# -- error handling utility


class NetworkViews(ConnectomeViews):
    """Visualization frontend for looking into the trained networks.

    Args:
        name (str): recovers the tnn from dvs.get_root_dir() / name. Make sure to
            set the root dir via dvs.set_root_dir() correctly.
        config (Mapping, Optional): the config of the network can be used, too, to recover
            the tnn.
        tnn (Datawrap): the tnn can also be specified directly.
        init_network (bool): initialize nn.Module as attribute self.network.
        init_response_times (bool): initialize response time fitter as attribute
            self.response_times and learned time constant object as self.tau.
        init_gratings (bool): init reference to default gratings dataset
            handling input and responses.
        init_flashes (bool): init reference to default flashes dataset
            handling input and responses.
        init_eyal_data (bool): init reference to eyal dataset
            handling input and responses.
        failsafe (bool): do not stop if initializing the default stimuli data
            fails because e.g. they were not recorded. Defaults to True.

    Attributes:
        tnn (Datawrap): wrap of the trained dvs neural network.

        Optional:

        network (nn.Module)
        dt (float)
        response_time (object): callable, e.g. response_time("T4a").
        tau (object): attribute style mapper, e.g. tau.T4a.

        gratings_dataset (Dataset)
        gratings_spec (dict)
        gratingsresponse (dict)

        flashes_dataset (Dataset)
        flashes_spec (dict)
        flashesresponse (dict)

        eyal_dataset (Dataset)
    """

    def __init__(
        self,
        name=None,
        config=None,
        tnn=None,
        init_network=False,
        init_response_times=False,
        init_gratings=False,
        init_flashes=False,
        init_movingbar=False,
        init_eyal_data=False,
        failsafe=True,
        size_check=False,
    ):

        with exp_path_context():
            _size_check = datawrapper.get_check_size_on_init()
            datawrapper.check_size_on_init(size_check)
            self.tnn, _ = init_network_wrap(name, config, tnn)
            datawrapper.check_size_on_init(_size_check)

        self.name = str(self.tnn.path).replace(str(exp_dir) + "/", "")

        if init_network or init_response_times:
            self.init_network()
            if init_response_times:
                self.response_time = ResponseTimes(self.network, 1 / 200)
                self.tau = initialization_v2.TimeConstant(
                    self.network.ctome, requires_grad=False
                )
                self.tau.update(
                    dict(self.network.named_parameters())["nodes_time_const"]
                )

        super().__init__(self.tnn)

        # currently the logic is to load the trained network wrap
        # mostly loaded and stimuli/ response datasets are tracke here
        # because these datasets can be in different subwraps, this is a dictionary
        # of stim/response dataset 'type': subwrap name
        self._initialized = {
            "gratings": "",
            "flashes": "",
            "eyal_data": "",
            "network": "",
            "tau": "",
            "response_time": "",
            "movingbar": "",
            "dots": "",
            "augmented_sintel": "",
            "oriented_bar": "",
            "mei": "",
            "normalizing_movingbar": "",
            "rectangular_flicker": "",
            "solver": "",
            "decoder": "",
            "rnmeis": "",
            "nmeis": "",
        }
        if init_gratings:
            self.init_gratings(failsafe=failsafe)
        if init_flashes:
            self.init_flashes(failsafe=failsafe)
        if init_eyal_data:
            self.init_eyal_data(failsafe=failsafe)
        if init_movingbar:
            self.init_movingbar(failsafe=failsafe)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"

    def arg_df(self):
        return utils.config_to_flat_df(self.tnn.spec, name=self.name)

    def init_network(
        self,
        checkpoint="best",
        validation_subwrap="original_validation_v2",
        validation_loss_name="epe",
    ):
        self.network, self.dt, _ = solver.trained_network(
            self.tnn.path,
            checkpoint,
            validation_subwrap=validation_subwrap,
            validation_loss_name=validation_loss_name,
        )
        self._initialized["network"] = True
        return self.network

    def init_decoder(self, checkpoint="best"):
        self.decoder, _ = solver.trained_decoder(self.tnn.path, checkpoint)
        self._initialized["decoder"] = True
        return self.decoder

    def init_solver(self, checkpoint="best"):
        with exp_path_context():
            self.solver = dvs.solver.MultiTaskSolver(self.name)
        self.solver.recover(checkpoint=checkpoint)
        self._initialized["solver"] = True

    def init_tau(self):
        self.tau = initialization_v2.TimeConstant(
            self.network.ctome, requires_grad=False
        )
        self.tau.update(dict(self.network.named_parameters())["nodes_time_const"])

    def init_response_time(self):
        self.response_time = ResponseTimes(self.network, 1 / 200)

    def _init_stim_responses(function):
        """Wrapper for failsafe loading of the NetworkViews object."""

        @functools.wraps(function)
        def _init_stim_response_(cls, *args, **kwargs):

            _kw = {key: val for key, val in kwargs.items()}
            subwrap = _kw.pop("subwrap", args[0] if args else function.__defaults__[0])
            failsafe = _kw.pop(
                "failsafe",
                args[1] if len(args) == 2 else function.__defaults__[1],
            )

            def test_exists(subwrap_name, datawrap):
                if subwrap_name not in datawrap:
                    raise ValueError(
                        f"subwrap {subwrap} unknown.\n"
                        "This probably means a typo or that the stimulus responses "
                        "have not been recorded yet."
                        f"\nPossible subwraps are {list(datawrap.keys())}"
                    )
                return datawrap[subwrap_name]

            _subwraps = subwrap.split("/")
            _datawrap = cls.tnn
            for subwrap_name in _subwraps:
                _datawrap = test_exists(subwrap_name, _datawrap)

            fn_name = function.__name__.replace("init_", "")
            if cls._initialized[fn_name] == subwrap:
                return

            def _has_stored_states(cls, subwrap, failsafe=True):
                if not (
                    hasattr(cls.tnn, subwrap)
                    and hasattr(cls.tnn[subwrap], "network_states")
                ):
                    err = f"{cls.tnn.path.name} has no {subwrap}"
                    if failsafe:
                        logging.warning(err)
                    else:
                        raise ValueError(err)

            _has_stored_states(cls, subwrap, failsafe)

            try:
                function(cls, *args, **kwargs)
            except Exception as Error:
                err = f"init {subwrap} failed for {cls.name}"
                if failsafe:
                    logging.warning(err)
                else:
                    logging.warning(err)
                    raise Error
                traceback.print_exc()

        return _init_stim_response_

    @_init_stim_responses
    def init_mei(self, subwrap="augmented_sintel", failsafe=True):
        if self._initialized["mei"] == subwrap:
            return

        self.mei = NMEI(
            self.name,
            dt=1 / 100,
            subwrap=subwrap,
            dt_sampling=True,
            nnv=self,
            delete_if_exists=False,
        )
        self._initialized["mei"] = subwrap

    @_init_stim_responses
    def init_gratings(self, subwrap="gratings", failsafe=True, **kwargs):

        config = self.tnn[subwrap].spec
        # config.update(dt=self.tnn.dt[()])
        self.gratings_dataset = Gratings(**config, tnn=self.tnn, chkpt_type=subwrap)
        self.gratings_spec = config
        self.gratingsresponse = CentralActivity(
            self.gratings_dataset.response(), self.ctome, keepref=True
        )
        self._initialized["gratings"] = subwrap

    @_init_stim_responses
    def init_flashes(self, subwrap="flashes", failsafe=True, **kwargs):

        config = self.tnn[subwrap].spec
        # config.update(dt=self.tnn.dt[()])
        self.flashes_dataset = Flashes(**config, subwrap=subwrap, tnn=self.tnn)
        self.flashes_spec = config
        self.flashresponse = self.flashes_dataset.centralactivity
        self._initialized["flashes"] = subwrap

    @_init_stim_responses
    def init_movingbar(
        self,
        subwrap="movingedge_chkpt_best_v4",
        failsafe=True,
        device="cpu",
        **kwargs,
    ):
        config = self.tnn[subwrap].spec
        self.movingbar = Movingbar(
            **config, tnn=self.tnn, subwrap=subwrap, device=device, **kwargs
        )
        self._initialized["movingbar"] = subwrap

    @_init_stim_responses
    # extra method because used a normalizer together with a movingbar stimulus
    def init_normalizing_movingbar(
        self,
        subwrap="shuffled_movingsquare",
        failsafe=True,
        device="cpu",
        **kwargs,
    ):
        """To be able to initialize two movingbar stim-response instances, e.g.
        if one is used for normalization of the other's responses."""
        config = self.tnn[subwrap].spec
        self.normalizing_movingbar = Movingbar(
            **config, tnn=self.tnn, subwrap=subwrap, device=device, **kwargs
        )
        self._initialized["normalizing_movingbar"] = subwrap

    @_init_stim_responses
    # extra method because used a normalizer together with a movingbar stimulus
    def init_rectangular_flicker(
        self,
        subwrap="square_flicker",
        failsafe=True,
        device="cpu",
        **kwargs,
    ):
        config = self.tnn[subwrap].spec
        self.rectangular_flicker = RectangularFlicker(
            **config, tnn=self.tnn, subwrap=subwrap, device=device, **kwargs
        )
        self._initialized["rectangular_flicker"] = subwrap

    @_init_stim_responses
    def init_oriented_bar(
        self,
        subwrap="oriented_bar",
        failsafe=True,
        device="cpu",
        **kwargs,
    ):
        config = self.tnn[subwrap].spec
        self.oriented_bar = OrientedBar(
            **config, tnn=self.tnn, subwrap=subwrap, device=device, **kwargs
        )

        _exp_time = config.t_pre + config.t_stim + config.t_post
        _timesteps = self.tnn[subwrap].network_states.nodes.activity_central[:].shape[1]

        if not np.isclose(int(_exp_time / config.dt), _timesteps):
            logging.info(
                f"dt {config.dt} in config and timesteps of responses {_timesteps} not consistent for experiment time {_exp_time}s\n"
                f"changing self.oriented_bar.dt to {_exp_time / _timesteps}"
            )
            self.oriented_bar.dt = _exp_time / _timesteps
            config.dt = _exp_time / _timesteps
            write_meta(
                self.tnn[subwrap].path,
                dict(spec=config, status="done"),
            )
            logging.info("meta updated")

        self._initialized["oriented_bar"] = subwrap

    @_init_stim_responses
    def init_augmented_sintel(
        self, subwrap="augmented_sintel", failsafe=True, device="cpu", **kwargs
    ):
        if self._initialized["augmented_sintel"] == subwrap:
            return

        # if subwrap not in self.tnn:
        #     raise ValueError(
        #         f"{subwrap} unknown - possible subwraps are {list(self.tnn.keys())}"
        #     )
        # TODO: does not yet store the config!
        # config = self.tnn[key].spec
        self.augmented_sintel = AugmentedPaddedSintelLum(device=device, **kwargs)
        self.augmented_sintel.init_responses(self.tnn, subwrap)
        self._initialized["augmented_sintel"] = subwrap

    @_init_stim_responses
    def init_dots(self, subwrap="impulse_v2", failsafe=True, device="cpu", **kwargs):
        if self._initialized["dots"] == subwrap:
            return

        # if subwrap not in self.tnn:
        #     raise ValueError(
        #         f"{subwrap} unknown - possible subwraps are {list(self.tnn.keys())}"
        #     )

        config = self.tnn[subwrap].spec
        config.update(subwrap=subwrap)
        self.dots = Dots(**config, device=device)
        self.dots._init_tnn(self.tnn)
        self._initialized["dots"] = subwrap

    @_init_stim_responses
    def init_rnmeis(self, subwrap="rnmeis", failsafe=True):
        if self._initialized["rnmeis"] == subwrap:
            return
        self._initialized["rnmeis"] = ""
        self.rnmeis = self.tnn[subwrap]
        self._initialized["rnmeis"] = subwrap

    @_init_stim_responses
    def init_nmeis(self, subwrap="nmeis", failsafe=True):
        if self._initialized["nmeis"] == subwrap:
            return
        self._initialized["nmeis"] = ""
        self.nmeis = self.tnn[subwrap]
        self._initialized["nmeis"] = subwrap

    def init_eyal_data(self):
        if hasattr(self.tnn.meta.spec, "task"):
            config = self.tnn.meta.spec.task.dataset
        elif hasattr(self.tnn.meta.spec, "conditions"):
            config = self.tnn.meta.spec.conditions.recordings.dataset
        if config.pop("type") == "EyalDataSet" and hasattr(self.tnn, "full_data_eval"):
            config.update(predict_mean=False, device="cpu")
            self.eyal_dataset = EyalDataSet(**config)
        self._initialized["eyal_data"] = True

    def _peak_responses_for_msi_from_flicker(
        self,
        subwrap="movingsquare",
        flicker_subwrap="square_flicker",
        nonlinearity=True,
        subtract_baseline=True,
    ):
        self.init_movingbar(subwrap=subwrap)
        self.init_rectangular_flicker(subwrap=flicker_subwrap)

        peak_responses, _ = self.movingbar.peak_response(
            nonlinearity=nonlinearity,
            subtract_baseline=subtract_baseline,
            pre_stim=False,
            post_stim=False,
        )

        peak_responses_other = self.rectangular_flicker.peak_response(
            other_to_match=self.movingbar,
            nonlinearity=nonlinearity,
            subtract_baseline=subtract_baseline,
            pre_stim=False,
            post_stim=False,
        )

        return peak_responses, peak_responses_other

    def _peak_responses_edges_for_msi_from_flicker(
        self,
        subwrap="movingedge_chkpt_best_v4",
        flicker_subwrap="edge_flicker",
        nonlinearity=True,
        subtract_baseline=True,
    ):
        self.init_movingbar(subwrap=subwrap)
        self.init_normalizing_movingbar(subwrap=flicker_subwrap)

        peak_responses, _ = self.movingbar.peak_response(
            nonlinearity=nonlinearity,
            subtract_baseline=subtract_baseline,
            pre_stim=False,
            post_stim=False,
        )

        peak_responses_other, _ = self.normalizing_movingbar.peak_response(
            nonlinearity=nonlinearity,
            subtract_baseline=subtract_baseline,
            pre_stim=False,
            post_stim=False,
        )

        return peak_responses, peak_responses_other

    def _peak_responses_edges_plus_naturalistic(
        self,
        subwrap="movingedge_chkpt_best_v4",
        flicker_subwrap="edge_flicker",
        naturalistic_subwrap="augmented_sintel",
        nonlinearity=True,
        subtract_baseline=True,
    ):

        r, q = self._peak_responses_edges_for_msi_from_flicker(
            subwrap, flicker_subwrap, nonlinearity, subtract_baseline
        )
        self.init_augmented_sintel(subwrap=naturalistic_subwrap)

        responses = self.augmented_sintel.response()
        if nonlinearity:
            responses = np.maximum(responses, 0)
        if subtract_baseline:
            responses -= responses[:, 0][:, None]
        s = np.nanmax(responses, axis=1)
        return r, q, s

    def _peak_responses_for_msi_from_flicker_and_naturalistic(
        self,
        subwrap="movingsquare",
        flicker_subwrap="square_flicker",
        naturalistic_subwrap="augmented_sintel",
        nonlinearity=True,
        subtract_baseline=True,
    ):
        r, q = self._peak_responses_for_msi_from_flicker(
            subwrap, flicker_subwrap, nonlinearity, subtract_baseline
        )
        self.init_augmented_sintel(subwrap=naturalistic_subwrap)

        responses = self.augmented_sintel.response()
        if nonlinearity:
            responses = np.maximum(responses, 0)
        if subtract_baseline:
            responses -= responses[:, 0][:, None]
        s = np.nanmax(responses, axis=1)

        return r, q, s

    def _peak_responses_for_msi_from_shuffled(
        self,
        subwrap="movingsquare",
        normalizing_subwrap="shuffled_movingsquare",
    ):
        self.init_movingbar(subwrap=subwrap)
        self.init_normalizing_movingbar(subwrap=normalizing_subwrap)

        peak_responses, _ = self.movingbar.peak_response(
            nonlinearity=True,
            subtract_baseline=True,
            pre_stim=False,
            post_stim=False,
        )

        peak_responses_other, _ = self.normalizing_movingbar.peak_response(
            nonlinearity=True,
            subtract_baseline=True,
            pre_stim=False,
            post_stim=False,
        )

        return peak_responses, peak_responses_other

    def msi_v1(self, nonlinearity=True, aggregate=True):
        """max(r - q)"""
        r, q = self._peak_responses_for_msi_from_flicker(nonlinearity=nonlinearity)
        msi = r - q
        if aggregate:
            msi = msi.max(axis=(0, 1, 2, 3))
        return self.node_types_sorted, msi

    def msi_v2(self, nonlinearity=True, aggregate=True):
        """max(r - q) / max max(q)"""
        r, q = self._peak_responses_for_msi_from_flicker(nonlinearity=nonlinearity)
        # if np.maximum(r, q).max(axis=(0, 1, 2, 3)) this value is simply 1
        # for any cell for which a q = 0
        # this way it is the maximum amount of spatio-temporal integration in relation to
        # the maximum amount of temporal integration
        denominator = q.max(axis=(0, 1, 2, 3))
        msi = (r - q) / denominator
        if aggregate:
            msi = msi.max(axis=(0, 1, 2, 3))
        return self.node_types_sorted, msi

    def msi_v3(self, nonlinearity=True, aggregate=True):
        """max([r] - [q]) / max(s)"""
        r, q, s = self._peak_responses_for_msi_from_flicker_and_naturalistic(
            nonlinearity=nonlinearity
        )
        # s_max is upper bound
        s_max = s.max()
        r = np.minimum(r, s_max)
        q = np.minimum(q, s_max)
        # s_min is lower bound
        s_min = s.min()
        r = np.maximum(r, s_min)
        q = np.maximum(q, s_min)
        msi = (r - q) / (s_max + 1e-3)
        if aggregate:
            msi = msi.max(axis=(0, 1, 2, 3))
        return self.node_types_sorted, msi

    def msi_v4(self, nonlinearity=True, aggregate=True):
        """max 2 * (r / max(r + q) - 1/2)"""
        r, q = self._peak_responses_for_msi_from_flicker(nonlinearity=nonlinearity)
        denominator = (r + q).max(axis=(0, 1, 2, 3))
        msi = 2 * (r / denominator - 1 / 2)
        if aggregate:
            msi = msi.max(axis=(0, 1, 2, 3))
        return self.node_types_sorted, msi

    def parameters(
        self,
        chkpt="best",
        validation_subwrap="original_validation_v2",
        loss_name="epe",
    ):
        network_params = {}
        if chkpt == "best":
            loss_name = dvs.analysis.validation_error._check_loss_name(
                self.tnn[validation_subwrap], loss_name
            )
            chkpt_idx = np.argmin(self.tnn[validation_subwrap][loss_name][:])
        elif chkpt == "initial":
            chkpt_idx = 0
        else:
            raise ValueError
        chkpt_params = torch.load(self.tnn.chkpts[f"chkpt_{chkpt_idx:05}"])
        for key, val in chkpt_params["network"].items():
            network_params[key] = val.cpu().numpy()
        return network_params

    def get_time_constants(
        self, mode="best", validation_subwrap="original_validation_v2"
    ):
        # TODO: check same parameter configs
        time_constants = self.parameters(
            chkpt=mode, validation_subwrap=validation_subwrap
        )["nodes_time_const"]

        return self.node_types_unsorted, time_constants

    # ---- LOSS
    def training_loss(
        self,
        fig=None,
        ax=None,
        smooth=0.005,
        mark_chkpts=True,
        fontsize=10,
        **kwargs,
    ):
        weighting = getattr(
            getattr(self.tnn.spec, "task", None), "task_weighting", None
        )
        title = (
            f"training (including weighting: {weighting})"
            if weighting is not None
            else "training"
        )
        xlabel = "iterations"
        loss = self.tnn.loss[:]
        x = np.arange(len(loss))
        fig, ax, smoothed_loss, _ = plots.traces(
            self.tnn.loss[:],
            x=x,
            ylabel="loss",
            xlabel=xlabel,
            smooth=smooth,
            fig=fig,
            ax=ax,
            fontsize=fontsize,
            **kwargs,
        )

        if mark_chkpts:
            chkpt_iter = self.tnn.chkpt_iter[:]
            ax.vlines(
                chkpt_iter,
                ymin=smoothed_loss.min(),
                ymax=smoothed_loss.max(),
                zorder=0,
                color="0.9",
            )
        ax.set_title(title, fontsize=fontsize)
        return fig, ax

    def test_loss(
        self,
        mode="validation",
        fig=None,
        ax=None,
        smooth=None,
        mark_chkpts=True,
        fontsize=10,
        mean=False,
        legend=True,
        weights=None,
        **kwargs,
    ):
        loss = []
        if mean:
            loss.append(self.tnn[mode].loss[:])

        task_loss_names = []
        for p in sorted(self.tnn[mode].path.glob("*loss_*")):
            task_loss_names.append(p.name.replace(".h5", ""))

        if weights is None:
            weights = (1,) * len(task_loss_names)

        for i, t in enumerate(task_loss_names):
            loss.extend([weights[i] * self.tnn[mode][t][:]])
        loss = np.array(loss)

        legends = []
        if legend:
            if mean:
                legends = ["mean over tasks"]
            for t in task_loss_names:
                legends.extend(t.replace("loss_", "").capitalize())

        chkpt_iter = self.tnn.chkpt_iter
        iters_per_chkpts = (chkpt_iter[1:] - chkpt_iter[:-1])[len(chkpt_iter) // 2]
        x = np.arange(loss.shape[1]) * iters_per_chkpts
        fig, ax, smoothed_loss, _ = plots.traces(
            loss,
            x=x,
            ylabel="loss",
            xlabel="iterations",
            smooth=smooth,
            legend=legends,
            fig=fig,
            ax=ax,
            fontsize=fontsize,
            **kwargs,
        )
        if mark_chkpts:
            ax.vlines(
                chkpt_iter,
                ymin=smoothed_loss.min(),
                ymax=smoothed_loss.max(),
                zorder=0,
                color="0.9",
            )
        ax.set_title(f"{mode} loss", fontsize=fontsize)
        return fig, ax

    def loss(
        self,
        mode,
        fig=None,
        ax=None,
        smooth=0.0,
        mark_chkpts=True,
        mean=True,
        weights=None,
        legend=True,
        fontsize=10,
        **kwargs,
    ):

        xlabel = "checkpoints"
        if mode == "train":
            mode = ""
            weighting = getattr(
                getattr(self.tnn.spec, "task", None), "task_weighting", None
            )
            title = (
                f"during training (including weighting: {weighting})"
                if weighting is not None
                else "during training"
            )
            xlabel = "iterations"
        elif mode == "tracked_train_batch":
            title = f"tracked training batch"
        elif "validation" in mode:
            title = f"validation set"
        elif "training" in mode:
            title = f"training set"
        else:
            raise ValueError(
                'mode must be one of "train", "tracked_train_batch", "validation"'
            )

        loss = []
        if mean == True:
            loss.append(self.tnn[mode].loss[:])

        task_loss_names = []
        for p in sorted(self.tnn[mode].path.glob("*loss_*")):
            task_loss_names.append(p.name.replace(".h5", ""))

        if weights is None:
            weights = (1,) * len(task_loss_names)

        for i, t in enumerate(task_loss_names):
            loss.extend([weights[i] * self.tnn[mode][t][:]])

        legends = []
        if legend:
            if mean:
                legends = ["mean over tasks"]
            for t in task_loss_names:
                legends.extend(t.replace("loss_", "").capitalize())

        # no legend in this case
        if len(loss) == 2:
            loss, legends = (loss[0], [])

        fig, ax, smoothed_loss, _ = plots.traces(
            loss,
            ylabel="loss",
            xlabel=xlabel,
            smooth=smooth,
            legend=legends,
            fig=fig,
            ax=ax,
            fontsize=fontsize,
            **kwargs,
        )

        chkpts = np.array(
            [
                int(p.name.replace("chkpt_", ""))
                for p in sorted(self.tnn.chkpts.path.glob("chkpt_*"))
            ]
        )
        if chkpts.size != 0 and mark_chkpts:
            if mode == "":
                # TODO newer wraps store chkpt_iter, old ones don't.
                _chkpts = self.tnn.chkpt_iter if "chkpt_iter" in self.tnn else []
                ax.plot(_chkpts, smoothed_loss[0][_chkpts], "|", color="k", ms=10)
            else:
                ax.plot(chkpts, smoothed_loss[0][chkpts], "|", color="k", ms=10)
        ax.set_title(title, fontsize=fontsize)
        return fig, ax, smoothed_loss

    @plt_utils.nbAgg
    def watch_loss(
        self,
        mode,
        smooth=0.1,
        chkpts=True,
        iters=True,
        fig=None,
        ax=None,
        fontsize=6,
        t_sleep=2,
        **kwargs,
    ):
        from time import sleep

        fig, ax, _ = self.loss(mode, smooth=smooth, fontsize=fontsize, **kwargs)

        plt.show()

        def update_loss(loss):
            if smooth is not None:
                loss = plt_utils.filter_trace(
                    np.array([loss]), int(len(loss) * smooth)
                )[0]
            iters = np.arange(0, len(loss))
            ax.lines[0].set_data(iters, loss)
            if loss.any():
                ymax = np.max(loss)
                ymin = np.min(loss)
            ax.axis([0, iters[-1], ymin, ymax])

        def update_val_losses(loss):
            chkpts = (
                np.arange(len(loss)) if len(ax.lines) == 2 else np.arange(len(loss[0]))
            )
            ymin, ymax = 1, 0
            for i, line in enumerate(ax.lines):
                loss = loss if len(ax.lines) == 2 else loss[i]
                if smooth is not None:
                    ax.lines[i].set_data(
                        chkpts,
                        plt_utils.filter_trace(
                            np.array([loss]), int(len(loss) * smooth)
                        )[0],
                    )
                else:
                    ax.lines[i].set_data(chkpts, loss)
                ymin = min(ymin, loss.min())
                ymax = max(ymax, loss.max())
            ax.axis([0, chkpts[-1], ymin, ymax])

        while True:
            if mode == "train":
                loss = self.tnn.loss[:]
                xlabel = "Iterations"
                legends = []
                update_loss(loss)
            elif mode == "tracked_train_batch":
                loss = self.tnn[mode].loss[:]
                xlabel = "Checkpoints"
                legends = []
                update_loss(loss)
            elif mode == "validation":
                loss = [self.tnn[mode].loss[:]]
                task_losses = [
                    p.name.replace(".h5", "")
                    for p in sorted(self.tnn.validation.path.glob("*loss_*"))
                ]
                loss.extend([self.tnn[mode][t] for t in task_losses])
                legends = [
                    "Mean",
                    *[t.replace("loss_", "").capitalize() for t in task_losses],
                ]
                loss, legends = (loss[0], []) if len(loss) == 2 else (loss, legends)
                xlabel = "Checkpoints"
                update_val_losses(loss)
            fig.canvas.draw()
            fig.canvas.flush_events()
            sleep(t_sleep)

    # ---- ACTIVITY

    def activity(self, **kwargs):
        """
        Plots the activity over iterations.

        Args:
            **kwargs: keyword arguments.
        """
        activity = self.tnn.activity[:]
        fig, ax, smooth_activity, _ = plots.traces(
            activity, ylabel="Activity", **kwargs
        )
        return fig, ax, smooth_activity

    def activity_bars(self, batch_type="full_val", figsize=(9, 5), **kwargs):
        """
        Plots activity bars.

        Args:
            batch_sample (int)
            batch_type_aggr: how to aggregate the activity over batch_types.
            batch_type (str or list): which activity to look at.
            **kwargs: arbitrary keyword arguments.
        """
        relu = lambda x: np.maximum(x, 0)

        central_activity = CentralActivity(
            self.tnn[batch_type].network_states.nodes.activity_central,
            self.ctome,
            keepref=True,
        )

        # Mean over samples, frames, hexals.
        means = [relu(central_activity[nt]).mean() for nt in self.node_types_sorted]
        # Std over samples, frames, hexals.
        stds = [relu(central_activity[nt]).std() for nt in self.node_types_sorted]

        fig, ax, _ = plots.bars(
            self.node_types_sorted,
            means,
            std=stds,
            title="Activity",
            figsize=figsize,
            **kwargs,
        )

        return fig, ax

    # ---- PARAMETER SCATTER

    def node_param_scatter(self, scale=5, fig=None, fontsize=8, **kwargs):
        """
        Plots the prior node params vs. the trained node params.
        """
        node_params = list(self.tnn.spec.network.node_config.keys())

        fig, axes, _ = plt_utils.get_axis_grid(
            range(len(node_params)), scale=scale, fig=fig
        )

        for i, p in enumerate(node_params):
            parameter = self.nodes[p + "_prior"]
            trained = self.nodes[p + "_trained"]
            plots.scatter(
                parameter,
                trained,
                fig=fig,
                ax=axes[i],
                xlabel="untrained " + p.replace("_", " "),
                ylabel="trained " + p.replace("_", " "),
                fontsize=fontsize,
                **kwargs,
            )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig, axes

    def edge_param_scatter(self, scale=5, fig=None, fontsize=8, **kwargs):
        """Plots the prior edge params vs. the trained node params."""
        edge_params = list(self.tnn.spec.network.edge_config.keys())
        edge_params.append("weight")

        fig, axes, _ = plt_utils.get_axis_grid(
            range(len(edge_params)), scale=scale, fig=fig
        )

        for i, p in enumerate(edge_params):
            parameter = self.edges[p + "_prior"]
            trained = self.edges[p + "_trained"]
            plots.scatter(
                parameter,
                trained,
                fig=fig,
                ax=axes[i],
                xlabel="untrained " + p.replace("_", " "),
                ylabel="trained " + p.replace("_", " "),
                fontsize=fontsize,
                **kwargs,
            )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig, axes

    def time_constants(self):

        if not self._initialized["tau"]:
            self.init_tau()
        if not self._initialized["response_time"]:
            self.init_response_time()

        times = []

        for node_type in self.node_types_sorted:
            times.append([self.tau[node_type].item(), self.response_time(node_type)[0]])

        times = np.ma.masked_invalid(times)
        fig, ax = plots.violin_groups(
            times.data * 1e3,
            xticklabels=self.node_types_sorted,
            rotation=90,
            as_bars=True,
            width=0.7,
            colors=["r", "b"],
            legend=["τ learned", "response time"],
            legend_kwargs=dict(fontsize=10),
            ylabel="time in ms",
        )
        ax.annotate(
            f"correlation: {np.ma.corrcoef(times.T).data[0, 1]:.2G}",
            (1, 1),
            fontsize=10,
            ha="right",
            va="bottom",
            xycoords="axes fraction",
            zorder=9,
        )
        return fig, ax

    # ---- PROBES RESPONSES

    def postsynaptic_current_animation(
        self,
        target_type,
        dataset,
        dt,
        indices: list,
        figsize=[22, 13],
        fontsize=9,
        gridheight=3,
        hspace=0.4,
        wspace=0.1,
        trace_vertical_ratio=0.3,
    ):
        self.init_network()
        stimulus, source_currents, responses = self.network.current_response(
            dataset, dt=dt, indices=indices, t_pre=0, t_fade_in=1.0
        )

        def _prepare_for_target_animation(source_currents, responses):
            rfs, max_extent = self.receptive_fields_df(target_type)
            layer_index = dvs.utils.LayerActivity(None, self.ctome, keepref=True)

            layer_index.update(responses)
            responses = layer_index[target_type]

            input_currents = {}
            for source_type, rf in rfs.items():
                u, v = rf.source_u.values, rf.source_v.values
                _, _, input_currents[source_type] = pad_to_regular_hex(
                    u,
                    v,
                    source_currents[:, :, rf.index],
                    extent=max_extent,
                    value=0,
                )

            # to derive the summed currents in the dendrite coordinate system, can now simply sum all
            target_currents = sum([v for v in input_currents.values()])

            return input_currents, target_currents, responses

        return animations.activations.ActivationGridPlusTraces_v2(
            target_type,
            stimulus,
            *_prepare_for_target_animation(source_currents, responses),
            dt=dt,
            figsize=figsize,
            fontsize=fontsize,
            gridheight=gridheight,
            wspace=wspace,
            hspace=hspace,
            trace_vertical_ratio=trace_vertical_ratio,
            path=self.tnn.path / "animations",
        )

    def stimuli_responses(self, node_type, figsize=[20, 10], fig=None, **kwargs):
        """
        Plots gratings and flash responses.
        """
        if not self._initialized["gratings"] or not self._initialized["flashes"]:
            self.init_gratings()
            self.init_flashes()

        projections = {i: "polar" for i in range(1, 29)}
        projections.update({i: None for i in range(29, 33)})
        projections.update({33: "polar", 34: None})
        fig, axes = plt_utils.divide_figure_to_grid(
            matrix=[
                [8, 9, 10, 11, 12, 13, 14],  ## Swapped OFF<>ON
                [1, 2, 3, 4, 5, 6, 7],
                [np.nan, 33, 33, np.nan, 34, 34, np.nan],
                [np.nan, 29, 29, np.nan, 31, 31, np.nan],
                [np.nan, 30, 30, np.nan, 32, 32, np.nan],
            ],
            fig=fig,
            projection=projections,
            figsize=figsize,
            wspace=0.3,
            hspace=0.7,
        )

        _grat_axes = [axes[i] for i in range(1, 15)]
        _flash_axes = [axes[i] for i in range(29, 33)]

        # GRATINGS
        samples = dict(
            n_bars=[1],
            dynamic_range=self.gratings_spec.dynamic_range,
            stim_width=self.gratings_spec.stim_width,
        )
        params = list(product(*(v for v in samples.values())))

        theta = np.array(self.gratings_spec.angles)
        background = sum(self.gratings_spec.dynamic_range) / 2

        activity = self.gratingsresponse[node_type].max(axis=1)
        ymin, ymax = activity.min(), activity.max()
        speed = np.array(self.gratings_spec.speed)
        for i, (nbars, intensity, width) in enumerate(params):
            mask = self.gratings_dataset.mask(
                angle=None,
                nbars=nbars,
                width=width,
                speed=None,
                intensity=intensity,
            )
            _activity = activity[mask].squeeze()
            cbar = True if i == 13 else False
            ylabel = (
                (
                    f"ON\n{nbars} Bars"
                    if intensity > background
                    else f"OFF\n{nbars} Bars"
                )
                if i in [0, 7, 14, 21]
                else ""
            )
            xlabel = (
                f"Bar width: {width} col." if i in [0, 7, 14, 21] else f"{width} col."
            )
            plots.speed_polar(
                theta,
                _activity.reshape(
                    len(self.gratings_spec.angles),
                    len(self.gratings_spec.speed),
                ).T,
                speed,
                ax=_grat_axes[i],
                fig=fig,
                ymin=ymin,
                ymax=ymax,
                cbar=cbar,
                ylabel=ylabel,
                xlabel=xlabel,
            )

        # Argmax DSI
        self.motion_tuning(node_type, fig=fig, ax=axes[33])
        self.gratings_trace(node_type, fig=fig, ax=axes[34])

        # FLASHES
        for i, ((baseline, intensity), radius) in enumerate(
            self.flashes_dataset.values
        ):
            ylabel = (
                ("Activity" if radius == -1 else "Activity\n(Circle Stim.)")
                if i in [0, 1]
                else ""
            )
            xlabel = "Time in s" if i in [1, 3] else ""
            self.flash_response(
                node_type,
                baseline,
                intensity,
                radius,
                ax=_flash_axes[i],
                fig=fig,
                title="",
                ylabel=ylabel,
                xlabel=xlabel,
            )

        return fig, axes

    def pathway_stimuli_response(self, fig=None, pathway="on"):

        if not self._initialized["movingbar"] or not self._initialized["flashes"]:
            self.init_movingbar()
            self.init_flashes()
        pathway = pathway.lower()
        projections = {i: "polar" for i in range(0, 20)}
        projections.update({i: None for i in range(4, 20)})
        fig, axes = plt_utils.divide_figure_to_grid(
            matrix=[
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [np.nan, np.nan, np.nan, np.nan],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, np.nan],
            ],
            constrained_layout=True,
            projection=projections,
            fig=fig,
            figsize=[11, 10],
            wspace=0.5,
            hspace=1,
        )
        # plt.tight_layout()

        mt_nodes = (
            ["T4a", "T4b", "T4c", "T4d"]
            if pathway == "on"
            else ["T5a", "T5b", "T5c", "T5d"]
        )

        width = 4
        if "edge" in self.movingbar.subwrap:
            width = 80

        cmap = cm.get_cmap("tab10")
        for i, node_type in enumerate(mt_nodes):
            if pathway == "on":
                self.motion_tuning(
                    node_type,
                    intensity=[1],
                    xlabel=node_type,
                    fig=fig,
                    color=cmap(i),
                    ax=axes[i],
                )
                self.movingbar_response(
                    node_type,
                    fig=fig,
                    ax=axes[i + 4],
                    xlabel="time in s" if i == 0 else " ",
                    width=width,
                    speed=19,
                    intensity=1,
                )
            elif pathway == "off":
                self.motion_tuning(
                    node_type,
                    intensity=[0],
                    xlabel=node_type,
                    fig=fig,
                    color=cmap(i),
                    ax=axes[i],
                )
                self.movingbar_response(
                    node_type,
                    fig=fig,
                    ax=axes[i + 4],
                    xlabel="time in s" if i == 0 else " ",
                    width=width,
                    speed=19,
                    intensity=0,
                )

        if pathway == "on":
            other_nodes = [
                "L1",
                "L3",
                "L5",
                "Mi1",
                "Tm3",
                "Mi4",
                "Mi9",
                "T4a",
                "T4b",
                "T4c",
                "T4d",
            ]
        else:
            other_nodes = [
                "L2",
                "L3",
                "L4",
                "Tm1",
                "Tm2",
                "Tm4",
                "Tm9",
                "T5a",
                "T5b",
                "T5c",
                "T5d",
            ]

        for i, node_type in enumerate(other_nodes):
            self.flash_response(
                node_type,
                baseline=0,
                intensity=1,
                radius=6,
                xlabel="",
                fig=fig,
                ylabel="activity" if i == 0 else "",
                ax=axes[i + 8],
            )

        return fig, axes

    # ---- FLASH RESPONSES

    def flash_response(
        self,
        node_type,
        intensity,
        radius,
        xlabel="time (s)",
        title=None,
        subwrap="flashes",
        **kwargs,
    ):
        if not self._initialized["flashes"] == subwrap:
            self.init_flashes(subwrap=subwrap)
        # Get activity for given parameters and node type.
        mask = self.flashes_dataset.mask(intensity, radius)
        response = self.flashresponse[node_type][mask].squeeze()

        # Create vector of binary conditions over time.
        t_steps_pre = int(self.flashes_spec.t_pre / self.flashes_spec.dt)
        t_steps_stim = int(self.flashes_spec.t_stim / self.flashes_spec.dt)

        dynamic_range = self.flashes_spec.dynamic_range
        conditions = np.concatenate(
            np.array(
                (
                    np.ones(t_steps_pre) * sum(dynamic_range) / 2,
                    np.ones(t_steps_stim) * intensity,
                )
            )[self.flashes_spec.alternations]
        )

        # Label.
        if not "+" in node_type and xlabel:
            tau = self.nodes.time_const_trained[:][self.ctome.central_nodes_index[:]][
                self.ctome.unique_node_types[:].astype(str) == node_type
            ].item()
            xlabel += f" (τ = {tau * 1_000:.1f} ms)" if xlabel else ""

        title = node_type if title is None else title

        return plots.traces(
            response,
            x=np.arange(response.shape[0]) * self.flashes_spec.dt,
            contour=conditions,
            **kwargs,
        )

    def flash_response_grid(
        self,
        baseline,
        intensity,
        radius,
        node_types=None,
        startswith="",
        scale=3,
        figsize=None,
        aspect_ratio=8 / 12,
        wspace=0.5,
        hspace=0.5,
        **kwargs,
    ):
        node_types = node_types or self.node_types_sorted
        # Create an axis grid.
        fig, ax, _ = plt_utils.get_axis_grid(
            node_types, scale=scale, figsize=figsize, aspect_ratio=aspect_ratio
        )
        for i, nt in enumerate(node_types):
            self.flash_response(
                nt,
                baseline,
                intensity,
                radius,
                fig=fig,
                ax=ax[i],
                ylabel="",
                xlabel="",
                **kwargs,
            )

        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        return fig, ax

    def flash_strother_2014(
        self, baseline, intensity, radius, figsize=[10, 20], **kwargs
    ):
        neuron_types = ["L1", "L2+L4", "L3", "L4"]
        return self.flash_response_grid(
            baseline, intensity, radius, node_types=neuron_types, **kwargs
        )

    def fri(self, radius, node_types=None, figsize=[5, 1], fontsize=5):
        node_types = node_types or self.node_types_sorted

        # argmax_fri = self.flashes_dataset.argmax_fri()[1]

        _, fris_dict = self.flashes_dataset.flash_response_index(radius)
        fris = np.array([fris_dict[nt] for nt in node_types])

        return plots.bars(
            node_types,
            fris,
            legend=False,
            figsize=figsize,
            fontsize=fontsize,
            grid=False,
        )

    def flashes_response_argmax_fri(self, node_type, xlabel=None, **kwargs):
        fri, argmax, _ = self.flashes_dataset.fri(node_type)
        args = (
            "FRI(baseline={}, intensity={}, radius={})={:2G}".format(*argmax, fri)
            if xlabel is None
            else xlabel
        )
        mask = self.flashes_dataset.mask(*argmax)
        r = self.flashresponse[node_type][mask].squeeze()
        input = self.flashes_dataset.flashes_wrap.flashes[mask, :, 360].squeeze()
        fig, ax = plots.gratings_traces(input, r, self.flashes_spec.dt, **kwargs)
        ax.set_xlabel(xlabel)

        return fig, ax

    def flashes_response_argmax_fri_grid(
        self, node_types=None, aspect_ratio=1, fontsize=10, figsize=[10, 10]
    ):
        node_types = node_types or self.node_types_sorted

        fig, axes, (gw, gh) = plt_utils.get_axis_grid(
            node_types, aspect_ratio=aspect_ratio, figsize=figsize
        )

        for i, node_type in enumerate(node_types):
            self.flashes_response_argmax_fri(
                node_type,
                xlabel=node_type,
                fig=fig,
                ax=axes[i],
                fontsize=fontsize,
            )
        return fig, axes

    # ---- GRATING RESPONSES

    # def grating_response(self, node_type, width, speed, intensity,  **kwargs):
    #     if not self._initialized["gratings"]:
    #         self.init_gratings()
    #     # Get activity for given parameters and node type.
    #     mask = self.gratings_dataset.mask(None, width, speed, intensity)
    #     activity = self.gratingsresponse[node_type].max(axis=1)
    #     ymin, ymax = activity.min(), activity.max()
    #     activity = np.maximum(activity[mask], 0)
    #     theta = np.array(self.gratings_spec.angles)
    #     return plots.polar(theta, activity, **kwargs)

    def dsi(self, intensity=[0, 1], figsize=[15, 7]):
        fig, axes, _ = plt_utils.get_axis_grid(
            gridwidth=1, gridheight=2, figsize=figsize, hspace=0.5
        )
        nodes, dsis, theta_pref = self.movingbar.dsi(intensity=intensity)
        plots.bars(nodes, dsis, legend=False, fig=fig, ax=axes[0])
        plots.bars(nodes, theta_pref, legend=False, fig=fig, ax=axes[1])
        axes[1].hlines(0, 0, len(nodes))
        axes[1].hlines(90, 0, len(nodes))
        axes[1].hlines(180, 0, len(nodes))
        axes[1].hlines(270, 0, len(nodes))
        axes[1].hlines(360, 0, len(nodes))
        axes[0].set_ylabel("DSI")
        axes[1].set_ylabel("PD")
        return fig, axes

    def motion_tuning(
        self,
        node_type,
        intensity=[0, 1],
        color="g",
        xlabel=None,
        stroke_kwargs={},
        groundtruth=False,
        **kwargs,
    ):

        if groundtruth and node_type in utils.groundtruth_utils.tuning_curves:
            r = np.array(utils.groundtruth_utils.tuning_curves[node_type])
            theta = np.arange(0, 360, 360 / len(r))
        else:
            dsi, theta_pref, _, (theta, r), _ = self.movingbar.dsi(
                node_type, intensity=intensity, round_angle=True, speed=13
            )
            if isinstance(dsi, Iterable):
                dsi = np.mean(dsi)
            xlabel = f"DSI: {dsi:.2G}, PD: {theta_pref}" if xlabel is None else xlabel
        fig, ax = plots.polar(
            theta,
            r / r.max(),
            xlabel=xlabel,
            color=color,
            fontweight="normal",
            stroke_kwargs=stroke_kwargs,
            **kwargs,
        )
        return fig, ax

    def motion_tuning_all(
        self,
        node_type,
        fontsize=5,
        nonlinearity=True,
        figsize=[1.5, 2.5],
        title_off=False,
        fig=None,
        axes=None,
        peak_responses=None,  # (angles, widths, intensities, speeds, node_types)
        angles=None,
        widths=None,
        intensities=None,
        speeds=None,
        xlabel_off=True,
        cbar=True,
        anglepad=-5,
        cmap=plt.cm.viridis_r,
        linewidth=1,
        cbar_offset=(1.1, 0),
        subwrap="movingbar",
        **kwargs,
    ):

        _type = ""
        if peak_responses is None:

            self.init_movingbar(subwrap=subwrap)

            response = np.nanmax(self.movingbar.response(node_type=node_type), axis=-1)
            if nonlinearity:
                response[np.isnan(response)] = 0
                response = np.maximum(response, 0)
            ymin = np.nanmin(response)
            ymax = np.nanmax(response)

            widths = widths or self.movingbar.widths
            intensities = intensities or self.movingbar.intensities
            angles = angles or self.movingbar.angles
            speeds = speeds or self.movingbar.speeds

            if "bar" in subwrap:
                _type = "bar"
            elif "edge" in subwrap:
                _type = "edge"

        else:
            ymin = np.nanmin(peak_responses)
            ymax = np.nanmax(peak_responses)

            def is_none(x):
                return x is None

            if any(map(is_none, [widths, intensities, angles, speeds])):
                raise ValueError("must define stim params with peak_responses")

        if fig is None or axes is None:
            fig, axes, _ = plt_utils.get_axis_grid(
                gridwidth=len(widths),
                gridheight=len(intensities),
                projection="polar",
                figsize=[len(widths) * figsize[0], figsize[1]],
                as_matrix=True,
            )

        for i, intensity in enumerate(intensities[::-1]):
            for j, width in enumerate(widths):

                if peak_responses is None:
                    response = self.movingbar.response(
                        node_type=node_type, width=width, intensity=intensity
                    ).reshape(len(angles), len(speeds), -1)
                    if nonlinearity:
                        response[np.isnan(response)] = 0
                        response = np.maximum(response, 0)
                    peak_response = np.nanmax(response, axis=-1)
                else:
                    peak_response = peak_responses[
                        :,
                        j,
                        len(intensities) - 1 - i,
                        :,
                        self.node_indexer[node_type],
                    ]

                if cbar:
                    _cbar = False
                    if i == 0 and j + 1 == len(widths):
                        _cbar = True
                else:
                    _cbar = False

                ylabel = ""
                if intensity == 0 and j % len(widths) == 0:
                    ylabel = f"off-{_type}\nresponse"
                elif intensity == 1 and j % len(widths) == 0:
                    ylabel = f"on-{_type}\nresponse"

                xlabel = ""
                if not xlabel_off:
                    if i + 1 == len(intensities):
                        xlabel = f"width: {width} col."

                plots.speed_polar(
                    angles,
                    peak_response.T,
                    speeds,
                    ymin=ymin,
                    ymax=ymax,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    ax=axes[i, j],
                    fig=fig,
                    cbar=_cbar,
                    fontsize=fontsize,
                    cmap=cmap,
                    anglepad=anglepad,
                    linewidth=linewidth,
                    cbar_offset=cbar_offset,
                    **kwargs,
                )
            if not title_off:
                fig.suptitle(node_type, fontsize=fontsize)
        return fig, axes, peak_response, speeds

    def motion_tuning_onoff(
        self,
        node_type,
        intensity=0,
        fontsize=5,
        nonlinearity=True,
        figsize=[1, 1],
        cbar_offset=(1.1, 0),
        linewidth=1,
        anglepad=-5,
        cmap=plt.cm.viridis_r,
        cbar=True,
        subwrap="movingedge_chkpt_best_v4",
    ):
        self.init_movingbar(subwrap=subwrap)
        widths = self.movingbar.widths
        angles = self.movingbar.angles
        speeds = self.movingbar.speeds
        fig, axes, _ = plt_utils.get_axis_grid(
            gridwidth=len(widths),
            gridheight=1,
            projection="polar",
            as_matrix=False,
            figsize=figsize,
            fontsize=fontsize,
        )
        response = np.nanmax(
            self.movingbar.response(node_type=node_type, intensity=intensity),
            axis=-1,
        )
        if nonlinearity:
            response[np.isnan(response)] = 0
            response = np.maximum(response, 0)
        ymin = np.nanmin(response)
        ymax = np.nanmax(response)

        for j, width in enumerate(widths):
            response = self.movingbar.response(
                node_type=node_type, width=width, intensity=intensity
            ).reshape(len(angles), len(speeds), -1)

            if nonlinearity:
                response[np.isnan(response)] = 0
                response = np.maximum(response, 0)
            peak_response = np.nanmax(response, axis=-1)

            _cbar = False
            if j + 1 == len(widths):
                _cbar = True & cbar

            ylabel = ""
            if intensity == 0 and j == 0:
                ylabel = "OFF"
            elif intensity == 1 and j == 0:
                ylabel = "ON"
            xlabel = f"width: {width} col."

            plots.speed_polar(
                angles,
                peak_response.T,
                speeds,
                ymin=ymin,
                ymax=ymax,
                xlabel=xlabel,
                ylabel=ylabel,
                ax=axes[j],
                fig=fig,
                cbar=_cbar,
                fontsize=fontsize,
                linewidth=linewidth,
                cbar_offset=cbar_offset,
                anglepad=anglepad,
                cmap=cmap,
            )

        fig.suptitle(node_type, fontsize=fontsize)
        return fig, axes

    def motion_tuning_width(
        self,
        node_type,
        width=4,
        intensity=1,
        cbar=True,
        fontsize=10,
        fig=None,
        ax=None,
        figsize=[3, 3],
        **kwargs,
    ):
        angles = self.movingbar.angles
        speeds = self.movingbar.speeds
        fig, ax = plt_utils.init_plot(
            figsize=figsize, projection="polar", fig=fig, ax=ax
        )
        response = np.nanmax(
            self.movingbar.response(
                node_type=node_type, width=width, intensity=intensity
            ),
            axis=-1,
        )
        ymin = np.nanmin(response)
        ymax = np.nanmax(response)

        response = self.movingbar.response(
            node_type=node_type, width=width, intensity=intensity
        ).reshape(len(angles), len(speeds), -1)
        peak_response = np.nanmax(response, axis=-1)

        ylabel = ""
        if intensity == 0:
            ylabel = "OFF"
        elif intensity == 1:
            ylabel = "ON"

        xlabel = f"width: {width} col."

        fig, ax, cbar = plots.speed_polar(
            angles,
            peak_response.T,
            speeds,
            ymin=ymin,
            ymax=ymax,
            xlabel=xlabel,
            ylabel=ylabel,
            ax=ax,
            fig=fig,
            cbar=cbar,
            fontsize=fontsize,
            **kwargs,
        )

        ax.set_title(node_type, fontsize=fontsize)
        return fig, ax, cbar

    def plot_strf(
        self,
        node_type,
        intensity,
        max_extent=4,
        hlines=True,
        vlines=True,
        time_axis=True,
        fontsize=6,
        fig=None,
        axes=None,
        figsize=[5, 1],
        wspace=-0.15,
        y_offset_time_axis=0,
        subwrap="impulse_v2",
    ):
        if self._initialized["dots"] != subwrap:
            self.init_dots(subwrap=subwrap)

        if intensity not in self.dots.intensities:
            raise ValueError(
                f"valid intensities for {subwrap} are {self.dots.intensities}"
            )
        rf = self.dots.receptive_field(node_type, intensity)
        rf -= rf[0][None]
        n_frames = rf.shape[0]
        time = np.arange(n_frames) * self.dots.dt
        t_steps = np.arange(0.0, 0.2, 0.01)[::2]

        u, v = utils.get_hex_coords(max_extent)
        x, y = utils.hex_to_pixel(u, v)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        elev = 0
        azim = 0

        #     x, y = hex_rows(1, 10)
        if fig is None or axes is None:
            fig, axes = plt_utils.divide_figure_to_grid(
                np.arange(10).reshape(1, 10),
                wspace=wspace,
                as_matrix=True,
                figsize=figsize,
            )

        crange = np.abs(rf).max()
        for i, t in enumerate(t_steps):
            mask = np.where(np.abs(time - t) <= 1e-15, True, False)
            _rf = rf[mask]
            plots.quick_hex_scatter(
                _rf,
                cmap=plt.cm.coolwarm,
                edgecolor=None,
                vmin=-crange,
                vmax=crange,
                midpoint=0,
                cbar=False,
                max_extent=max_extent,
                fig=fig,
                ax=axes[0, i],
                fill=True,
                fontsize=fontsize,
            )

            if hlines:
                axes[0, i].hlines(elev, xmin, xmax, color="#006400", linewidth=0.25)
            if vlines:
                axes[0, i].vlines(azim, ymin, ymax, color="#006400", linewidth=0.25)

        if time_axis:

            left = fig.transFigure.inverted().transform(
                axes[0, 0].transData.transform((0, 0))
            )[0]
            right = fig.transFigure.inverted().transform(
                axes[0, -1].transData.transform((0, 0))
            )[0]

            lefts, bottoms, rights, tops = np.array(
                [ax.get_position().extents for ax in axes.flatten()]
            ).T
            #         time_axis = fig.add_axes((lefts.min(), bottoms.min(), rights.max() - lefts.min(), 0.01))
            time_axis = fig.add_axes(
                (
                    left,
                    bottoms.min() + y_offset_time_axis * bottoms.min(),
                    right - left,
                    0.01,
                )
            )
            dvs.plots.rm_spines(
                time_axis,
                ("left", "top", "right"),
                rm_yticks=True,
                rm_xticks=False,
            )

            data_centers_in_points = np.array(
                [ax.transData.transform((0, 0)) for ax in axes.flatten()]
            )
            time_axis.tick_params(axis="both", labelsize=fontsize)
            ticks = time_axis.transData.inverted().transform(data_centers_in_points)[
                :, 0
            ]
            time_axis.set_xticks(ticks)
            time_axis.set_xticklabels(np.arange(0, 200, 20))
            time_axis.set_xlabel("time (ms)", fontsize=fontsize, labelpad=2)
            plt_utils.set_spine_tick_params(
                time_axis,
                spinewidth=0.25,
                tickwidth=0.25,
                ticklength=3,
                ticklabelpad=2,
                spines=("top", "right", "bottom", "left"),
                labelsize=fontsize,
            )
        #         time_axis.set_xlim(*dvs.plots.get_lims(ticks, 0))

        #     axes[0].set_ylabel(node_type, fontsize=FONTSIZE, labelpad=-8)
        #     plt.subplots_adjust(wspace=-0.1)
        return fig, axes

    def motion_tuning_quartett_speed(
        self,
        node_types=None,
        width=4,
        intensity=1,
        aspect_ratio=4,
        fontsize=10,
        figsize=[8, 0.2],
    ):
        """Good defaults for a grid of 4 node types."""
        node_types = node_types or self.node_types_sorted

        fig, axes, _ = plt_utils.get_axis_grid(
            node_types,
            projection="polar",
            aspect_ratio=aspect_ratio,
            figsize=figsize,
            wspace=0.5,
        )

        # cmap = cm.get_cmap("tab10") if len(node_types) <= 10 else plt_utils.cm_uniform_2d
        cbar = False
        for i, node_type in enumerate(node_types):
            if i + 1 == len(node_types):
                cbar = True
            fig, ax = self.motion_tuning_width(
                node_type,
                width=width,
                intensity=intensity,
                fig=fig,
                ax=axes[i],
                cbar=cbar,
                cbar_offset=(1.3, 0),
            )

            ax.set_ylabel("")
            ax.set_title(node_type, fontsize=fontsize)
            ax.set_xlabel("")
        return fig, axes

    def motion_tuning_quartett(
        self,
        node_types=None,
        aspect_ratio=4,
        fontsize=10,
        colors=None,
        fig=None,
        axes=None,
        wspace=0.1,
        figsize=[8, 0.5],
        **kwargs,
    ):
        """Good defaults for a grid of 4 node types."""
        node_types = node_types or self.node_types_sorted

        if fig is None or axes is None:
            fig, axes, _ = plt_utils.get_axis_grid(
                node_types,
                projection="polar",
                aspect_ratio=aspect_ratio,
                figsize=figsize,
                wspace=wspace,
            )

        cmap = (
            cm.get_cmap("tab10") if len(node_types) <= 10 else plt_utils.cm_uniform_2d
        )
        for i, node_type in enumerate(node_types):
            if colors is None:
                color = cmap(i)
            else:
                color = colors[i]
            self.motion_tuning(
                node_type,
                xlabel=node_type,
                color=color,
                fig=fig,
                ax=axes[i],
                fontsize=fontsize,
                **kwargs,
            )
        return fig, axes

    def motion_tuning_grid(
        self,
        node_types=None,
        scale=1,
        figsize=[7.5, 6],
        aspect_ratio=1.5,
        fontsize=8,
        **kwargs,
    ):
        """Good defaults for a grid of all node types."""
        node_types = node_types or self.node_types_sorted
        # Create an axis grid.
        fig, ax, _ = plt_utils.get_axis_grid(
            node_types,
            projection="polar",
            scale=scale,
            figsize=figsize,
            aspect_ratio=aspect_ratio,
        )

        peak_responses, _ = self.movingbar.peak_response_angular(None)
        peak_responses = np.abs(peak_responses.sum(axis=(1, 2, 3))) / (
            np.abs(peak_responses).sum(axis=(0, 1, 2, 3)) + 1e-15
        )
        ymax = peak_responses.max()

        for i, nt in enumerate(node_types):
            self.motion_tuning(
                nt, fig=fig, ax=ax[i], xlabel=nt, fontsize=fontsize, **kwargs
            )
            ax[i].set_ylim(0, ymax)
        return fig, ax

    def motion_tuning_grid_on_off(
        self,
        node_types=None,
        scale=1,
        figsize=[7.5, 6],
        aspect_ratio=1.5,
        fontsize=5,
        linewidth=1,
        **kwargs,
    ):
        """Good defaults for a grid of all node types."""

        ON = utils.color_utils.ON
        OFF = utils.color_utils.OFF

        node_types = node_types or self.node_types_sorted
        # Create an axis grid.
        fig, ax, _ = plt_utils.get_axis_grid(
            node_types,
            projection="polar",
            scale=scale,
            figsize=figsize,
            aspect_ratio=aspect_ratio,
        )

        peak_responses, _ = self.movingbar.peak_response_angular(None)
        peak_responses = np.abs(peak_responses.sum(axis=(1, 2, 3))) / (
            np.abs(peak_responses).sum(axis=(0, 1, 2, 3)) + 1e-15
        )
        ymax = peak_responses.max()

        for i, nt in enumerate(node_types):
            self.motion_tuning(
                nt,
                fig=fig,
                ax=ax[i],
                xlabel=nt,
                fontsize=fontsize,
                color=OFF,
                intensity=0,
                linewidth=linewidth,
                **kwargs,
            )
            self.motion_tuning(
                nt,
                fig=fig,
                ax=ax[i],
                xlabel=nt,
                fontsize=fontsize,
                color=ON,
                intensity=1,
                linewidth=linewidth,
                **kwargs,
            )
            ax[i].set_ylim(0, ymax)

        for _ax in ax:
            [i.set_linewidth(0.5) for i in _ax.spines.values()]
            _ax.grid(True, linewidth=0.5)
        return fig, ax

    def paper_direction_tuning(
        self,
        node_types="T4",
        subwrap="movingedge_chkpt_best_v4",
        bottom_contrast="off",
        contrasts=[],
        fig=None,
        axes=None,
        on_color=None,
        off_color=None,
        groundtruth=False,
    ):

        self.init_movingbar(subwrap=subwrap)

        ON = on_color or utils.color_utils.ON
        OFF = off_color or utils.color_utils.OFF

        if isinstance(node_types, list):
            pass
        elif node_types == "T4":
            node_types = ["T4a", "T4b", "T4c", "T4d"]
            bottom_contrast = None
            contrasts = contrasts or [0, 1]
        elif node_types == "T5":
            node_types = ["T5a", "T5b", "T5c", "T5d"]
            bottom_contrast = None
            contrasts = contrasts or [1, 0]
        elif node_types == "TmY":
            node_types = ["TmY3", "TmY4", "TmY13", "TmY18"]
            contrasts = contrasts or [0, 1]
        else:
            raise ValueError(f"{node_types}")

        kwargs = dict(
            node_types=node_types,
            intensity=0,
            colors=(OFF,) * 4,
            linewidth=1,
            aspect_ratio=4,
            fontsize=5,
            figsize=[2.95, 0.83],
            anglepad=-7,
            xlabelpad=-1,
            stroke_kwargs=dict(),
            wspace=0.25,
            fig=fig,
            axes=axes,
        )

        if bottom_contrast is not None:
            logging.warning(
                "kw bottom_contrast is deprecated. use contrasts" " instead"
            )

        if bottom_contrast in ["off", 1]:
            fig, axes = self.motion_tuning_quartett(**kwargs)
            kwargs.update(intensity=1, colors=(ON,) * 4, fig=fig, axes=axes)
            fig, axes = self.motion_tuning_quartett(**kwargs)
        elif bottom_contrast in ["on", 0]:
            kwargs.update(intensity=1, colors=(ON,) * 4)
            fig, axes = self.motion_tuning_quartett(**kwargs)
            kwargs.update(intensity=0, colors=(OFF,) * 4, fig=fig, axes=axes)
            fig, axes = self.motion_tuning_quartett(**kwargs)

        def _update_contrast_kwargs(contrast, kwargs):
            if contrast in ["off", 0]:
                kwargs.update(intensity=0, colors=(OFF,) * 4)
            elif contrast in ["on", 1]:
                kwargs.update(intensity=1, colors=(ON,) * 4)
            return kwargs

        for contrast in contrasts:
            kwargs = _update_contrast_kwargs(contrast, kwargs)
            fig, axes = self.motion_tuning_quartett(**kwargs)
            kwargs.update(fig=fig, axes=axes)

        if groundtruth:
            kwargs.update(colors=("k",) * 4)
            kwargs.update(groundtruth=True)
            kwargs.update(zorder=10)
            fig, axes = self.motion_tuning_quartett(**kwargs)

        for ax in axes:
            ax.xaxis.label.set_fontsize(8)
            #     ax.xaxis.label.set_fontweight("bold")
            [i.set_linewidth(0.5) for i in ax.spines.values()]
            ax.grid(True, linewidth=0.5)
        return fig, axes

    def paper_t5_tuning(self, subwrap="movingedge_chkpt_best_v4"):
        self.init_movingbar(subwrap=subwrap)
        ON = utils.color_utils.ON
        OFF = utils.color_utils.OFF
        fig, axes = self.motion_tuning_quartett(
            ["T5a", "T5b", "T5c", "T5d"],
            intensity=1,
            colors=(ON,) * 4,
            linewidth=1,
            aspect_ratio=4,
            fontsize=5,
            figsize=[2.95, 0.83],
            anglepad=-7,
            xlabelpad=-1,
            stroke_kwargs=dict(linewidth=1.2, foreground="0.5"),
            wspace=0.25,
        )

        fig, axes = self.motion_tuning_quartett(
            ["T5a", "T5b", "T5c", "T5d"],
            intensity=0,
            colors=(OFF,) * 4,
            linewidth=1,
            aspect_ratio=4,
            fontsize=5,
            figsize=[2.95, 0.83],
            anglepad=-7,
            fig=fig,
            axes=axes,
            xlabelpad=-1,
            stroke_kwargs=dict(linewidth=1.2, foreground="0.5"),
            wspace=0.25,
        )

        for ax in axes:
            ax.xaxis.label.set_fontsize(8)
            #     ax.set_xlabel("")
            [i.set_linewidth(0.5) for i in ax.spines.values()]
            ax.grid(True, linewidth=0.5)
        return fig, axes

    def paper_t4c_traces(self, subwrap="movingedge_chkpt_best_v4"):
        self.init_movingbar(subwrap=subwrap)
        fig, ax, (time, op_resp) = self.movingbar_response(
            "T4c",
            angle=90,
            width=80,
            speed=19,
            intensity=1,
            figsize=[2, 1],
            contour_mode="bottom",
            fancy=True,
            scale_pos="",
            linewidth=1,
            null_line=True,
            fontsize=6,
            color=[
                hex2color(utils.color_utils.PD),
                hex2color(utils.color_utils.ND),
            ],
        )
        # ax.plot(time, op_resp, zorder=0, linewidth=2)
        ax.set_title("")
        return fig, ax

    def paper_t5c_traces(self, subwrap="movingedge_chkpt_best_v4"):
        self.init_movingbar(subwrap=subwrap)
        fig, ax, (time, op_resp) = self.movingbar_response(
            "T5c",
            angle=90,
            width=80,
            speed=19,
            intensity=0,
            figsize=[2, 1],
            contour_mode="bottom",
            fancy=True,
            scale_pos="",
            linewidth=1,
            null_line=True,
            fontsize=6,
            color=[
                hex2color(utils.color_utils.PD),
                hex2color(utils.color_utils.ND),
            ],
        )
        # ax.plot(time, op_resp, zorder=0, linewidth=2)
        ax.set_title("")
        return fig, ax

    def movingbar_response(
        self,
        node_type,
        stim_onset=1,
        null_direction=True,
        angle=None,
        width=None,
        speed=None,
        intensity=None,
        title=None,
        fontsize=10,
        full_title=False,
        ax=None,
        fig=None,
        figsize=[5, 3],
        xlim=(-0.5, 1),
        pre_stim=True,
        post_stim=True,
        zero_at="center",
        color=None,
        **kwargs,
    ):
        """
        Trace on top of gratings contour for argmax dsi.
        """
        dsi, _, argmax, _, _ = self.movingbar.dsi(
            node_type,
            round_angle=True,
            width=width,
            speed=speed,
            intensity=intensity,
        )

        if angle is None:
            angle = argmax[0]

        if intensity is None:
            intensity = argmax[2]

        kw_stim = dict(
            angle=angle,
            width=width or argmax[1],
            speed=speed or argmax[3],
            intensity=intensity,
        )

        dt = self.movingbar.dt
        stim = self.movingbar.stimulus(
            **kw_stim,
            pre_stim=pre_stim,
            post_stim=post_stim,
        )
        resp = self.movingbar.response(
            node_type=node_type,
            pre_stim=pre_stim,
            post_stim=post_stim,
            **kw_stim,
        )
        opposite_resp = self.movingbar.response(
            node_type=node_type,
            pre_stim=pre_stim,
            post_stim=post_stim,
            angle=(angle + 180) % 360,
            width=kw_stim["width"],
            intensity=kw_stim["intensity"],
            speed=kw_stim["speed"],
        )

        _nans = np.isnan(stim)
        stim = stim[~_nans]
        resp = resp[
            ~_nans
        ]  # TODO: the response can be non-nan for longer -> nan takes some timesteps to reach the neuron.
        opposite_resp = opposite_resp[~_nans]
        opposite_resp = opposite_resp - np.mean(opposite_resp)

        time_to_center = (
            np.abs(self.movingbar.spec.offsets[0])
            * np.radians(2.25)
            / (kw_stim["speed"] * np.radians(5.8))
        )
        if zero_at == "center":
            time = np.linspace(
                -(stim_onset + time_to_center),
                len(resp) * dt - (stim_onset + time_to_center),
                len(resp),
            )
        elif zero_at == "onset":
            time = np.linspace(-stim_onset, len(resp) * dt - stim_onset, len(resp))

        if xlim:
            mask = (time >= xlim[0]) & (time <= xlim[1])
            time = time[mask]
            resp = resp[mask]
            stim = stim[mask]
            opposite_resp = opposite_resp[mask]

        if null_direction:
            _responses = [resp, opposite_resp]

            if color is None:
                color = [
                    hex2color(utils.color_utils.PD),
                    hex2color(utils.color_utils.ND),
                ]
        else:
            _responses = resp
            if color is None:
                color = [hex2color(utils.color_utils.PD)]
        fig, ax, _, _ = plots.traces(
            _responses,
            x=time,
            contour=stim,
            fontsize=fontsize,
            fig=fig,
            ax=ax,
            figsize=figsize,
            color=color,
            **kwargs,
        )

        if title and title != "none":
            ax.set_title(title, fontsize=fontsize)
        elif title is None:
            title = "{}, {}={:.0F}"
            if full_title:
                title = "{}, {}={:.0F}, width={:.0F}°, speed={:.0F}°, intensity={}"
            # if _pd:
            #     title = title.format(node_type, "$\\theta_\mathrm{pref}$", angle, kw_stim["width"]*2.25, kw_stim["speed"]*5.8, kw_stim["intensity"])
            # elif _nd:
            #     title = title.format(node_type, "$\\theta_\mathrm{null}$", angle, kw_stim["width"]*2.25, kw_stim["speed"]*5.8, kw_stim["intensity"])
            else:
                title = title.format(
                    node_type,
                    "$\\theta$",
                    angle,
                    kw_stim["width"] * 2.25,
                    kw_stim["speed"] * 5.8,
                    kw_stim["intensity"],
                )
            ax.set_title(title, fontsize=fontsize)

        return fig, ax, (time, opposite_resp)

    def paper_movingbar_response(
        self,
        node_type,
        angle,
        width,
        speed,
        intensity,
        figsize=[2, 1],
        xlim=[-0.5, 1.0],
        contour_mode="bottom",
        fancy=True,
        fontsize=5,
    ):
        return self.movingbar_response(
            node_type,
            angle=angle,
            width=width,
            speed=speed,
            null_direction=True,
            intensity=intensity,
            figsize=figsize,
            xlim=xlim,
            contour_mode=contour_mode,
            fancy=fancy,
            scale_pos="",
            linewidth=1,
            null_line=True,
            title="none",
            fontsize=fontsize,
        )

    def movingbar_inhibitory_excitatory_inputs(
        self,
        target_type,
        time,
        stims,
        postsynaptic_inputs,
        responses,
        xlim=(-0.5, 1),
        figsize=[2, 1],
        contour_mode="bottom",
        **kwargs,
    ):

        (
            inhibitory_inputs,
            excitatory_inputs,
            target_responses,
            rfs,
        ) = dvs.analysis.inputs.inhibitory_excitatory_inputs_for_target(
            target_type, postsynaptic_inputs, responses, self.ctome
        )

        mask = (time >= xlim[0]) & (time <= xlim[1])
        _time = time[mask]
        _inhibitory_inputs = inhibitory_inputs[0, mask]
        _inhibitory_inputs -= _inhibitory_inputs[0]
        _excitatory_inputs = excitatory_inputs[0, mask]
        _excitatory_inputs -= _excitatory_inputs[0]

        stim = stims[0, mask, stims.shape[-1] // 2]

        inputs = np.stack([_inhibitory_inputs, _excitatory_inputs])

        fig, ax, _, _ = dvs.plots.traces(
            inputs,
            x=_time,
            contour=stim,
            figsize=figsize,
            contour_mode=contour_mode,
            fancy=True,
            scale_pos="",
            linewidth=1,
            null_line=True,
            color=[
                hex2color(utils.color_utils.INH),
                hex2color(utils.color_utils.EXC),
            ],
            **kwargs,
        )

        return fig, ax

    def movingbar_responses_all_directions(
        self,
        node_type,
        speed=13,
        width=4,
        angles=[0, 90, 180, 270],
        fig=None,
        axes=None,
        figsize=[3.6, 1],
        intensities=[0, 1],
        fontsize=5,
        cwheel=True,
        contour=True,
    ):

        cmap = dvs.plots.cm_uniform_2d
        sm, norm = dvs.plots.get_scalarmapper(
            cmap=cmap, vmin=-np.pi, vmax=np.pi, midpoint=0
        )
        colors = sm.to_rgba(np.radians(self.movingbar.angles))

        if fig is None or axes is None:
            fig, axes = dvs.plots.divide_figure_to_grid(
                np.array([np.arange(len(intensities))]),
                figsize=figsize,
                wspace=0.3,
                as_matrix=True,
                fontsize=fontsize,
            )
            axes = axes.flatten()

        for i, intensity in enumerate(intensities):
            ax = axes[i]
            for i, angle in enumerate(angles):
                time, stim, resp, opposite_resp = self.movingbar.traces(
                    node_type,
                    speed=speed,
                    angle=angle,
                    width=width,
                    intensity=intensity,
                    pre_stim=False,
                    post_stim=False,
                )
                ax.plot(time, resp, c=sm.to_rgba(np.radians(180 - angle)))
            ax.set_xlabel("time (s)", fontsize=5)

        axes[0].set_ylabel("voltage (a.u.)", fontsize=5)
        if cwheel:
            dvs.plots.add_colorwheel_2d(
                fig,
                [axes[-1]],
                norm=norm,
                sm=sm,
                radius=0.5,
                cmap=cmap,
                pos="northeast",
                x_offset=0.2,
                y_offset=-0.4,
                fontsize=5,
                labelpad=-6.5,
                mode="1d",
                ticks=[0, 90],
            )

        ylims = plt_utils.get_lims([ax.get_ylim() for ax in axes.flatten()], 0.05)

        if contour:
            for i, intensity in enumerate(intensities):
                ax = axes[i]
                time, stim, _, _ = self.movingbar.traces(
                    node_type,
                    speed=speed,
                    angle=angle,
                    width=width,
                    intensity=intensity,
                    pre_stim=False,
                    post_stim=False,
                )
                plt_utils.plot_stim_contour(time, stim, ax)

        for ax in axes:
            ax.set_ylim(*ylims)

        fig.suptitle(
            f"{node_type} moving edge responses (speed={speed * 5.8:.2f}°/s)",
            fontsize=5,
        )
        return fig, axes

    def movingbar_inputs(
        self,
        node_type,
        angle,
        width,
        speed,
        intensity,
        sorted_input_types=None,
        colors=None,
        n_inputs=None,
        stim_onset=1,
        fancy=False,
        subwrap="movingedge_chkpt_best_v4",
        title=None,
        fontsize=10,
        fig=None,
        ax=None,
        figsize=[5, 3],
        contour_mode="bottom",
        time_scale="lower left",
        cmap=cm.get_cmap("coolwarm"),
        xlim=(-0.5, 1),
        pre_stim=True,
        post_stim=True,
        zero_at="center",
        categorical_color_mode=False,
        ax_title="{node_type} - moving angle {angle}, speed {speed}, intensity {intensity}",
        legend=True,
        **kwargs,
    ):
        def get_ylims(inputs):
            ylims = np.array([v for v in inputs.values()])
            return plt_utils.get_lims(ylims, 0.1)

        if not self._initialized["network"]:
            self.init_network()
        self.init_movingbar(subwrap)

        indices = [
            self.movingbar._key(
                angle=angle, width=width, intensity=intensity, speed=speed
            )
        ]
        stim_resp = list(
            self.network.stimulus_response(
                self.movingbar, 1 / 100, t_pre=0, indices=indices
            )
        )
        stims = np.concatenate([r[1] for r in stim_resp], axis=0)
        activities = np.concatenate([r[0] for r in stim_resp], axis=0)
        result = dvs.analysis.input_currents(node_type, stims, activities, self.ctome)
        inputs = result["source_current"]
        inputs = valmap(lambda v: v.sum(-1), inputs)

        stim = self.movingbar.stimulus(
            angle=angle, width=width, speed=speed, intensity=intensity
        )
        _nans = np.isnan(stim)

        dt = self.movingbar.dt

        time_to_center = (
            np.abs(self.movingbar.spec.offsets[0])
            * np.radians(2.25)
            / (speed * np.radians(5.8))
        )
        if zero_at == "center":
            time = np.linspace(
                -(stim_onset + time_to_center),
                len(stim) * dt - (stim_onset + time_to_center),
                len(stim),
            )
        elif zero_at == "onset":
            time = np.linspace(-stim_onset, len(stim) * dt - stim_onset, len(stim))

        tmask = np.where(time >= xlim[0], True, False) & np.where(
            time <= xlim[1], True, False
        )
        _inp = valmap(lambda v: v[:, tmask], inputs)
        if sorted_input_types is None:
            extr = dict(
                sorted(
                    valmap(
                        lambda v: v[:, np.nanargmax(np.abs(v), axis=1)].item(),
                        _inp,
                    ).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
        else:
            extr = {cell_type: _inp[cell_type] for cell_type in sorted_input_types}

        if colors is None and categorical_color_mode:
            source_types = list(inputs.keys())
            colors = plt_utils.get_colors(len(source_types))
            colors = {source_types[i]: to_hex(c) for i, c in enumerate(colors)}
            source_types = list(extr.keys())
        elif colors is None:
            source_types = list(extr.keys())
            sm, norm = plt_utils.get_scalarmapper(
                vmin=min(np.min(list(extr.values())), -0.1),
                vmax=max(np.max(list(extr.values())), 0.1),
                midpoint=0,
                cmap=cmap,
            )
            colors = {
                source_types[i]: c
                for i, c in enumerate(sm.to_rgba(list(extr.values())))
            }
        else:
            source_types = list(extr.keys())

        if not fancy:
            time_scale = ""
            contour_mode = "full"

        fig, ax = plt_utils.init_plot(figsize=figsize, fontsize=fontsize)
        ylim = get_ylims(inputs)
        time = time[~_nans]
        contour = None
        for j, (source_type, activation) in enumerate(inputs.items()):
            if source_type not in source_types:
                continue
            activation = activation.squeeze()[~_nans]
            if j + 1 == len(inputs):
                contour = stim[~_nans]
            plots.traces(
                activation,
                x=time,
                contour=contour,
                color=colors[source_type],
                fancy=fancy,
                ylim=ylim,
                fig=fig,
                ax=ax,
                linewidth=0.5,
                contour_mode=contour_mode,
                scale_pos=time_scale,
                null_line=True,
                fontsize=fontsize,
                **kwargs,
            )

        ax.set_xlim(*xlim)
        ax.set_xlabel("time (s)", fontsize=fontsize)
        ax.set_ylabel("current (a.u.)", fontsize=fontsize)
        ax.set_title(
            ax_title.format(
                node_type=node_type,
                angle=angle,
                speed=speed,
                intensity=intensity,
            ),
            fontsize=fontsize,
        )
        if legend:
            plt_utils.add_legend(
                ax,
                source_types,
                [colors[st] for st in source_types],
                fontsize=fontsize,
            )

        return fig, ax, (source_types, [colors[st] for st in source_types])

    def movingbar_outputs(
        self,
        node_type,
        angle,
        width,
        speed,
        intensity,
        n_outputs=5,
        stim_onset=1,
        fancy=True,
        subwrap="movingedge_chkpt_best_v4",
        title=None,
        fontsize=10,
        fig=None,
        ax=None,
        figsize=[5, 3],
        xlim=(-0.5, 1),
        pre_stim=True,
        post_stim=True,
        zero_at="center",
        **kwargs,
    ):
        return NotImplemented

        # def get_ylims(outputs):
        #     ylims = np.array([v for v in outputs.values()])
        #     return dvs.plots.plt_utils.get_lims(ylims, 0.1)

        # if not self._initialized["network"]:
        #     self.init_network()
        # self.init_movingbar(subwrap)

        # indices = [self.movingbar._key(angle=angle, width=width, intensity=intensity, speed=speed)]
        # stim_resp = list(self.network.stimulus_response(self.movingbar, 1/200, t_pre=0, indices=indices))
        # stims = np.concatenate([r[1] for r in stim_resp], axis=0)
        # activities = np.concatenate([r[0] for r in stim_resp], axis=0)
        # result = input_currents(node_type, stims, activities, self.ctome)
        # outputs = result['source_current']
        # outputs = valmap(lambda v: v.sum(-1), outputs)

        # # get std deviations of outputs
        # stds = dict(sorted(valmap(lambda v: np.nanstd(v), outputs).items(), key=lambda item: item[1], reverse=True))
        # source_types = list(stds.keys())[:n_outputs]

        # stim = self.movingbar.stimulus(angle=angle, width=width, speed=speed, intensity=intensity)
        # _nans = np.isnan(stim)

        # dt = self.movingbar.dt

        # time_to_center = np.abs(self.movingbar.spec.offsets[0]) * np.radians(2.25) / (speed * np.radians(5.8))
        # if zero_at == "center":
        #     time = np.linspace(-(stim_onset + time_to_center), len(stim) * dt - (stim_onset + time_to_center), len(stim))
        # elif zero_at == "onset":
        #     time = np.linspace(-stim_onset, len(stim) * dt - stim_onset, len(stim))

        # time = time[~_nans]

        # fig, ax = plt_utils.init_plot(figsize=figsize, fontsize=fontsize)

        # _contoured = False
        # for j, (source_type, activation) in enumerate(outputs.items()):
        #     if source_type not in source_types:
        #         continue
        #     activation = activation.squeeze()[~_nans]
        #     if not _contoured:
        #         contour = stim[~_nans]
        #         _contoured = True
        #     else:
        #         contour=None
        #     plots.traces(activation, x=time, contour=contour,
        #                 fig=fig, ax=ax, linewidth=2, fancy=fancy,
        #                 null_line=True, fontsize=fontsize, **kwargs)

        # ax.set_xlim(*xlim)
        # ax.set_ylim(*get_ylims(outputs))
        return fig, ax, (source_types, [l.get_color() for l in ax.lines])

    def movingbar_responses_grid(
        self,
        node_types=None,
        aspect_ratio=4,
        figsize=[8, 0.5],
        fontsize=10,
        null_direction=False,
        **kwargs,
    ):
        """
        Trace on top of gratings contour for argmax dsi.
        """
        node_types = node_types or self.node_types_sorted

        fig, axes, (gw, gh) = plt_utils.get_axis_grid(
            node_types, aspect_ratio=aspect_ratio, figsize=figsize
        )

        for i, node_type in enumerate(node_types):
            self.movingbar_response(
                node_type,
                fig=fig,
                ax=axes[i],
                fontsize=fontsize,
                title=None,
                null_direction=null_direction,
                **kwargs,
            )

        return fig, axes

    def oriented_bar_responses_all_directions(
        self,
        node_type,
        width=4,
        angles=[0, 90, 180, 270],
        fig=None,
        axes=None,
        figsize=[3.6, 1],
        intensities=[0, 1],
        fontsize=5,
        cwheel=True,
        contour=True,
        subwrap="oriented_edge",
    ):
        self.init_oriented_bar(subwrap=subwrap)

        cmap = dvs.plots.cm_uniform_2d
        sm, norm = dvs.plots.get_scalarmapper(
            cmap=cmap, vmin=-np.pi, vmax=np.pi, midpoint=0
        )
        colors = sm.to_rgba(np.radians(self.oriented_bar.angles))

        if fig is None or axes is None:
            fig, axes = dvs.plots.divide_figure_to_grid(
                np.array([np.arange(len(intensities))]),
                figsize=figsize,
                wspace=0.3,
                as_matrix=True,
                fontsize=fontsize,
            )
            axes = axes.flatten()

        for i, intensity in enumerate(intensities):
            ax = axes[i]
            for i, angle in enumerate(angles):
                time, stim, resp = self.oriented_bar.traces(
                    node_type,
                    angle=angle,
                    width=width,
                    intensity=intensity,
                    pre_stim=False,
                    post_stim=False,
                )
                ax.plot(time, resp, c=sm.to_rgba(np.radians(180 - angle)))
            ax.set_xlabel("time (s)", fontsize=5)

        axes[0].set_ylabel("voltage (a.u.)", fontsize=5)
        if cwheel:
            dvs.plots.add_colorwheel_2d(
                fig,
                [axes[-1]],
                norm=norm,
                sm=sm,
                radius=0.5,
                cmap=cmap,
                pos="northeast",
                x_offset=0.2,
                y_offset=-0.4,
                fontsize=5,
                labelpad=-6.5,
                mode="1d",
                ticks=[0, 90],
            )

        ylims = plt_utils.get_lims([ax.get_ylim() for ax in axes.flatten()], 0.05)

        if contour:
            for i, intensity in enumerate(intensities):
                ax = axes[i]
                time, stim, _ = self.oriented_bar.traces(
                    node_type,
                    angle=angle,
                    width=width,
                    intensity=intensity,
                    pre_stim=False,
                    post_stim=False,
                )
                plt_utils.plot_stim_contour(time, stim, ax)

        for ax in axes:
            ax.set_ylim(*ylims)

        fig.suptitle(
            f"{node_type} orientation stimulus-responses",
            fontsize=5,
        )
        return fig, axes

    def gratings_trace(
        self,
        node_type,
        title=None,
        fontsize=10,
        null_direction=False,
        include_pre_stim_activity=True,
        **kwargs,
    ):
        """
        Trace on top of gratings contour for argmax dsi.
        """
        dsi, _, argmax, _, _, _ = self.gratings_dataset.dsi(node_type, round_angle=True)
        args = [
            kwargs.pop("angle", None),
            kwargs.pop("nbars", None),
            kwargs.pop("width", None),
            kwargs.pop("speed", None),
            kwargs.pop("intensity", None),
        ]
        args = [arg if arg is not None else argmax[i] for i, arg in enumerate(args)]
        pd = args[0]
        nd = int((pd + 180) % 360)
        if null_direction:
            args[0] = nd
        dt = self.gratings_spec.dt
        prestim = (
            0
            if include_pre_stim_activity
            else int(self.gratings_spec.t_pre / self.gratings_spec.dt)
        )
        input = self.gratings_dataset.stimulus(
            *args, include_pre_stim_activity=include_pre_stim_activity
        )
        activity = self.gratings_dataset.response(
            *args,
            node_type=node_type,
            include_pre_stim_activity=include_pre_stim_activity,
        )
        fig, ax = plots.gratings_traces(
            input.squeeze(),
            activity.squeeze(),
            dt=dt,
            fontsize=fontsize,
            **kwargs,
        )

        if title is None:
            title = node_type
            # title += "\nDSI(nbars={}, width={},\n speed={}, intensity={})={:1F}\n".format(*argmax[1:], dsi)
            title += (
                ", $\\theta_\mathrm{pref}$"
                if not null_direction
                else ", $\\theta_\mathrm{null}$"
            )
            title += f"={args[0]:.0F}"
        title = title or node_type
        ax.set_title(title, fontsize=fontsize)

        stim_onset = self.gratings_spec.t_pre if include_pre_stim_activity else 0
        ax.vlines(stim_onset, -2000, 2000, colors="r", linewidth=2)

        return fig, ax

    def gratings_traces_grid(
        self,
        node_types=None,
        aspect_ratio=4,
        figsize=[8, 0.5],
        fontsize=10,
        null_direction=False,
        **kwargs,
    ):
        """
        Trace on top of gratings contour for argmax dsi.
        """
        node_types = node_types or self.node_types_sorted

        fig, axes, (gw, gh) = plt_utils.get_axis_grid(
            node_types, aspect_ratio=aspect_ratio, figsize=figsize
        )

        for i, node_type in enumerate(node_types):
            self.gratings_trace(
                node_type,
                fig=fig,
                ax=axes[i],
                fontsize=fontsize,
                title=None,
                null_direction=null_direction,
                **kwargs,
            )

        return fig, axes

    # ---- POTENTIAL

    def potential_central(
        self,
        batch_type,
        startswith="",
        node_types=None,
        offset=10,
        time_const=True,
        batch_sample=0,
        **kwargs,
    ):
        """Plots the membrane potential over time (frames).

        Args:
            node_types (list, str or None): if list, each entry must specify a
                valid node type. If str, must be either "input" or "output".
                If None, all nodes containing substr in their name are shown.
            startswith (str): if node_types is None, nodes starting with
                startwith are plotted.
            time_const (bool): optional time constant annotation.
            **kwargs: arbitrary keyword arguments.
        """

        # get intended node types and centers to index activity
        nodes, valid_node_types = self.filter_df_by_list(startswith, node_types)
        nodes = nodes[(nodes.u == 0) & (nodes.v == 0)]

        # get activity
        activity = self.tnn[batch_type].network_states.nodes.activity[:]
        activity = activity[batch_sample][..., nodes.index]

        return plots.potential_over_frames(
            nodes, activity, time_const=time_const, **kwargs
        )

    def potential_mean(
        self,
        batch_type,
        startswith="",
        node_types=None,
        offset=10,
        time_const=True,
        batch_sample=0,
        **kwargs,
    ):
        """Plots the membrane potential over time (frames).

        Args:
            node_types (list, str or None): if list, each entry must specify a
                valid node type. If str, must be either "input" or "output".
                If None, all nodes containing substr in their name are shown.
            startswith (str): if node_types is None, nodes starting with
                startwith are plotted.
            time_const (bool): optional time constant annotation.
            **kwargs: arbitrary keyword arguments.
        """
        # get intended node types to index activity
        nodes, valid_node_types = self.filter_df_by_list(startswith, node_types)

        # get activity
        activity = self.tnn[batch_type].network_states.nodes.activity[:]
        activity = activity[batch_sample][..., nodes.index]

        # spatially aggregate activity
        _activity = pd.DataFrame(dict(type=nodes.type.copy()))
        for frame, act in enumerate(activity):
            _activity[f"frame_{frame}"] = act
        _activity = _activity.groupby(by=["type"], sort=False).mean()
        activity = _activity[
            [column for column in _activity.columns if column.startswith("frame_")]
        ].values.T
        nodes = nodes.groupby(by=["type"], sort=False, as_index=False).mean()

        return plots.potential_over_frames(
            nodes, activity, time_const=time_const, **kwargs
        )

    # ---- ANIMATIONS

    def sintel_animation(self, **kwargs):
        anim = animations.SintelMultiTask(self.tnn, **kwargs)
        return anim

    def layer_animation(self, **kwargs):
        anim = animations.LayerActivityGrid(self.tnn, **kwargs)
        return anim

    def input_recording_animation(self, **kwargs):
        anim = animations.InputPlusRecordingsAndPrediction(self.tnn, **kwargs)
        return anim

    def _response(self, batch_type, _assert):
        # get probe wrap
        response = (
            batch_type
            if isinstance(batch_type, Datawrap)
            else Datawrap(self.tnn.path / f"{batch_type}")
        )
        assert response.path.exists() and isinstance(
            response, _assert
        ), f"{batch_type.capitalize()} is not {_assert}."
        return response

    # ---- Network Graphs
    def network_graph(self, color, **kwargs):
        """Plots the abstracted network graph."""
        nodes = self.node_types_sorted
        edges = pd.DataFrame(
            dict(
                source_type=self.edges.source_type[:].astype(str),
                target_type=self.edges.target_type[:].astype(str),
            )
        ).drop_duplicates()
        edges = list(
            map(
                lambda x: x.split(","),
                (edges.source_type + "," + edges.target_type),
            )
        )
        return plots.network_graph(
            nodes, edges, self.layout, node_color=color, **kwargs
        )

    def network_graph_dsi(self, intensity=[0, 1], **kwargs):
        """Plots the abstracted network graph."""
        edges = pd.DataFrame(
            dict(
                source_type=self.edges.source_type[:].astype(str),
                target_type=self.edges.target_type[:].astype(str),
            )
        ).drop_duplicates()
        edges = list(
            map(
                lambda x: x.split(","),
                (edges.source_type + "," + edges.target_type),
            )
        )
        nodes, dsis, theta = self.movingbar.dsi("all", intensity=intensity)
        return plots.network_graph(nodes, edges, self.layout, node_color=dsis, **kwargs)

    def network_graph_fri(
        self,
        cmap=cm.get_cmap("seismic"),
        mode="transient",
        absolute=False,
        groundtruth=False,
        **kwargs,
    ):
        """Plots the abstracted network graph."""
        nodes = self.node_types_sorted
        edges = pd.DataFrame(
            dict(
                source_type=self.edges.source_type[:].astype(str),
                target_type=self.edges.target_type[:].astype(str),
            )
        ).drop_duplicates()
        edges = list(
            map(
                lambda x: x.split(","),
                (edges.source_type + "," + edges.target_type),
            )
        )
        node_color = []
        for nt in nodes:
            if groundtruth:
                node_color.append(utils.polarity[nt])
            else:
                fri, _, _ = self.flashes_dataset.fri(nt, mode=mode)
                node_color.append(fri)
        node_color = np.array(node_color)
        if absolute:
            node_color[node_color < 0] = -1
            node_color[node_color > 0] = 1
        return plots.network_graph(
            nodes,
            edges,
            self.layout,
            node_color=node_color,
            node_cmap=cmap,
            **kwargs,
        )

    def super_threshold_ratio(self, subwrap="augmented_sintel"):
        """Returns percentage active on full validation set per neuron type."""
        activity = self.tnn[subwrap].network_states.nodes.activity_central[:]
        activity = (activity >= 0).sum(axis=(0, 1)) / np.prod(activity.shape[:-1])
        return activity

    def dead_ratio(self, cmap=cm.get_cmap("Purples_r"), **kwargs):
        dead_ratio = utils.dead_ratio(self.tnn)

        nodes = self.node_types_sorted
        edges = pd.DataFrame(
            dict(
                source_type=self.edges.source_type[:].astype(str),
                target_type=self.edges.target_type[:].astype(str),
            )
        ).drop_duplicates()
        edges = list(
            map(
                lambda x: x.split(","),
                (edges.source_type + "," + edges.target_type),
            )
        )
        node_color = []
        for nt in nodes:
            node_color.append(dead_ratio[nt])
        node_color = np.array(node_color)
        return plots.network_graph(
            nodes,
            edges,
            self.layout,
            node_color=node_color,
            node_cmap=cmap,
            **kwargs,
        )

    # ---- PARAMETER HISTOGRAMS

    def weight_hist(self, trained=False, n_syn=False, **kwargs):
        """Plots the weight histogram."""
        weights = self._weights(trained, n_syn)
        return plots.param_hist(weights, ylabel="Weight", **kwargs)

    def node_param_hist(
        self,
        scale=5,
        fig=None,
        trained=False,
        fontsize=8,
        aspect_ratio=1,
        **kwargs,
    ):
        """Plots histograms of all node parameters."""
        node_params = list(self.tnn.spec.network.node_config.keys())
        fig, axes, _ = plt_utils.get_axis_grid(
            range(len(node_params)),
            scale=scale,
            fig=fig,
            aspect_ratio=aspect_ratio,
        )

        for i, p in enumerate(node_params):
            _p = p + "_prior" if not trained else p + "_trained"
            params = self.nodes[_p]
            xlabel = (
                "Prior " + p.replace("_", " ")
                if not trained
                else "trained " + p.replace("_", " ")
            )
            plots.param_hist(
                params,
                fig=fig,
                ax=axes[i],
                xlabel=xlabel,
                fontsize=fontsize,
                **kwargs,
            )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig, axes

    def edge_param_hist(
        self,
        scale=5,
        fig=None,
        trained=False,
        fontsize=8,
        aspect_ratio=1,
        **kwargs,
    ):
        """Plots histograms of all edge parameters."""
        edge_params = list(self.tnn.spec.network.edge_config.keys())
        edge_params.append("weight")
        fig, axes, _ = plt_utils.get_axis_grid(
            range(len(edge_params)),
            scale=scale,
            fig=fig,
            aspect_ratio=aspect_ratio,
        )

        for i, p in enumerate(edge_params):
            _p = p + "_prior" if not trained else p + "_trained"
            params = self.edges[_p]
            xlabel = (
                "Prior " + p.replace("_", " ")
                if not trained
                else "trained " + p.replace("_", " ")
            )
            plots.param_hist(
                params,
                fig=fig,
                ax=axes[i],
                xlabel=xlabel,
                fontsize=fontsize,
                **kwargs,
            )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig, axes

    # --- MEIS

    def generate_nmei(
        self, node_type, subwrap="augmented_sintel", mode="transient", **kwargs
    ):

        if not self._initialized["mei"] == subwrap:
            self.init_mei(subwrap=subwrap)

        nmei, activity = self.mei.natural_mei(node_type, subwrap=subwrap, mode=mode)

        u, v = self.get_uv(node_type)
        return plot_stim_response(
            nmei, activity, dt=1 / 100, u=u, v=v, steps=10, **kwargs
        )

    def generate_rnmei(
        self,
        node_type,
        subwrap="augmented_sintel",
        mode="transient",
        figsize=[5, 2],
        **kwargs,
    ):

        if not self._initialized["mei"] == subwrap:
            self.init_mei(subwrap=subwrap)

        (
            rnmei,
            rnmei_response,
            central_predicted_activity,
            central_target_activity,
            nmei,
            nmei_response,
            losses,
        ) = self.mei.rnmei(node_type, subwrap=subwrap, mode=mode)

        u, v = self.get_uv(node_type)
        return plot_stim_response(
            rnmei,
            rnmei_response,
            dt=1 / 100,
            u=u,
            v=v,
            steps=10,
            figsize=figsize,
            **kwargs,
        )

    def plot_nmei(self, node_type, subwrap="nmeis", **kwargs):

        self.init_nmeis(subwrap=subwrap)
        nmei = self.nmeis[node_type].stimulus[:]
        nmei_response = self.nmeis[node_type].response[:]

        u, v = self.get_uv(node_type)
        return plot_stim_response(
            nmei, nmei_response, dt=1 / 100, u=u, v=v, steps=10, **kwargs
        )

    def plot_rnmei(
        self,
        node_type,
        figsize=[5, 2],
        subwrap="rnmeis",
        **kwargs,
    ):
        self.init_rnmeis(subwrap=subwrap)

        rnmei = self.rnmeis[node_type].stimulus[:]
        rnmei_response = self.rnmeis[node_type].response[:]
        u, v = self.get_uv(node_type)
        return plot_stim_response(
            rnmei,
            rnmei_response,
            dt=1 / 100,
            u=u,
            v=v,
            steps=10,
            figsize=figsize,
            **kwargs,
        )

    def plot_stim_response(
        self, cell_type, stimulus, response, dt, figsize=[5, 2], **kwargs
    ):
        """

        Args:
            cell_type (str)
            stimulus (array): (1, n_frames, 1, n_cells)
            response (array): (1, n_frames, n_cells)
        """
        u, v = self.get_uv(cell_type)
        return plot_stim_response(
            stimulus,
            response,
            dt=dt,
            u=u,
            v=v,
            steps=10,
            figsize=figsize,
            **kwargs,
        )

    # --- STRFS

    def spatial_rfs_EMD_inputs(
        self,
        intensity,
        subwrap="impulse_v2",
        maxmode="central",
        share_color_range=False,
        footnote_known_narrow_broad=False,
        time_window=[0.05, 0.2],  # s
        average_over_column_radius=0,  # 0 corresponds to one column, 1 to
    ):
        column_mask = HexLattice.filled_ring(
            radius=average_over_column_radius, as_lattice=True
        ).where(1)

        if self._initialized["dots"] != subwrap:
            self.init_dots(subwrap=subwrap)

        if intensity not in self.dots.intensities:
            raise ValueError(
                f"valid intensities for {subwrap} are {self.dots.intensities}"
            )

        node_types = np.array(
            [
                [
                    "Mi1",
                    "Tm3",
                    "Mi4",
                    "Mi9",
                    "CT1(M10)",
                ],
                ["Tm1", "Tm2", "Tm4", "Tm9", "CT1(Lo1)"],
            ]
        )

        rfs = []
        for i, row in enumerate(node_types):
            for j, node_type in enumerate(row):
                rf = self.dots.receptive_field(node_type, intensity)
                rf -= rf[0]
                rf = rf[
                    int(time_window[0] / self.dots.dt) : int(
                        time_window[1] / self.dots.dt
                    )
                ]
                if maxmode == "central":
                    rf = rf[np.argmax(np.abs(rf[:, column_mask].mean(axis=1)))]
                elif maxmode == "individual":
                    rf = rf[np.argmax(np.abs(rf), axis=0), range(rf.shape[1])]
                else:
                    raise ValueError(
                        "mode for taking the maximum must be 'central' or 'individual'"
                    )
                rfs.append(rf)
        rfs = np.array(rfs).reshape(*node_types.shape, 721)

        if share_color_range:
            crange = np.max(np.abs(rfs))
            logging.info(
                "if you are sharing color ranges to judge contributions hold on: voltages of different cell types can be scaled differently by the respective weights. thus, I suggest to compare currents ;)"
            )

        x, y = hex_rows(2, 5)

        fig, axes, pos = dvs.plots.plt_utils.regular_ax_scatter(
            x, y, figsize=[3.5, 2], hpad=0, wpad=0.1, wspace=-0.5, hspace=-0.4
        )

        axes = np.array(axes).reshape(2, 5)

        for i, row in enumerate(node_types):
            for j, node_type in enumerate(row):
                rf = rfs[i, j]

                if not share_color_range:
                    crange = np.max(np.abs(rf))

                fig, ax, _ = dvs.plots.quick_hex_scatter(
                    rf,
                    cmap=plt.cm.coolwarm,
                    vmin=-crange,
                    vmax=crange,
                    midpoint=0,
                    cbar=False,
                    max_extent=4,
                    figsize=[3, 3],
                    fig=fig,
                    ax=axes[1 - i, j],
                )
                ax.annotate(
                    node_type,
                    (0.5, 0.0),
                    fontsize=6,
                    color="k",
                    xycoords="axes fraction",
                    va="center",
                    ha="center",
                    zorder=1,
                )

        for ax in axes.flatten():
            dvs.analysis.strfs.patch_text_color_polarity(ax)
            dvs.analysis.strfs.patch_CT1_texts_v2(ax)
            if footnote_known_narrow_broad:
                dvs.analysis.strfs.patch_known_narrowbroad(ax)
        return fig, axes

    def temporal_rfs_EMD_inputs(
        self,
        intensity=1,
        subwrap="impulse_v2",
        footnote_known_fast_slow=False,
        label_panels=True,
        time=None,
        t_lims=(0, 0.4),
        # two_column_impulses_dataset=None,
        # two_column_impulses=["Tm3", "Tm4"],
        average_over_column_radius=0,  # 0 corresponds to one column, 1 to
    ):
        column_mask = HexLattice.filled_ring(
            radius=average_over_column_radius, as_lattice=True
        ).where(1)

        if self._initialized["dots"] != subwrap:
            self.init_dots(subwrap=subwrap)

        if intensity not in self.dots.intensities:
            raise ValueError(
                f"valid intensities for {subwrap} are {self.dots.intensities}"
            )
        if time is None:
            time = np.arange(0, self.dots.t_stim + self.dots.t_post, self.dots.dt)
        mask = (time >= t_lims[0]) & (time <= t_lims[1])

        #
        x, y = [2, 0, 1], [1, 1, 1]
        fig, axes, pos = dvs.plots.plt_utils.regular_ax_scatter(
            x, y, figsize=[3.0, 2.5], hpad=0, wpad=0, wspace=0.5, hspace=0.15
        )

        axes = np.array(axes).reshape(3, 1)
        for ax in axes.flatten():
            dvs.plots.rm_spines(ax)

        for ax in axes.flatten():
            ax.set_alpha(0)

        ax = axes[0, 0]
        nodes = ["L1", "L2", "L3", "L4", "L5"][::-1]
        activity = []
        for node_type in nodes:
            y = self.dots.receptive_field(node_type, intensity)[mask][:, column_mask]
            y -= y[0]
            y = y.mean(axis=1)
            y = y / np.max(np.abs(y))
            activity.append(y)

        activity = np.array(activity)
        colors = (plt.cm.Blues(256),) * activity.shape[0]
        fig, ax = dvs.analysis.strfs.temporal_filters(
            time[mask],
            activity,
            nodes,
            fig=fig,
            ax=ax,
            fontsize=6,
            max_x_offset=0.0,
            y_offset=2.1,
            colors=colors,
            vmin=None,
            vmax=None,
            midpoint=None,
            fix_y_offset=True,
        )

        ax.set_xlabel("", fontsize=6)
        ax.set_xticks([])
        if label_panels:
            ax.set_title("lamina cells", fontsize=6, pad=2)

        ax = axes[1, 0]
        nodes = ["Mi1", "Tm3", "Mi4", "Mi9", "CT1(M10)"][::-1]
        activity = []
        for node_type in nodes:
            # if (
            #     node_type in two_column_impulses
            #     and two_column_impulses_dataset is not None
            # ):
            #     y = two_column_impulses_dataset.receptive_field(
            #         node_type, intensity
            #     )[mask][:, column_mask]
            # else:
            y = self.dots.receptive_field(node_type, intensity)[mask][:, column_mask]
            y -= y[0]
            y = y.mean(axis=1)
            y = y / np.max(np.abs(y))
            activity.append(y)

        activity = np.array(activity)
        fig, ax = dvs.analysis.strfs.temporal_filters(
            time[mask],
            activity,
            nodes,
            fig=fig,
            ax=ax,
            fontsize=6,
            max_x_offset=0.0,
            y_offset=2.1,
            colors=colors,
            vmin=None,
            vmax=None,
            midpoint=None,
            fix_y_offset=True,
        )
        ax.set_xlabel("", fontsize=6)
        if label_panels:
            ax.set_title("T4 inputs", fontsize=6, pad=2)

        ax = axes[2, 0]
        nodes = ["Tm1", "Tm2", "Tm4", "Tm9", "CT1(Lo1)"][::-1]
        activity = []
        for node_type in nodes:
            # if (
            #     node_type in two_column_impulses
            #     and two_column_impulses_dataset is not None
            # ):
            #     y = two_column_impulses_dataset.receptive_field(
            #         node_type, intensity
            #     )[mask][:, column_mask]
            # else:
            y = self.dots.receptive_field(node_type, intensity)[mask][:, column_mask]
            y -= y[0]
            y = y.mean(axis=1)
            y = y / np.max(np.abs(y))
            activity.append(y)

        activity = np.array(activity)
        fig, ax = dvs.analysis.strfs.temporal_filters(
            time[mask],
            activity,
            nodes,
            fig=fig,
            ax=ax,
            fontsize=6,
            max_x_offset=0.0,
            y_offset=2.1,
            colors=colors,
            vmin=None,
            vmax=None,
            midpoint=None,
            fix_y_offset=True,
        )
        ax.spines["bottom"].set_visible(False)
        if label_panels:
            ax.set_title("T5 inputs", fontsize=6, pad=2)
        ax.set_xlabel("")
        ax.set_xticks([])

        for ax in axes.flatten():
            ax.set_ylim(-0.1, 12.6)
            dvs.analysis.strfs.patch_text_color_polarity(ax)
            dvs.analysis.strfs.patch_CT1_texts_v2(ax)
            if footnote_known_fast_slow:
                dvs.analysis.strfs.patch_known_fastslow(ax)

        return fig, axes

    # --- Recordings

    def pred_eyal_rec(
        self,
        angle,
        offset,
        t_stim,
        width,
        intensity,
        node_type,
        fig=None,
        ax=None,
        title=None,
    ):

        sample = self.eyal_dataset.get_params(
            angle=angle,
            offset=offset,
            t_stim=t_stim,
            width=width,
            intensity=intensity,
        )
        if len(sample.index) == 1:
            sample = sample.index.values.item()
        else:
            return None, None, None, None, None

        return (
            self.prediction_of_eyal_recording(sample, node_type, fig, ax, title=title),
            sample,
        )

    def pred_eyal_rec_v2(
        self,
        angle,
        offset,
        t_stim,
        width,
        intensity,
        node_type,
        fig=None,
        ax=None,
        title=None,
    ):

        sample = self.eyal_dataset.get_params(
            angle=angle,
            offset=offset,
            t_stim=t_stim,
            width=width,
            intensity=intensity,
        )
        if len(sample.index) == 1:
            sample = sample.index.values.item()
        else:
            return None, None, None, None, None

        return (
            self.prediction_of_eyal_recording_v2(
                sample, node_type, fig, ax, title=title
            ),
            sample,
        )

    def pred_eyal_rec_grid(
        self, node_type, width, intensity, t_stim, fig=None, axes=None, title=""
    ):

        offsets = np.arange(-2.5, 3.0, 0.5)

        if "a" in node_type:
            angle = 180
        elif "b" in node_type:
            angle = 0
        elif "c" in node_type:
            angle = 90
        elif "d" in node_type:
            angle = 270
        else:
            raise ValueError

        fig, axes, _ = plt_utils.get_axis_grid(
            gridwidth=len(offsets), gridheight=1, axes=axes
        )

        ymin = 1e15
        ymax = -1e15

        samples = []
        for i, offset in enumerate(offsets):

            (fig, ax, ylim, train_or_val), sample = self.pred_eyal_rec(
                angle,
                offset,
                t_stim,
                width,
                intensity,
                node_type,
                fig=fig,
                ax=axes[i],
                title=title,
            )
            if fig is not None:
                samples.append(sample)
                _ymin = ylim[0]
                _ymax = ylim[1]
                ymin = min(ymin, _ymin)
                ymax = max(ymax, _ymax)
                ax.set_title(train_or_val, fontsize=10)
            else:
                return None, None, None, None

        for i, ax in enumerate(axes):
            ax.set_ylim(ymin, ymax)
            if i == 0:
                ax.set_ylabel("activity", fontsize=10)
            else:
                ax.set_ylabel("")
                plt_utils.rm_spines(ax, spines=["left"], rm_yticks=True)

        for ax in axes[:-1]:
            ax.legend_ = None

        fig.suptitle(
            f"{node_type}, width: {width}, intensity: {intensity}, duration: {t_stim}",
            y=1.1,
            fontsize=10,
        )

        return fig, axes, (ymin, ymax), samples

    def pred_eyal_rec_grid_v2(
        self, node_type, width, intensity, t_stim, fig=None, axes=None, title=""
    ):

        offsets = np.arange(-2.5, 3.0, 0.5)

        if "a" in node_type:
            angle = 0
        elif "b" in node_type:
            angle = 180
        elif "c" in node_type:
            angle = 90
        elif "d" in node_type:
            angle = 270
        else:
            raise ValueError

        fig, axes, _ = plt_utils.get_axis_grid(
            gridwidth=len(offsets), gridheight=1, axes=axes
        )

        ymin = 1e15
        ymax = -1e15

        samples = []
        for i, offset in enumerate(offsets):

            (fig, ax, ylim, train_or_val), sample = self.pred_eyal_rec_v2(
                angle,
                offset,
                t_stim,
                width,
                intensity,
                node_type,
                fig=fig,
                ax=axes[i],
                title=title,
            )
            if fig is not None:
                samples.append(sample)
                _ymin = ylim[0]
                _ymax = ylim[1]
                ymin = min(ymin, _ymin)
                ymax = max(ymax, _ymax)
                ax.set_title(train_or_val, fontsize=10)
            else:
                return None, None, None, None

        for i, ax in enumerate(axes):
            ax.set_ylim(ymin, ymax)
            if i == 0:
                ax.set_ylabel("activity", fontsize=10)
            else:
                ax.set_ylabel("")
                plt_utils.rm_spines(ax, spines=["left"], rm_yticks=True)

        for ax in axes[:-1]:
            ax.legend_ = None

        fig.suptitle(
            f"{node_type}, width: {width}, intensity: {intensity}, duration: {t_stim}",
            y=1.1,
            fontsize=10,
        )

        return fig, axes, (ymin, ymax), samples

    def prediction_of_eyal_recording(
        self, sample, node_type, fig=None, ax=None, title=None
    ):

        # Get stim parameter, recordings, and response
        params = self.eyal_dataset.params.iloc[[sample]]
        recordings = getattr(
            self.eyal_dataset.items_on_gpu[sample].recordings, node_type, None
        )
        response = self.tnn.full_data_eval.y_est.eyal[node_type][sample]
        time = np.arange(len(response)) * self.tnn.full_data_eval.dt[()]

        if recordings is None:
            return None, None, None, None

        fig, ax, _, _ = plots.traces(
            recordings.cpu().numpy(),
            color="0.9",
            x=time,
            highlight_mean=True,
            fig=fig,
            ax=ax,
        )

        # Get response
        ax.plot(
            time,
            response,
            label=f"{self.tnn.path.name}",
            color="#FF5733",
            linewidth=2,
        )
        ax.set_xlabel("time in s")
        ax.set_ylabel("activity")
        ax.legend(fontsize=10)
        if title is None:
            ax.set_title(
                (
                    f"type: {node_type}, angle: {params.angle.item()}, "
                    f"offset: {params.offset.item()}, \n "
                    f"duration: {params.t_stim.item()}, "
                    f"width: {params.width.item()}, "
                    f"contrast: {params.intensity.item()}"
                ),
                fontsize=10,
            )
        else:
            ax.set_title(title, fontsize=10)

        train_or_val = (
            "validation (unseen)"
            if sample in self.tnn.val_data_index[:]
            else "train (seen)"
        )
        fig.suptitle(train_or_val, fontsize=10)

        return fig, ax, ax.get_ylim(), train_or_val

    def prediction_of_eyal_recording_with_input(self, sample, node_type):

        params = self.eyal_dataset.params.iloc[[sample]]
        data = self.eyal_dataset.items_on_gpu[sample]

        if not hasattr(data.recordings, node_type):
            return None, None, None, None

        fig, axes, (_, _) = plt_utils.get_axis_grid(
            gridwidth=2, gridheight=1, wspace=0.3, figsize=[12, 4]
        )

        stim_onset = int(params.t_pre / params.dt) + 1
        plots.quick_hex_scatter(
            data.flash[stim_onset].squeeze(),
            fig=fig,
            ax=axes[0],
            vmin=0,
            vmax=1,
        )

        _, _, ylim, train_or_val = self.prediction_of_eyal_recording(
            sample, node_type, fig=fig, ax=axes[1]
        )

        return fig, axes, ylim, train_or_val

    def prediction_of_eyal_recording_v2(
        self, sample, node_type, fig=None, ax=None, title=None, subwrap=""
    ):

        # Get stim parameter, recordings, and response
        params = self.eyal_dataset.params.iloc[[sample]]
        recordings = getattr(
            self.eyal_dataset.items_on_gpu[sample].recordings, node_type, None
        )
        response = self.tnn.recordings.full_data_eval.y_est.eyal[node_type][sample]
        time = np.arange(len(response)) * self.tnn.recordings.full_data_eval.dt[()]

        if recordings is None:
            return None, None, None, None

        fig, ax, _, _ = plots.traces(
            recordings.cpu().numpy(),
            color="0.9",
            x=time,
            highlight_mean=True,
            fig=fig,
            ax=ax,
        )

        # Get response
        ax.plot(
            time,
            response,
            label=f"{self.tnn.path.name}",
            color="#FF5733",
            linewidth=2,
        )
        ax.set_xlabel("time in s")
        ax.set_ylabel("activity")
        ax.legend(fontsize=10)
        if title is None:
            ax.set_title(
                (
                    f"type: {node_type}, angle: {params.angle.item()}, "
                    f"offset: {params.offset.item()}, \n "
                    f"duration: {params.t_stim.item()}, "
                    f"width: {params.width.item()}, "
                    f"contrast: {params.intensity.item()}"
                ),
                fontsize=10,
            )
        else:
            ax.set_title(title, fontsize=10)

        train_or_val = (
            "validation (unseen)"
            if sample in self.tnn.recordings.full_data_eval.val_data_index[:]
            else "train (seen)"
        )
        fig.suptitle(train_or_val, fontsize=10)

        return fig, ax, ax.get_ylim(), train_or_val


# -- animations to video -------------------------------------------------------


def sintel_anim(
    nnv, fr_scale=1 / 8, batch_type="validation", samples=range(10), **kwargs
):
    """Plots a pdf with the Sintel Animation."""
    anim = nnv.sintel_animation(batch_type=batch_type, **kwargs)
    framerate = int(fr_scale / nnv.tnn[batch_type].dt[()])
    anim.to_vid(
        fname=f"sintel_anim_{batch_type}_{fr_scale:.2G}",
        framerate=framerate,
        dpi=200,
        samples=samples,
        delete_if_exists=True,
    )
    # store in real time
    anim.convert(
        fname=f"sintel_anim_{batch_type}_rt",
        framerate=int(1 / nnv.tnn[batch_type].dt[()]),
        delete_if_exists=True,
    )


def layer_activity(
    nnv,
    fr_scale=1 / 8,
    batch_type="validation",
    activity_type="activity",
    samples=range(10),
    **kwargs,
):
    anim = nnv.layer_animation(
        batch_type=batch_type, activity_type=activity_type, **kwargs
    )
    framerate = int(fr_scale / nnv.tnn[batch_type].dt[()])
    anim.to_vid(
        fname=f"network_{activity_type}_{batch_type}_{fr_scale:.2G}",
        framerate=framerate,
        dpi=100,
        samples=samples,
        delete_if_exists=True,
    )
    # store in real time
    anim.convert(
        fname=f"network_{activity_type}_{batch_type}_rt",
        framerate=int(1 / nnv.tnn[batch_type].dt[()]),
        delete_if_exists=True,
    )


def input_recording(
    nnv, fr_scale=1 / 8, batch_type="full_val", samples=range(10), **kwargs
):
    anim = nnv.input_recording_animation(batch_type=batch_type)
    framerate = int(fr_scale / nnv.tnn[batch_type].dt[()])
    anim.to_vid(
        fname=f"input_recording_{batch_type}_{fr_scale:.2G}",
        framerate=framerate,
        dpi=100,
        samples=samples,
        delete_if_exists=True,
    )
    # store in real time
    anim.convert(
        fname=f"input_recording_{batch_type}_rt",
        framerate=int(1 / nnv.tnn[batch_type].dt[()]),
        delete_if_exists=True,
    )


# -- plots to pdfs -------------------------------------------------------------

A4_HORIZONTAL = [11.69, 8.27]
A4_VERTICAL = [8.27, 11.69]

TITLE_FONTSIZE = 10
FONTSIZE = 6


PDFS = [  # "network_graph",
    "loss",
    "activity",
    "connectivity_matrix",
    "param_hist",
    "param_scatter",
    # "receptive_fields",
    # "projective_fields", # fix pf_index needed
    "motion_tuning",
    "stimuli_responses",
    "stimuli_response_ON_OFF_pathway",
    "dsi",
    "fri",
    "dead_ratio",
    "time_constants",
]

PDFS_TRAINED = [
    "connectivity_matrix",
    "param_hist",
    #  "receptive_fields",
    #  "projective_fields",
]

VIDS = ["sintel_anim", "layer_activity", "input_recording"]


def mk_pdf_dir(func):
    @functools.wraps(func)
    def wrapper(nnv, *args, **kwargs):
        # Create directory for storing pdfs.
        if not (nnv.tnn.path / "pdfs").exists():
            (nnv.tnn.path / "pdfs").mkdir()
        return func(nnv, *args, **kwargs)

    return wrapper


@mk_pdf_dir
def time_constants(nnv, **kwargs):
    """Plots a pdf with the network graph."""
    logging.info("pdfpages: time_constants.pdf")
    with PdfPages(nnv.tnn.pdfs.path / f"time_constants.pdf") as pdf:
        nnv.time_constants()
        plt.tight_layout()
        pdf.savefig()
        plt.close()


# @mk_pdf_dir
# def network_graph(nnv, **kwargs):
#     """Plots a pdf with the network graph."""
#     logging.info("pdfpages: network_graph.pdf")
#     with PdfPages(nnv.tnn.pdfs.path / f"nn_graph.pdf") as pdf:
#         fig = plt_utils.figure(figsize=A4_HORIZONTAL)
#         fig.suptitle("Network Graph")
#         nnv.node_layout_graph(fig=fig)
#         pdf.savefig()
#         plt.close()


@mk_pdf_dir
def loss(nnv, keys=["validation", "training", "train"], **kwargs):
    """Plots a pdf with loss and activity over iterations."""
    logging.info("pdfpages: loss.pdf")
    filename = (nnv.tnn.pdfs.path / kwargs.pop("filename", "loss.pdf")).with_suffix(
        ".pdf"
    )
    filename.parent.mkdir(exist_ok=True)
    with PdfPages(filename) as pdf:
        fig = plt_utils.figure(figsize=A4_VERTICAL)
        fig.suptitle("losses", fontsize=10)
        try:
            ax = plt_utils.subplot("training", grid=(2, 1), location=(0, 0))
            nnv.training_loss(fig=fig, ax=ax)
        except Exception as e:
            logging.warning(f"Could not plot training loss {e}.")
            raise
        try:
            ax = plt_utils.subplot("validation", grid=(2, 1), location=(1, 0))
            nnv.test_loss(mode="validation", fig=fig, ax=ax)
        except Exception as e:
            logging.warning(f"Could not plot validation set loss {e}.")
            raise
        # try:
        #     ax = plt_utils.subplot("training", grid=(3, 1), location=(2, 0))
        #     nnv.test_loss(mode="training", fig=fig, ax=ax)
        # except Exception as e:
        #     logging.warning(f"Could not plot training set loss {e}.")
        #     raise

        # ymin = 10e15
        # ymax = -10e15
        # try:
        #     ax = plt_utils.subplot(
        #         "validation set", grid=(3, 1), location=(0, 0)
        #     )
        #     nnv.loss(keys[0], fig=fig, ax=ax)
        #     ylim = ax.get_ylim()
        #     ymin = min(ymin, ylim[0])
        #     ymax = max(ymax, ylim[1])
        # except Exception as e:
        #     logging.warning(f"Could not plot training loss {e}.")
        #     # traceback.print_exc()
        #     raise
        # try:
        #     ax = plt_utils.subplot(
        #         "training set", grid=(3, 1), location=(1, 0)
        #     )
        #     nnv.loss(keys[1], fig=fig, ax=ax)
        #     ylim = ax.get_ylim()
        #     ymin = min(ymin, ylim[0])
        #     ymax = max(ymax, ylim[1])
        #     ax.set_ylim(ymin, ymax)
        # except Exception as e:
        #     logging.warning(f"Could not plot training loss {e}.")
        #     # traceback.print_exc()
        #     raise
        # try:
        #     ax = plt_utils.subplot(
        #         "during training", grid=(3, 1), location=(2, 0)
        #     )
        #     nnv.loss(keys[2], fig=fig, ax=ax, smooth=0.05)
        # except Exception as e:
        #     logging.warning(f"Could not plot training loss {e}.")
        #     # traceback.print_exc()
        #     raise
        pdf.savefig()
        plt.close()


@mk_pdf_dir
def activity(nnv, **kwargs):
    """Plots a pdf with loss and activity over iterations."""
    logging.info("pdfpages: activity.pdf")
    with PdfPages(nnv.tnn.pdfs.path / "activity.pdf") as pdf:
        fig = plt_utils.figure(figsize=A4_VERTICAL)
        fig.suptitle("Activity")
        ax = plt.subplot(211)
        nnv.activity(fig=fig, ax=ax)
        ax = plt.subplot(212)
        nnv.activity_bars(fig=fig, ax=ax)
        pdf.savefig()
        plt.close()


@mk_pdf_dir
def connectivity_matrix(nnv, trained=False, **kwargs):
    """Plots a pdf with the connectivity matrices."""
    title = "untrained" if not trained else "trained"
    logging.info(f"pdfpages: connectivity_{title}.pdf")
    with PdfPages(nnv.tnn.pdfs.path / f"connectivity_{title}.pdf") as pdf:
        fig, ax, _, _ = nnv.connectivity_matrix(
            plot_type="count", fontsize=4, trained=trained
        )
        fig.suptitle(title)
        pdf.savefig()
        plt.close()
        fig, ax, _, _ = nnv.connectivity_matrix(
            plot_type="weight", fontsize=4, trained=trained
        )
        pdf.savefig()
        plt.close()
        fig, ax, _, _ = nnv.connectivity_matrix(
            plot_type="n_syn", fontsize=4, trained=trained
        )
        pdf.savefig()
        plt.close()


@mk_pdf_dir
def param_hist(nnv, trained=False, **kwargs):
    """Plots a pdf with parameter histrograms."""
    title = "untrained" if not trained else "trained"
    logging.info(f"pdfpages: param_histograms_{title}.pdf")
    with PdfPages(nnv.tnn.pdfs.path / f"param_histograms_{title}.pdf") as pdf:
        fig = plt_utils.figure(figsize=A4_VERTICAL)
        fig.suptitle(f"{title} Node Parameters")
        nnv.node_param_hist(
            fig=fig,
            trained=trained,
            fontsize=6,
            aspect_ratio=np.divide(*A4_VERTICAL),
        )
        pdf.savefig()
        plt.close()

        fig = plt_utils.figure(figsize=A4_VERTICAL)
        fig.suptitle(f"{title} Edge Parameters")
        nnv.edge_param_hist(
            fig=fig,
            trained=trained,
            fontsize=6,
            aspect_ratio=np.divide(*A4_VERTICAL),
        )
        pdf.savefig()
        plt.close()


@mk_pdf_dir
def param_scatter(nnv, **kwargs):
    """Plots a pdf with parameter scatters."""
    logging.info(f"pdfpages: param_scatter_trained_vs_untrained.pdf")
    with PdfPages(nnv.tnn.pdfs.path / f"param_scatter_trained_vs_untrained.pdf") as pdf:
        fig = plt_utils.figure(figsize=A4_VERTICAL)
        fig.suptitle("Node Parameters")
        nnv.node_param_scatter(fig=fig, fontsize=6)
        pdf.savefig()
        plt.close()

        fig = plt_utils.figure(figsize=A4_VERTICAL)
        fig.suptitle("Edge Parameters")
        nnv.edge_param_scatter(fig=fig, fontsize=6)
        pdf.savefig()
        plt.close()


@mk_pdf_dir
def receptive_fields(nnv, trained=False, **kwargs):
    """Plots a pdf with receptive fields."""
    title = "untrained" if not trained else "trained"
    logging.info(f"pdfpages: receptive_fields_{title}.pdf")
    with PdfPages(nnv.tnn.pdfs.path / f"receptive_fields_{title}.pdf") as pdf:
        title += " Receptive Field {}"
        for target in tqdm(nnv.node_types_sorted, desc="Plotting receptive fields"):
            fig = plt_utils.figure(figsize=A4_VERTICAL)
            fig.suptitle(title.format(target))
            nnv.receptive_fields_grid(
                target,
                fig=fig,
                fontsize=6,
                trained=trained,
                aspect_ratio=np.divide(*A4_VERTICAL),
            )
            pdf.savefig()
            plt.close()


@mk_pdf_dir
def projective_fields(nnv, trained=False, **kwargs):
    """Plots a pdf with projective fields."""
    title = "untrained" if not trained else "trained"
    logging.info(f"pdfpages: projective_fields_{title}.pdf")
    with PdfPages(nnv.tnn.pdfs.path / f"projective_fields_{title}.pdf") as pdf:
        title += " Projective Field {}"
        for source in tqdm(nnv.node_types_sorted, desc="Plotting projective fields"):
            fig = plt_utils.figure(figsize=A4_VERTICAL)
            fig.suptitle(title.format(source))
            nnv.projective_fields_grid(
                source,
                fig=fig,
                fontsize=6,
                trained=trained,
                aspect_ratio=np.divide(*A4_VERTICAL),
            )
            pdf.savefig()
            plt.close()


@mk_pdf_dir
def stimuli_responses(nnv, node_types=None, **kwargs):
    logging.info(f"pdfpages: stimuli_responses.pdf")
    path = (nnv.tnn.pdfs.path / "stimuli_responses").with_suffix(".pdf")
    with PdfPages(path) as pdf:
        node_types = node_types if node_types else nnv.node_types_sorted
        for node_type in node_types:
            fig = plt_utils.figure(figsize=A4_HORIZONTAL)
            fig.suptitle(node_type)
            _ = nnv.stimuli_responses(
                node_type, fig=fig, figsize=np.array(A4_HORIZONTAL) * 2
            )
            pdf.savefig()
            plt.close()


@mk_pdf_dir
def motion_tuning(nnv, node_types=None, **kwargs):
    logging.info(f"pdfpages: motion_tuning.pdf")
    path = (nnv.tnn.pdfs.path / "motion_tuning").with_suffix(".pdf")
    with PdfPages(path) as pdf:
        node_types = node_types if node_types else nnv.node_types_sorted
        for node_type in node_types:
            _ = nnv.motion_tuning_all(node_type)
            pdf.savefig()
            plt.close()


@mk_pdf_dir
def flash_response_grid(nnv, **kwargs):
    logging.info("pdfpages: flash_response_grid.pdf")
    filename = (
        nnv.tnn.pdfs.path / kwargs.pop("filename", "flash_response_grid.pdf")
    ).with_suffix(".pdf")
    filename.parent.mkdir(exist_ok=True)
    with PdfPages(filename) as pdf:
        fig, axes = nnv.flash_response_grid(0, 1, 6)
        pdf.savefig()
        plt.close()


@mk_pdf_dir
def motion_tuning_grid(nnv, **kwargs):
    logging.info("pdfpages: motion_tuning_grid.pdf")
    filename = (
        nnv.tnn.pdfs.path / kwargs.pop("filename", "motion_tuning_grid.pdf")
    ).with_suffix(".pdf")
    filename.parent.mkdir(exist_ok=True)

    with PdfPages(filename) as pdf:
        fig, axes = nnv.motion_tuning_grid()
        pdf.savefig()
        plt.close()


@mk_pdf_dir
def stimuli_response_ON_OFF_pathway(nnv, node_types=None, **kwargs):
    logging.info("pdfpages: stimuli_responses_on_off_pathways.pdf")
    filename = (
        nnv.tnn.pdfs.path
        / kwargs.pop("filename", "stimuli_responses_on_off_pathways.pdf")
    ).with_suffix(".pdf")
    filename.parent.mkdir(exist_ok=True)

    with PdfPages(filename) as pdf:
        for pathway in ["on", "off"]:
            nnv.pathway_stimuli_response(pathway=pathway)
            pdf.savefig()
            plt.close()


@mk_pdf_dir
def dsi(nnv, **kwargs):
    """Plots a pdf with direction selectivity indices."""
    logging.info("pdfpages: dsi.pdf")
    with PdfPages(nnv.tnn.pdfs.path / f"dsi.pdf") as pdf:
        nnv.network_graph_dsi(title="DSI", fontsize=FONTSIZE)
        pdf.savefig()
        plt.close()


@mk_pdf_dir
def fri(nnv, **kwargs):
    """Plots a pdf with flash response indices."""
    logging.info("pdfpages: fri.pdf")
    with PdfPages(nnv.tnn.pdfs.path / f"fri.pdf") as pdf:
        fig = plt_utils.figure(figsize=A4_VERTICAL)
        ax = plt.subplot(311)
        nnv.network_graph_fri(
            mode="transient",
            fig=fig,
            ax=ax,
            title="FRI (transient)",
            fontsize=FONTSIZE,
        )
        ax = plt.subplot(312)
        nnv.network_graph_fri(
            mode="sustained",
            fig=fig,
            ax=ax,
            title="FRI (sustained)",
            fontsize=FONTSIZE,
        )

        ax = plt.subplot(313)
        nnv.network_graph_fri(
            groundtruth=True,
            fig=fig,
            ax=ax,
            title="FRI (groundtruth)",
            fontsize=FONTSIZE,
        )

        pdf.savefig()
        plt.close()


@mk_pdf_dir
def dead_ratio(nnv, **kwargs):
    """Plots a pdf with dead ratios."""
    logging.info("pdfpages: dead_ratio.pdf")
    with PdfPages(nnv.tnn.pdfs.path / f"dead_ratio.pdf") as pdf:
        nnv.dead_ratio(title="dead ratio", fontsize=FONTSIZE)
        pdf.savefig()
        plt.close()


def store(functions, nnv, **kwargs):
    for function in functions:
        try:
            function = globals()[function]
            function(nnv, **kwargs)
        except Exception as e:
            traceback.print_exc()
            logging.info(
                f"Exception occured for {nnv.tnn.path.parts[-1]}"
                f"in function {function.__name__}: {e}."
            )
