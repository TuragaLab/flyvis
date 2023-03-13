import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Tuple, Union
import textwrap
import traceback
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from toolz import valmap
from tqdm.auto import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps as cm
from matplotlib.colors import Colormap, Normalize, to_hex

from dvs import analysis
from dvs.analysis.validation_error import _check_loss_name
from dvs.datasets.flicker import _subtract_baseline
import dvs.analysis
from dvs import exp_dir, scheduler, utils
from dvs.analysis import clustering
from dvs.analysis.clustering import (
    EnsembleNaturalisticResponses,
    cluster_motion_tuning,
    gaussian_mixture,
    init_naturalistic_responses,
    plot_embedding,
    task_error_sort_labels,
)
from dvs.analysis.network_views import NetworkViews
from dvs.initialization import init_parameter
from dvs.plots import decoration, plots, plt_utils
from dvs.utils import color_utils, groundtruth_utils
from dvs.utils.datawrapper import Datawrap, Namespace
from dvs.utils.nodes_edges_utils import (
    order_nodes_list,
    param_to_tex,
    pretty_param_label,
)
from dvs.utils.tnn_utils import always_dead_questionmark
from dvs.utils.wrap_utils import is_up_to_date
from dvs.analysis.significance_tests_dsis import EnsembleMovingEdgeResponses

logging = logging.getLogger("dvs")

# -- Adding error bounds to normalize ensemble errors accordingly from 4 flownet
# networks on same train / test split flownets seemed to be overconfident on the
# test split - so this is a very generous lower bound for flownet
# LOWER_ERROR_BOUND_OVERCONFIDENT = 2.9885484129190445
# # from 4 flownet networks on random train / test splits
# LOWER_ERROR_BOUND = 4.4425379275361445
# # from 50 fly networks for which we only trained the decoder
# # this was based on kfold validation or augmented_sintel_validation
# # UPPER_ERROR_BOUND = 6.32194522023201
# # this is l2 on the overconfident original validation set
# UPPER_ERROR_BOUND = 5.740314735174179
# UPPER_ERROR_BOUND_KFOLD_VALIDATION = 5.8612595307992565


# achieved by VanillaHexCNN baseline, architecture in
# scripts/sintel/vanilla_hex_cnn_config/arch_kwargs/L_structured
# with 414666 parameters. trained with original Dataset object, same
# train and validation splits. 4 frames temporal context
LOWER_ERROR_BOUND = 4.814054543321783

# achieved by ensemble 0170 - randomly initialized paramaters (gaussian),
# only trained decoder.
UPPER_ERROR_BOUND = 5.692145279943943

# -- Methods to modularize the __init__ of the class Ensemble
def tnn_paths_from_parent(tnns):
    ensemble_path = exp_dir / tnns
    tnn_paths = sorted(
        filter(
            lambda p: p.name.isnumeric() and p.is_dir(),
            ensemble_path.iterdir(),
        )
    )
    return tnn_paths, ensemble_path


def tnn_paths_from_names_or_paths(tnns):
    tnn_paths = []
    _ensemble_paths = []
    for tnn in tnns:
        if isinstance(tnn, str):
            # assuming task/ensemble_id/model_id
            if len(tnn.split("/")) == 3:
                tnn_paths.append(exp_dir / tnn)
            # assuming task/ensemble_id
            elif len(tnn.split("/")) == 2:
                tnn_paths.extend(tnn_paths_from_parent(tnn)[0])
        elif isinstance(tnn, Path):
            tnn_paths.append(tnn)
        _ensemble_paths.append(tnn_paths[-1].parent)
    ensemble_path = np.unique(_ensemble_paths)
    return tnn_paths, ensemble_path


def tnn_path_names(tnn_paths):
    tnn_names = [str(path).replace(str(exp_dir) + "/", "") for path in tnn_paths]
    ensemble_name = ", ".join(np.unique([n[:-4] for n in tnn_names]).tolist())
    return tnn_names, ensemble_name


class Ensemble(dict):
    def __init__(self, tnns: Union[List[str], str]):
        super().__init__()

        if isinstance(tnns, str):
            self.tnn_paths, self.path = tnn_paths_from_parent(tnns)
        elif isinstance(tnns, Iterable):
            self.tnn_paths, self.path = tnn_paths_from_names_or_paths(tnns)

        self.names, self.name = tnn_path_names(self.tnn_paths)

        # Initialize datawraps.
        for i, name in enumerate(self.names):
            self[name] = Datawrap(self.tnn_paths[i])

        if isinstance(self.path, Path):
            self.wrap = Datawrap(self.path)
        elif isinstance(self.path, list):
            self.wrap = None

        # Initialize colors.
        colors = plt_utils.get_colors(len(self.names))
        self.colors = {name: to_hex(colors[i]) for i, name in enumerate(self.names)}

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __getitem__(
        self, key: Union[str, int, slice, np.ndarray]
    ) -> Union[Datawrap, "Ensemble"]:
        if isinstance(key, (int, np.integer)):
            return dict.__getitem__(self, self.names[key])
        elif isinstance(key, slice):
            return self.__class__(self.names[key])
        elif isinstance(key, np.ndarray):
            return self.__class__(np.array(self.names)[key])
        elif key in self.names:
            return dict.__getitem__(self, key)
        else:
            raise ValueError(f"{key}")

    def __dir__(self):
        return list({*dict.__dir__(self), *dict.__iter__(self)})

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        yield from self.names

    def items(self) -> Iterator[Tuple[str, Union[Datawrap, NetworkViews]]]:
        return iter((k, self[k]) for k in self)

    def keys(self):
        return list(self)

    def values(self) -> List[Union[Datawrap, NetworkViews]]:
        return [self[k] for k in self]

    def pop(self, key, *args):
        index = self.names.index(key)
        self.names.pop(index)
        # to keep track of indices and masks
        _model_index = self._model_index.tolist()
        _model_index.pop(index)
        self._model_index = _model_index
        _model_mask = self._model_mask.tolist()
        _model_mask.pop(index)
        self._model_mask = _model_mask
        return dict.pop(self, key, *args)

    @contextmanager
    def sort(
        self,
        mode="min",
        reverse=False,
        validation_subwrap="original_validation_v2",
        loss_name="epe",
    ):
        """
        To temporarily sort the ensemble based on a type of task error. In
        ascending order.

        This method sorts the self.names attribute temporarily, which serve as
        references to the Datawrap instances of the trained Networks.
        """
        if mode not in ["name", "min", "last", "mean"]:
            raise ValueError(
                f"mode {mode} must be one of 'name', 'min', 'last', 'mean'."
            )

        _names = deepcopy(self.names)
        loss_name = dvs.analysis.validation_error._check_loss_name(
            self[0][validation_subwrap], loss_name
        )

        try:
            if mode == "name":
                self.names = sorted(self.names)
            elif mode == "min":
                self.names = sorted(
                    self.keys(),
                    key=lambda key: self[key][validation_subwrap][loss_name][:].min(),
                    reverse=reverse,
                )
            elif mode == "last":
                self.names = sorted(
                    self.keys(),
                    key=lambda key: self[key][validation_subwrap][loss_name][-1],
                    reverse=reverse,
                )
            elif mode == "mean":
                self.names = sorted(
                    self.keys(),
                    key=lambda key: self[key][validation_subwrap][loss_name][:].mean(),
                    reverse=reverse,
                )
            else:
                raise ValueError
        except Exception as e:
            logging.info(f"sorting failed: {e}")
        try:
            yield
        finally:
            self.names = list(_names)


class EnsembleEval(Ensemble):

    _initialized = {}
    _edge_stats = ()
    _cache = {}
    naturalistic_responses: EnsembleNaturalisticResponses = None
    _naturalistic_responses = None

    def __init__(self, tnns):
        super().__init__(tnns)
        _unvalid = []
        for key, tnn in tqdm(self.items(), total=len(self), disable=dvs.disable_tqdm):
            try:
                self[key] = NetworkViews(tnn=tnn)
            except Exception as e:
                logging.warning(
                    f"Problem with {key}: {e}. Excluding {key} from analysis."
                )
                _unvalid.append(key)
                print(traceback.print_exc())
        for key in _unvalid:
            self.pop(key)
            self.names.pop(self.names.index(key))
        self._model_index = np.arange(len(self))
        self._model_mask = np.ones(len(self)).astype(bool)

    # def store_best_params(self):
    #     "store params from best checkpoint"
    #     import gc

    #     for name, nnv in self.items():
    #         if "best_network_parameters" not in nnv.tnn:
    #             nnv.init_network(checkpoint="best")
    #             nnv.tnn.best_network_parameters = valmap(
    #                 lambda x: x.detach().cpu().numpy(),
    #                 dict(nnv.network.named_parameters()),
    #             )
    #             logging.info(f"stored {name} best parameters")
    #             nnv.network = None
    #             gc.collect()
    #             torch.cuda.empty_cache()

    # def store_initial_params(self):
    #     import gc

    #     # TODO: this is a workaround we have not stored the initial parameters
    #     # cocnsistently per type, but we have stored them per node and edge, so
    #     # we need the scatter indices and perform a scatter_mean operation
    #     # afterwards to get them per type or connection type assuming all
    #     # networks in the ensemble use the same initialization class it does not
    #     # matter which network or checkpoint we get the indices from
    #     nnv0 = self[0]
    #     nnv0.init_network()
    #     param_api = nnv0.network._param_api()

    #     for name, nnv in self.items():
    #         param_dict = {}
    #         # make the
    #         if "initial_network_parameters" not in nnv.tnn:

    #             for param_name in list(
    #                 dict(nnv0.network.named_parameters()).keys()
    #             ):
    #                 # named network parameters are called "nodes_<>", "edges_<>"
    #                 element = param_name[: param_name.find("_")]
    #                 _param_name = param_name[param_name.find("_") + 1 :]

    #                 params = torch.tensor(
    #                     nnv.tnn.prior_param_api[element][_param_name][:]
    #                 )
    #                 reduced_params = utils.tensor_utils.RefTensor.reduce(
    #                     params,
    #                     param_api[element]
    #                     .get_as_reftensor(_param_name)
    #                     .indices,
    #                 )
    #                 param_dict[param_name] = reduced_params
    #             nnv.tnn.initial_network_parameters = valmap(
    #                 lambda x: x.cpu().numpy(),
    #                 param_dict,
    #             )
    #             # logging.info(f"stored {name} initial parameters")
    #     nnv.network = None
    #     gc.collect()
    #     torch.cuda.empty_cache()

    # def _check_param_stored(self, mode):
    #     if mode not in ["best", "initial"]:
    #         raise ValueError
    #     if not all(
    #         [f"{mode}_network_parameters" in nnv.tnn for nnv in self.values()]
    #     ):
    #         getattr(self, f"store_{mode}_params")()

    def __getitem__(
        self, key: Union[str, int, slice, np.ndarray]
    ) -> Union[NetworkViews, "EnsembleViews"]:
        return super().__getitem__(key)

    def init_movingbar(
        self, subwrap="movingedge_chkpt_best_v4", failsafe=False, device="cpu"
    ):
        if self._initialized["movingbar"] == subwrap:
            return
        self._initialized["movingbar"] = ""
        # stims are shared for all network objects, responses are not also, stim
        # initialization is slow therefore, init only first stim/response object
        # with stims and reference the shared stimulus related attributes in all
        # other stim/response objects
        nnv_0 = self[0]
        nnv_0.init_movingbar(
            subwrap=subwrap,
            build_stim_on_init=True,
            failsafe=failsafe,
            device=device,
        )
        for nnv in self.values()[1:]:
            nnv.init_movingbar(
                subwrap=subwrap,
                build_stim_on_init=False,
                failsafe=failsafe,
                device=device,
            )

            nnv.movingbar.sequences = nnv_0.movingbar.sequences
            nnv.movingbar.wrap = nnv_0.movingbar.wrap
            nnv.movingbar._offsets = nnv_0.movingbar._offsets
            nnv.movingbar._built = True

        self._initialized["movingbar"] = subwrap

    def init_normalizing_movingbar(
        self, subwrap="shuffled_movingsquare", failsafe=False, device="cpu"
    ):
        if self._initialized["normalizing_movingbar"] == subwrap:
            return
        self._initialized["normalizing_movingbar"] = ""
        # stims are shared for all network objects, responses are not also, stim
        # initialization is slow therefore, init only first stim/response object
        # with stims and reference the shared stimulus related attributes in all
        # other stim/response objects
        nnv_0 = self[0]
        nnv_0.init_normalizing_movingbar(
            subwrap=subwrap,
            build_stim_on_init=True,
            failsafe=failsafe,
            device=device,
        )
        for nnv in self.values()[1:]:
            nnv.init_normalizing_movingbar(
                subwrap=subwrap,
                build_stim_on_init=False,
                failsafe=failsafe,
                device=device,
            )

            nnv.normalizing_movingbar.sequences = nnv_0.normalizing_movingbar.sequences
            nnv.normalizing_movingbar.wrap = nnv_0.normalizing_movingbar.wrap
            nnv.normalizing_movingbar._offsets = nnv_0.normalizing_movingbar._offsets
            nnv.normalizing_movingbar._built = True

        self._initialized["normalizing_movingbar"] = subwrap

    def init_rectangular_flicker(
        self, subwrap="square_flicker", failsafe=False, device="cpu"
    ):
        if self._initialized["rectangular_flicker"] == subwrap:
            return
        self._initialized["rectangular_flicker"] = ""
        # stims are shared for all network objects, responses are not also, stim
        # initialization is slow therefore, init only first stim/response object
        # with stims and reference the shared stimulus related attributes in all
        # other stim/response objects
        nnv_0 = self[0]
        nnv_0.init_rectangular_flicker(
            subwrap=subwrap,
            build_stim_on_init=True,
            failsafe=failsafe,
            device=device,
        )
        for nnv in self.values()[1:]:
            nnv.init_rectangular_flicker(
                subwrap=subwrap,
                build_stim_on_init=False,
                failsafe=failsafe,
                device=device,
            )

            nnv.rectangular_flicker.sequences = nnv_0.rectangular_flicker.sequences
            nnv.rectangular_flicker.wrap = nnv_0.rectangular_flicker.wrap
            nnv.rectangular_flicker._flicker = nnv_0.rectangular_flicker._flicker
            nnv.rectangular_flicker._built = True

        self._initialized["rectangular_flicker"] = subwrap

    def init_oriented_bar(self, subwrap="oriented_bar", failsafe=False, device="cpu"):
        if self._initialized["oriented_bar"] == subwrap:
            return
        self._initialized["oriented_bar"] = ""
        nnv_0 = self[0]
        nnv_0.init_oriented_bar(
            subwrap=subwrap,
            build_stim_on_init=True,
            failsafe=failsafe,
            device=device,
        )
        for nnv in self.values()[1:]:
            nnv.init_oriented_bar(
                subwrap=subwrap,
                build_stim_on_init=False,
                failsafe=failsafe,
                device=device,
            )

            nnv.oriented_bar.sequences = nnv_0.oriented_bar.sequences
            nnv.oriented_bar.wrap = nnv_0.oriented_bar.wrap
            nnv.oriented_bar._offsets = nnv_0.oriented_bar._offsets
            nnv.oriented_bar._built = True
            # nnv.oriented_bar.dt = nnv_0.oriented_bar.dt

        self._initialized["oriented_bar"] = subwrap

    def init_naturalistic_response_clustering(self, subwrap="augmented_sintel_v2"):
        if self._initialized["naturalistic_response_cluster"] == subwrap:
            return
        self.naturalistic_response_clustering = clustering.NaturalisticReponseCluster(
            self, subwrap=subwrap
        )
        self._initialized["naturalistic_response_cluster"] = subwrap

    def init_naturalistic_responses(self, subwrap="augmented_sintel"):
        if self._initialized["naturalistic_responses"] == subwrap:
            return
        self._initialized["naturalistic_responses"] = ""
        # the responses are 3GB so the caching does not work.
        # if subwrap in ensemble.wrap:
        #     # to retrieve the responses from all models from top level directory
        #     # of the ensemble
        #     self._naturalistic_responses = ensemble.wrap[subwrap][:]
        # else:
        # collect and preprocess the responses from all models
        self._naturalistic_responses = init_naturalistic_responses(
            self, subwrap=subwrap
        )

        # # to cache the responses from all models in top level directory
        # # of the ensemble
        # ensemble.wrap[subwrap] = self._naturalistic_responses

        self._initialized["naturalistic_responses"] = subwrap

    @property
    def naturalistic_responses(
        self,
    ) -> clustering.EnsembleNaturalisticResponses:
        # call separate method to use cache based on model index
        return self._get_naturalistic_responses(self._model_index)

    @utils.cache
    def _get_naturalistic_responses(
        self, model_index
    ) -> clustering.EnsembleNaturalisticResponses:
        return clustering.EnsembleNaturalisticResponses(
            self._naturalistic_responses[model_index], self.ctome
        )

    def init_augmented_sintel(
        self, dt=None, subwrap="augmented_sintel", failsafe=False, device="cpu"
    ):
        if self._initialized["augmented_sintel"] == subwrap:
            return
        self._initialized["augmented_sintel"] = ""
        nnv_0 = self[0]
        nnv_0.init_augmented_sintel(
            subwrap=subwrap,
            build_stim_on_init=True,
            failsafe=failsafe,
            device=device,
        )
        dt = dt or nnv_0.augmented_sintel.dt

        for nnv in self.values()[1:]:
            nnv.init_augmented_sintel(
                subwrap=subwrap,
                build_stim_on_init=False,
                failsafe=failsafe,
                device=device,
                _init_cache=False,
            )

            nnv.augmented_sintel.sequences = nnv_0.augmented_sintel.sequences
            nnv.augmented_sintel.params = nnv_0.augmented_sintel.params
            nnv.augmented_sintel.arg_df = nnv_0.augmented_sintel.arg_df
            nnv.augmented_sintel.n_frames = nnv_0.augmented_sintel.n_frames
            nnv.augmented_sintel.dt = dt
            nnv.augmented_sintel._built = True

        self._initialized["augmented_sintel"] = subwrap

    def init_dots(self, subwrap="impulse_v4"):
        pass

    def init_flashes(self, subwrap="flashes", failsafe=False):
        if self._initialized["flashes"] == subwrap:
            return
        self._initialized["flashes"] = ""
        for nnv in self.values():
            nnv.init_flashes(subwrap=subwrap, failsafe=failsafe)
        self._initialized["flashes"] = subwrap

    def _eval_each(self, eval_fn):
        for name, nnv in self.items():
            eval_fn(nnv)

    def eval_ensemble(
        self,
        dataloader,
        loss_fns=None,
        dt=1 / 50,
        save=True,
        directory="validation",
        only_original_frames=False,
    ):

        import logging

        from dvs.solver import (
            recover_decoder,
            recover_network,
            stored_checkpoints,
        )

        dvs_logger = logging.getLogger("dvs")
        dvs_logger.setLevel(logging.WARNING)

        if loss_fns is None:
            loss_fns = [
                dvs.objectives.mrcse,
                dvs.objectives.mrmse,
                dvs.objectives.epe,
                dvs.objectives.mae,
                dvs.objectives.rmse,
                dvs.objectives.cosine,
            ]

        print(loss_fns)
        dataloader.dataset.dt = dt

        def eval_model(nnv):
            network = nnv.init_network()
            decoder = nnv.init_decoder()

            ids, paths = stored_checkpoints(nnv.tnn.chkpts.path)

            val_losses = []
            for chkpt in paths:
                recover_network(network, chkpt)
                recover_decoder(decoder, chkpt)
                val_loss, task_loss = dvs.analysis.validation_error.test(
                    network,
                    decoder,
                    dataloader,
                    loss_fns=loss_fns,
                    dt=dt,
                    only_original_frames=only_original_frames,
                )
                val_losses.append(val_loss)
            return np.array(val_losses)  # (n_chkpts, len(loss_fns))

        losses = []
        for nnv in self.values():
            losses.append(eval_model(nnv))  # (n_models, n_chkpts, len(loss_fns))

        losses = dict(
            zip(
                [fn.__name__ for fn in loss_fns],
                np.transpose(np.array(losses), (2, 0, 1)),
            )
        )
        if save:
            self.wrap[directory] = losses
        return losses

    def parameters(
        self,
        chkpt="best",
        validation_subwrap="original_validation_v2",
        loss_name="epe",
    ):
        network_params = {}
        for nnv in self.values():
            if chkpt == "best":
                loss_name = dvs.analysis.validation_error._check_loss_name(
                    nnv.tnn[validation_subwrap], loss_name
                )
                chkpt_idx = np.argmin(nnv.tnn[validation_subwrap][loss_name][:])
            elif chkpt == "initial":
                chkpt_idx = 0
            else:
                raise ValueError
            chkpt_params = torch.load(nnv.tnn.chkpts[f"chkpt_{chkpt_idx:05}"])
            for key, val in chkpt_params["network"].items():
                if key not in network_params:
                    network_params[key] = []
                network_params[key].append(val.cpu().numpy())
        for key, val in network_params.items():
            network_params[key] = np.array(val)
        return network_params

    def check_configs_are_equal(self):
        return all(
            [self[0].tnn.meta.spec == self[i].tnn.meta.spec for i in range(len(self))]
        )  # for j in range(len(self) - i)])

    def _mvn_parameter_sample(self, seed=0):
        """To return a multivariate normal parameter sample.

        Note, the values are not clamped to their respective ranges.
        """

        if isinstance(seed, str):
            # this assumes that the string is numeric and of the form
            # \d{4}.\d{3} (regex) or similar
            new_seed = int("".join(seed.split("/")))
            logging.info(
                f"converting string {seed} to integer seed {new_seed}"
                " for numpy random number generator to work"
            )
            seed = new_seed

        params_dict = self.parameters()

        # concatenate the parameter to an array of shape (samples, features)
        # to sample new parameters from multivariate gaussian respecting covariance
        params = np.concatenate(list(params_dict.values()), axis=1)

        # to obtain a new parameter sample
        mean = params.mean(axis=0)
        covariance = np.cov(params.T)

        randomstate = np.random.RandomState(seed)
        param_sample = randomstate.multivariate_normal(mean, covariance)

        # to map back to select the respective parameter type
        index = np.cumsum([0, *[v.shape[1] for v in params_dict.values()]])
        _param_dict = {}
        for param_name_idx, key in enumerate(params_dict.keys()):
            _param_dict[key] = param_sample[
                index[param_name_idx] : index[param_name_idx + 1]
            ]
        return _param_dict

    def get_biases(self, mode="best", validation_subwrap="original_validation_v2"):
        # TODO: check same parameter configs

        biases = self.parameters(chkpt=mode, validation_subwrap=validation_subwrap)[
            "nodes_bias"
        ]

        return self.node_types_unsorted, biases

    def get_time_constants(
        self, mode="best", validation_subwrap="original_validation_v2"
    ):
        # TODO: check same parameter configs
        time_constants = self.parameters(
            chkpt=mode, validation_subwrap=validation_subwrap
        )["nodes_time_const"]

        return self.node_types_unsorted, time_constants

    def _edge_tuples(
        self,
        target_sorted=False,
        groupby=["source_type", "target_type", "edge_type"],
    ):
        edges = self.ctome.edges.to_df()
        edges = edges.groupby(
            groupby,
            sort=False,
            as_index=False,
        ).first()
        edges = [(s, t) for s, t in edges[["source_type", "target_type"]].values]
        index = None
        if target_sorted:
            edges, index = utils.sort_edges_list([(e[1], e[0]) for e in edges])
            edges = np.array(edges)[:, ::-1]
        return edges, index

    def _param_keys(self, subwrap, param_name):
        # TODO: check all models have equal parameter configs otherwise raise err

        _configs = {"edges": "edge_config", "nodes": "node_config"}
        if subwrap not in _configs:
            raise ValueError(f"subwrap {subwrap}")

        param_config = self[0].tnn.spec.network[_configs[subwrap]][param_name]
        param = init_parameter(param_config, self.ctome[subwrap])
        return param.keys

    def get_syn_strength(
        self,
        target_sorted=False,
        mode="best",
        validation_subwrap="original_validation_v2",
    ):
        # TODO: check all models have equal parameter configs otherwise raise err
        syn_strength = self.parameters(
            chkpt=mode, validation_subwrap=validation_subwrap
        )["edges_syn_strength"]

        edges = self._param_keys("edges", "syn_strength")
        # edges, index = self._edge_tuples(
        #     target_sorted=target_sorted,
        #     groupby=["source_type", "target_type", "edge_type"],
        # )
        if target_sorted:
            edges, index = utils.sort_edges_list([(e[1], e[0]) for e in edges])
            edges = np.array(edges)[:, ::-1]
            syn_strength = syn_strength[:, index]
        return edges, syn_strength

    def get_signs(
        self,
        target_sorted=False,
        mode="best",
        validation_subwrap="original_validation_v2",
    ):
        # TODO: check same parameter configs
        signs = self.parameters(chkpt=mode, validation_subwrap=validation_subwrap)[
            "edges_sign"
        ]

        edges = self._param_keys("edges", "sign")

        # edges, index = self._edge_tuples(
        #     target_sorted=target_sorted,
        #     groupby=["source_type", "target_type"],
        # )

        if target_sorted:
            edges, index = utils.sort_edges_list([(e[1], e[0]) for e in edges])
            edges = np.array(edges)[:, ::-1]
            signs = signs[:, index]
        return edges, signs

    def get_syn_count(
        self,
        target_sorted=False,
        mode="best",
        validation_subwrap="original_validation_v2",
    ):
        # TODO: check same parameter configs
        syn_count = self.parameters(chkpt=mode, validation_subwrap=validation_subwrap)[
            "edges_syn_count"
        ]

        edges = self._param_keys("edges", "syn_count")

        # edges, index = self._edge_tuples(
        #     target_sorted=target_sorted,
        #     groupby=["source_type", "target_type", "du", "dv"],
        # )

        if target_sorted:
            edges, index = utils.sort_edges_list(
                [(e[1], e[0], e[2], e[3]) for e in edges]
            )
            # breakpoint()
            edges = [(e[1], e[0], e[2], e[3]) for e in edges]
            syn_count = syn_count[:, index]
        return edges, syn_count

    # def cross_val_loss(self):
    #     """
    #     Cross validation scores between best vals.
    #     """
    #     losses = {}
    #     mean_loss = 0
    #     for key, nnv in self.items():
    #         losses[key] = np.min(nnv.tnn.validation.loss[:])
    #         mean_loss += losses[key]
    #     mean_loss /= len(self)
    #     return mean_loss, losses

    # def compute_time_to_best_val(self):
    #     """
    #     Total time needed between first and last chkpt.
    #     """

    #     def get_tstamp_diff(first, last):
    #         fmt = "%a %b %d %H:%M:%S %Y"
    #         time_first = datetime.strptime(first, fmt)
    #         time_last = datetime.strptime(last, fmt)
    #         return round((time_last - time_first).seconds / (60 ** 2), 2)

    #     compute_time = {}
    #     mean_compute = 0

    #     for key, nnv in self.items():
    #         chkpts = sorted(nnv.tnn.chkpts.path.iterdir())
    #         chkpt_first = torch.load(chkpts[0], map_location="cpu")["time"]
    #         chkpt_last = torch.load(chkpts[-1], map_location="cpu")["time"]
    #         compute_time[key] = get_tstamp_diff(chkpt_first, chkpt_last)
    #         mean_compute += compute_time[key]

    #     mean_compute /= len(self)
    #     return mean_compute, compute_time

    def compute_time_trained(self):
        """
        Total time needed to train n_iters.
        """
        compute_time = {}
        mean_compute = 0

        for key, nnv in self.items():
            try:
                compute_time[key] = nnv.tnn.time_trained[()]
                mean_compute += compute_time[key]
            except AssertionError as e:
                logging.warning(f"Ignoring error for {key}. {e}")

        mean_compute /= len(self)
        return mean_compute, compute_time

    # def dt_last(self):
    #     """
    #     dt of last chckpt.
    #     """

    #     def get_dt(tnn):
    #         n_iterations = tnn.spec.task.n_iters
    #         config = tnn.spec.scheduler
    #         _function = (
    #             getattr(scheduler.HyperParamScheduler, config.pop("function"))
    #             if "function" in config
    #             else scheduler.HyperParamScheduler.stepwise
    #         )
    #         dec_array = _function(
    #             n_iterations, config.dt.start, config.dt.stop, config.dt.steps
    #         )
    #         chkpts = sorted(tnn.chkpts.path.iterdir())
    #         chkpt_last = torch.load(chkpts[-1], map_location="cpu")["t"]
    #         return dec_array[chkpt_last]

    #     dts = {}
    #     mean_dt = 0
    #     for key, nnv in self.items():
    #         dts[key] = get_dt(nnv.tnn)
    #         mean_dt += dts[key]

    #     mean_dt /= len(self)

    #     return mean_dt, dts

    # def always_dead(self):
    #     """
    #     True if neuron's activity is always negative across the entire
    #     validation set.
    #     """
    #     ad = {}
    #     for key, nnv in self.items():
    #         ad[key] = always_dead_questionmark(nnv.tnn)
    #     return ad

    # def num_always_dead(self):
    #     ad = self.always_dead()
    #     num_ad = {}
    #     total = 0
    #     for key, ad_per_node_type in ad.items():
    #         _num = sum([1 for v in ad_per_node_type.values() if v is True])
    #         num_ad[key] = _num
    #         total += _num
    #     return total, num_ad

    # def filter_statistics(self):
    #     if self._edge_stats:
    #         return self._edge_stats
    #     means, coef_of_var, _ = get_rf_statistics(self.names)
    #     self._edge_stats = (means, coef_of_var)
    #     return self._edge_stats

    @contextmanager
    def sort(
        self,
        mode="min",
        reverse=False,
        validation_subwrap="original_validation_v2",
        loss_name="epe",
    ):
        """
        To temporarily sort the ensemble based on a type of task error. In
        ascending order.

        This method sorts the self.names attribute temporarily, which serve as
        references to the Datawrap instances of the trained Networks.

        Overriding Ensemble's sort method to deal with NetworkViews objects.
        """
        if mode not in ["name", "min", "last", "mean"]:
            raise ValueError(
                f"mode {mode} must be one of 'name', 'min', 'last', 'mean'."
            )
        _names = deepcopy(self.names)
        _initialized = deepcopy(self._initialized)
        loss_name = dvs.analysis.validation_error._check_loss_name(
            self[0].tnn[validation_subwrap], loss_name
        )
        try:
            if mode == "name":
                self.names = sorted(self.names)
            elif mode == "min":
                self.names = sorted(
                    self.keys(),
                    key=lambda key: self[key]
                    .tnn[validation_subwrap][loss_name][:]
                    .min(),
                    reverse=reverse,
                )
            elif mode == "last":
                self.names = sorted(
                    self.keys(),
                    key=lambda key: self[key].tnn[validation_subwrap][loss_name][-1],
                    reverse=reverse,
                )
            elif mode == "mean":
                self.names = sorted(
                    self.keys(),
                    key=lambda key: self[key]
                    .tnn[validation_subwrap][loss_name][:]
                    .mean(),
                    reverse=reverse,
                )
            else:
                raise ValueError
        except Exception as e:
            logging.info(f"sorting failed: {e}")
        try:
            yield
        finally:
            self.names = _names
            self._initialized = _initialized

    def argsort(
        self,
        mode="min",
        reverse=False,
        validation_subwrap="original_validation_v2",
        loss_name="epe",
    ):
        """To return indices for sorting the ensemble based on types of
        task_errors.

        This is useful e.g. if an array is of shape [#models of ensemble, ...]
        and requires sorting.
        """
        if mode not in ["name", "min", "last", "mean"]:
            raise ValueError(
                f"mode {mode} must be one of 'name', 'min', 'last', 'mean'."
            )
        loss_name = dvs.analysis.validation_error._check_loss_name(
            self[0].tnn[validation_subwrap], loss_name
        )
        try:
            if mode == "name":
                return np.argsort(self.names)
            elif mode == "min":
                return np.argsort(
                    [
                        self[name].tnn[validation_subwrap][loss_name][:].min()
                        for name in self.names
                    ]
                )
            elif mode == "last":
                return np.argsort(
                    [
                        self[name].tnn[validation_subwrap][loss_name][-1]
                        for name in self.names
                    ]
                )
            elif mode == "mean":
                return np.argsort(
                    [
                        self[name].tnn[validation_subwrap][loss_name][:].mean()
                        for name in self.names
                    ]
                )

            else:
                raise ValueError
        except Exception as e:
            logging.info(f"sorting failed: {e}")

    def fri_correlations(self, flash_subwrap):
        known_preferred_contrasts = groundtruth_utils.known_preferred_contrasts
        known_cell_types = list(known_preferred_contrasts.keys())
        groundtruth = list(known_preferred_contrasts.values())

        fris, node_types = self.fris(6, subwrap=flash_subwrap)

        index = np.array(
            [
                np.where(nt == node_types)[0].item()
                for i, nt in enumerate(known_cell_types)
            ]
        )

        fris_for_known = fris[:, index]

        corr_fri, _ = analysis.simple_correlation.correlation(
            groundtruth, fris_for_known
        )

        return corr_fri

    def dsi_correlations(self, movingedge_subwrap):
        motion_tuning = groundtruth_utils.motion_tuning
        # no_motion_tuning = groundtruth_utils.no_motion_tuning
        known_dsi_types = groundtruth_utils.known_dsi_types
        dsis, node_types = self.dsis(
            subwrap=movingedge_subwrap,
            intensity=[0, 1],
            average="false",
            reshape=False,
        )
        known_dsi_types_index = np.array(
            [np.where(node_types == nt)[0].item() for nt in known_dsi_types]
        )
        dsis_for_known = np.take(dsis, known_dsi_types_index, axis=-1)
        dsis_for_known = np.median(dsis_for_known, axis=(1, 2, 3, 4))
        groundtruth_mt = np.array(
            [True if nt in motion_tuning else False for nt in known_dsi_types]
        )
        corr_dsi, _ = analysis.simple_correlation.correlation(
            groundtruth_mt, dsis_for_known
        )
        return corr_dsi

    def fri_task_error_correlation(
        self,
        validation_subwrap,
        validation_loss_fn,
        flash_subwrap,
        normalize=False,
        average_min_lower_error=None,
        average_min_upper_error=None,
    ):

        task_error = self.task_error(
            validation_subwrap=validation_subwrap,
            loss_name=validation_loss_fn,
            normalize=normalize,
            lower_error_bound=average_min_lower_error,
            upper_error_bound=average_min_upper_error,
        )
        task_error_values = task_error.values

        X = task_error_values
        Y = self.fri_correlations(flash_subwrap)
        linear_model = analysis.simple_correlation.linear_model(
            X, Y, alternative="less"
        )

        return analysis.simple_correlation.MetricTaskErrorCorrelation(
            X, Y, linear_model
        )

    def dsi_task_error_correlation(
        self,
        validation_subwrap,
        validation_loss_fn,
        movingedge_subwrap,
        normalize=False,
        average_min_lower_error=None,
        average_min_upper_error=None,
    ):

        task_error = self.task_error(
            validation_subwrap=validation_subwrap,
            loss_name=validation_loss_fn,
            normalize=normalize,
            lower_error_bound=average_min_lower_error,
            upper_error_bound=average_min_upper_error,
        )
        task_error_values = task_error.values

        X = task_error_values
        Y = self.dsi_correlations(movingedge_subwrap)
        linear_model = analysis.simple_correlation.linear_model(
            X, Y, alternative="less"
        )

        return analysis.simple_correlation.MetricTaskErrorCorrelation(
            X, Y, linear_model
        )

    def argsort_by_fri_correlation(self, flash_subwrap):
        return np.argsort(self.fri_correlations(flash_subwrap))[::-1]

    def argsort_by_dsi_correlation(self, movingedge_subwrap):
        return np.argsort(self.dsi_correlations(movingedge_subwrap))[::-1]

    def best_models_mask(
        self,
        best,
        mode="min",
        validation_subwrap="original_validation_v2",
        loss_name="epe",
    ):
        """To return a Boolean mask for the ensembles or arrays derived thereof,
        for a ratio of best models according to a type of task error."""
        argsort = self.argsort(
            mode=mode,
            validation_subwrap=validation_subwrap,
            loss_name=loss_name,
        )
        best = argsort[: int(best * len(self))]
        mask = np.zeros(len(self), dtype=bool)
        mask[[i for i in range(len(mask)) if i in best]] = True
        return mask

    def super_threshold_ratio_violins(self, stimulus_subwrap="augmented_sintel"):
        """To visualize the ratio of cell type activity above threshold to
        identify dead neurons."""
        sup_thr = np.array(
            [
                nnv.super_threshold_ratio(subwrap=stimulus_subwrap)
                for nnv in self.values()
            ]
        )
        return plots.violins(
            sup_thr.reshape(-1, sup_thr.shape[-1]),
            xticklabels=self.node_types_sorted,
        )

    # -- returning peak responses to types of stimuli to develop a motion
    # selectivity index that is independent of the direction selectivity index
    def _peak_responses(
        self,
        subwrap="movingsquare",
        flicker_subwrap="square_flicker",
        nonlinearity=False,
    ):
        if self._initialized["movingbar"] != subwrap:
            self.init_movingbar(subwrap=subwrap, device="cpu")
        if self._initialized["rectangular_flicker"] != subwrap:
            self.init_rectangular_flicker(subwrap=flicker_subwrap, device="cpu")

        moving = []
        flicker = []
        for name, nnv in self.items():
            r, q = nnv._peak_responses_for_msi_from_flicker()
            moving.append(r)
            flicker.append(q)

        moving = np.array(moving)
        flicker = np.array(flicker)
        return moving, flicker

    def _peak_responses_edges(
        self,
        subwrap="movingedge_chkpt_best_v4",
        flicker_subwrap="edge_flicker",
        nonlinearity=False,
    ):
        if self._initialized["movingbar"] != subwrap:
            self.init_movingbar(subwrap=subwrap, device="cpu")
        if self._initialized["normalizing_movingbar"] != subwrap:
            self.init_normalizing_movingbar(subwrap=flicker_subwrap, device="cpu")

        moving = []
        flicker = []
        for name, nnv in self.items():
            r, q = nnv._peak_responses_edges_for_msi_from_flicker()
            moving.append(r)
            flicker.append(q)

        moving = np.array(moving)
        flicker = np.array(flicker)
        return moving, flicker

    def _peak_responses_edges_plus_naturalistic(
        self,
        subwrap="movingedge_chkpt_best_v4",
        flicker_subwrap="edge_flicker",
        naturalistic_subwrap="augmented_sintel",
        nonlinearity=False,
    ):
        self.init_movingbar(subwrap=subwrap, device="cpu")
        self.init_normalizing_movingbar(subwrap=flicker_subwrap, device="cpu")
        self.init_augmented_sintel(subwrap=naturalistic_subwrap, device="cpu")

        moving = []
        flicker = []
        naturalistic = []
        for name, nnv in self.items():
            r, q, s = nnv._peak_responses_edges_plus_naturalistic(
                nonlinearity=nonlinearity
            )
            moving.append(r)
            flicker.append(q)
            naturalistic.append(s)

        moving = np.array(moving)
        flicker = np.array(flicker)
        naturalistic = np.array(naturalistic)
        return moving, flicker, naturalistic

    def peak_responses(self, stimulus_subwrap="augmented_sintel"):
        """To return peak responses to a type of stimulus in shape of (#models,
        #cell types)."""
        # (#models, #samples, #frames, #nodes)
        responses = np.array(
            [
                nnv.tnn[stimulus_subwrap].network_states.nodes.activity_central[:]
                for nnv in self.values()
            ]
        )
        peak_responses = np.nanmax(responses, axis=2)
        return peak_responses

    def peak_response_violins(self, stimulus_subwrap="augmented_sintel"):
        """To visualize peak responses to a type of stimulus in violins across
        models (marginals)."""
        peak_responses = self.peak_responses(stimulus_subwrap=stimulus_subwrap)
        return plots.violins(
            peak_responses.reshape(-1, peak_responses.shape[-1]),
            xticklabels=self.node_types_sorted,
        )

    def peak_response_violins_groups(
        self,
        stimulus_subwraps=[
            "movingedge_chkpt_best_v4",
            "edge_flicker",
            "augmented_sintel",
        ],
        legend_labels=None,
    ):
        """To visualize peak responses to multiple types of stimuli in grouped
        violins across models (marginals)."""
        raise NotImplementedError
        # TODO: different numbers of samples! won't reshape
        peak_responses = np.array(
            [
                self.peak_responses(stimulus_subwrap=subwrap)
                for subwrap in stimulus_subwraps
            ]
        )
        return plots.violins(
            peak_responses.reshape(
                len(stimulus_subwraps), -1, peak_responses.shape[-1]
            ),
            xticklabels=self.node_types_sorted,
            cmap=plt.cm.tab10,
            cstart=0,
            cdist=1,
            legend=legend_labels or stimulus_subwraps,
            legend_kwargs=dict(bbox_to_anchor=[1, 1.4], fontsize=6),
        )

    def msis(
        self,
        version,
        subwrap="movingsquare",
        flicker_subwrap="square_flicker",
        naturalistic_subwrap="augmented_sintel",
        nonlinearity=False,
    ):
        """To return tentative motion selectivity indices."""
        if self._initialized["movingbar"] != subwrap:
            self.init_movingbar(subwrap=subwrap, device="cpu")
        if self._initialized["rectangular_flicker"] != subwrap:
            self.init_rectangular_flicker(subwrap=flicker_subwrap, device="cpu")
        if version == "v3":
            self.init_augmented_sintel(subwrap=naturalistic_subwrap, device="cpu")

        msis = []
        for name, nnv in self.items():
            if version == "v1":
                node_type, msi = nnv.msi_v1(nonlinearity=nonlinearity)
            elif version == "v2":
                node_type, msi = nnv.msi_v2(nonlinearity=nonlinearity)
            elif version == "v3":
                node_type, msi = nnv.msi_v3(nonlinearity=nonlinearity)
            elif version == "v4":
                node_type, msi = nnv.msi_v4(nonlinearity=nonlinearity)
            msis.append(msi)
        msis = np.array(msis)
        return msis, np.array(node_type)

    def dsis(
        self,
        subwrap,
        intensity,
        pre_stim=False,  # debugging flag
        post_stim=False,  # debugging flag
        subtract_baseline=False,
        nonlinearity=True,
        average="true",
        reshape=True,
    ):
        if self._initialized["movingbar"] != subwrap:
            self.init_movingbar(subwrap=subwrap, device="cpu")

        if average == "true":
            dsis = []
            for name, nnv in self.items():
                cell_types, dsi, theta_pref = nnv.movingbar.dsi(
                    None,
                    intensity=intensity,
                    pre_stim=pre_stim,
                    post_stim=post_stim,
                    subtract_baseline=subtract_baseline,
                    nonlinearity=nonlinearity,
                )
                dsis.append(dsi)
            dsis = np.array(dsis)

        elif average == "false":
            intensity = (
                [intensity] if not isinstance(intensity, Iterable) else intensity
            )
            peak_responses = self.moving_edge_peak_responses(subwrap)

            intensity_index = [
                np.where(_intensity == peak_responses.intensities)[0].item()
                for _intensity in intensity
            ]
            dsis = peak_responses.dsi_from_peak_angular_responses(average=False)[
                :, :, :, intensity_index
            ]
            cell_types = self.node_types_sorted
            if reshape:
                dsis = np.reshape(dsis, (-1, len(cell_types)))
        elif average == "v2":
            intensity = (
                [intensity] if not isinstance(intensity, Iterable) else intensity
            )
            peak_responses = self.moving_edge_peak_responses(subwrap)

            intensity_index = [
                np.where(_intensity == peak_responses.intensities)[0].item()
                for _intensity in intensity
            ]
            dsis = peak_responses.dsi_from_peak_angular_responses(average=True)[
                :, intensity_index
            ].squeeze()
            cell_types = self.node_types_sorted
            # dsis = np.reshape(dsis, (-1, len(cell_types)))

        return dsis, np.array(cell_types)

    def moving_edge_peak_responses(
        self,
        subwrap,
    ) -> EnsembleMovingEdgeResponses:
        if self._initialized["movingbar"] != subwrap:
            self.init_movingbar(subwrap=subwrap, device="cpu")

        peak_response0 = self[0].movingbar.peak_response()[0]
        (
            n_angles,
            n_widths,
            n_intensities,
            n_speeds,
            n_cell_types,
        ) = peak_response0.shape
        n_models = len(self)
        peak_responses = np.zeros(
            [
                n_models,
                n_angles,
                n_widths,
                n_intensities,
                n_speeds,
                n_cell_types,
            ]
        )
        for i, nnv in enumerate(tqdm(self.values())):
            if i == 0:
                peak_responses[i] = peak_response0
            else:
                peak_responses[i] = nnv.movingbar.peak_response()[0]
        (
            angles,
            widths,
            intensities,
            speeds,
        ) = nnv.movingbar.stimulus_parameters()

        return EnsembleMovingEdgeResponses(
            peak_responses,
            angles,
            widths,
            intensities,
            speeds,
            subwrap,
            best_ratio=self._best_ratio,
            worst_ratio=self._worst_ratio,
            responses=None,
        )

    def preferred_directions(
        self,
        subwrap,
        intensity,
        pre_stim=False,  # debugging flag
        post_stim=False,  # debugging flag
        subtract_baseline=False,
    ):
        """To return preferred directions across the ensemble."""
        if self._initialized["movingbar"] != subwrap:
            self.init_movingbar(subwrap=subwrap, device="cpu")

        preferred_directions = []
        for name, nnv in self.items():
            node_type, dsi, theta_pref = nnv.movingbar.dsi(
                None,
                intensity=intensity,
                pre_stim=pre_stim,
                post_stim=post_stim,
                subtract_baseline=subtract_baseline,
            )
            preferred_directions.append(theta_pref)
        preferred_directions = np.array(preferred_directions)
        return preferred_directions, np.array(node_type)

    def fris(
        self,
        radius,
        mode="convention?",
        subwrap="flashes",
        subtract_baseline=False,
        nonlinearity=False,
        nonnegative=True,
    ):

        if self._initialized["flashes"] != subwrap:
            self.init_flashes(subwrap=subwrap)
        fris = []

        for name, nnv in self.items():
            fri, fri_dict = nnv.flashes_dataset.flash_response_index(
                radius,
                mode,
                subtract_baseline=subtract_baseline,
                nonlinearity=nonlinearity,
                nonnegative=nonnegative,
            )
            fris.append(fri)
        fris = np.array(fris)
        return fris, np.array([nt for nt in fri_dict.keys()])

    def osis(self, subwrap, intensity, nonlinearity=True):
        if self._initialized["oriented_bar"] != subwrap:
            self.init_oriented_bar(subwrap=subwrap, device="cpu")

        osis = []
        for name, nnv in self.items():
            node_type, osi, theta_pref = nnv.oriented_bar.osi(
                None, intensity=intensity, nonlinearity=nonlinearity
            )
            osis.append(osi)
        osis = np.array(osis)
        return osis, np.array(node_type)

    def _best_checkpoints(
        self,
        validation_subwrap="original_validation_v2",
        loss_name="epe",
    ):
        loss_name = dvs.analysis.validation_error._check_loss_name(
            self[0].tnn[validation_subwrap], loss_name
        )
        best_chkpts = [
            np.argmin(nnv.tnn[validation_subwrap][loss_name][:])
            for nnv in self.values()
        ]

        return best_chkpts

    def _task_error(
        self,
        validation_subwrap="original_validation_v2",
        normalize=True,
        loss_name="epe",
        lower_error_bound=LOWER_ERROR_BOUND,
        upper_error_bound=UPPER_ERROR_BOUND,
    ):
        """To return the minimal error in shape (#models) with optional
        normalization."""

        loss_name = dvs.analysis.validation_error._check_loss_name(
            self[0].tnn[validation_subwrap], loss_name
        )
        error = np.array(
            [nnv.tnn[validation_subwrap][loss_name][:].min() for nnv in self.values()]
        )

        if normalize:

            if lower_error_bound is None:
                lower_error_bound = 0
            if upper_error_bound is None:
                upper_error_bound = 1

            error = (error - lower_error_bound) / (
                upper_error_bound - lower_error_bound
            )
        # avg_loss0 = np.array( [nnv.tnn[subwrap].loss[0] for nnv in
        #     self.values()] ).mean() normed_error = losses / avg_loss0 if
        #     minmaxscale: normed_error = (normed_error - normed_error.min()) /
        #     (normed_error.max() - normed_error.min())
        return error

    def task_error(
        self,
        validation_subwrap="original_validation_v2",
        normalize=False,
        loss_name="epe",
        lower_error_bound=LOWER_ERROR_BOUND,
        upper_error_bound=UPPER_ERROR_BOUND,
        cmap="Blues_r",
        truncate=None,
        vmin=None,
        vmax=None,
    ):

        error = self._task_error(
            validation_subwrap,
            normalize,
            loss_name,
            lower_error_bound,
            upper_error_bound,
        )

        if truncate is None:
            # truncate because the maxval would be white with the default colormap
            # which would be invisible on a white background
            truncate = {"minval": 0.0, "maxval": 0.9, "n": 256}
        cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        cmap = plt_utils.truncate_colormap(cmap, **truncate)
        sm, norm = plt_utils.get_scalarmapper(
            cmap=cmap,
            vmin=vmin or np.min(error),
            vmax=vmax or np.max(error),
        )
        colors = sm.to_rgba(np.array(error))

        return TaskError(error, colors, cmap, norm, sm)

    def best_index(self, validation_subwrap="original_validation_v2", loss_name="epe"):
        error = self.task_error(
            validation_subwrap=validation_subwrap, loss_name=loss_name
        )
        return np.argmin(error.values)

    def best_color(self, validation_subwrap="original_validation_v2", loss_name="epe"):
        task_error = self.task_error(
            validation_subwrap=validation_subwrap, loss_name=loss_name
        )
        return task_error.colors[self.best_index(validation_subwrap, loss_name)]

    def sintel_responses_no_nans(self, nonlinearity=False, dt=1 / 100):
        data = dvs.datasets.sintel.AugmentedPaddedSintelLum(
            path="/groups/turaga/home/lappalainenj/FlyVis/dvs-sim/data/SintelDataSet",
            tasks=["lum"],
            n_frames=19,
            sample_all=True,
            dt_sampling=True,
            boxfilter=dict(extent=15, kernel_size=13),
            vertical_splits=3,
            p_flip=0.0,
            p_rot=0.0,
            contrast=0.0,
            brightness=0.0,
            noise=0.00,
            cache="gpu",
            gamma=1,
        )
        start_frame = int(0.25 / dt)
        end_frames = np.ceil(((data.arg_df.frames.values / 24) / dt)).astype(int)
        n_ensemble = len(self)
        n_features = (end_frames - start_frame).sum().astype(int)

        node_index = dvs.utils.NodeIndexer(self.ctome)

        node_types = self.node_types
        responses_dict = {nt: [] for nt in node_types}
        for key, nnv in self.items():
            responses = nnv.tnn.augmented_sintel.network_states.nodes.activity_central[
                :
            ]

            if nonlinearity:
                responses = np.maximum(responses, 0)

            for node_type in node_types:
                _responses = []
                _index = node_index[node_type]
                for j, sample in enumerate(responses):
                    _responses.append(sample[start_frame : end_frames[j], _index])

                responses_dict[node_type].append(np.concatenate(_responses))

        responses_dict = {key: np.array(value) for key, value in responses_dict.items()}
        return responses_dict

    def flash_response_data(self, node_type, z_score=False, timelim=(1, 2), stack=True):
        fr_on = []
        fr_off = []
        for nnv in self.values():
            fr = nnv.flashes_dataset.response(
                node_type=node_type, intensity=None, radius=6
            )
            fr_off.append(fr[0])
            fr_on.append(fr[1])

        fr_off = np.array(fr_off)
        fr_on = np.array(fr_on)
        time = (
            np.arange(fr_on.shape[1]) * nnv.flashes_dataset.dt
            - nnv.flashes_dataset.t_pre
        )

        if z_score:
            fr_off = (fr_off - fr_off.mean(axis=1, keepdims=True)) / (
                fr_off.std(axis=1, keepdims=True) + 1e-15
            )
            fr_on = (fr_on - fr_on.mean(axis=1, keepdims=True)) / (
                fr_on.std(axis=1, keepdims=True) + 1e-15
            )

        mask = (time >= timelim[0]) & (time < timelim[1])

        if stack:
            return time, np.stack((fr_off, fr_on), axis=2), mask
        return time, fr_off, fr_on, mask

    def umap_embedding(self, *args, **kwargs):
        return self.cluster_embedding(*args, **kwargs)

    def cluster_embedding(
        self,
        cell_type: Union[str, Iterable],
        cell_type_agnostic=False,
        naturalistic_responses_wrap="augmented_sintel",
        embedding_method=None,
        embedding_kwargs=None,
        range_n_clusters=None,
        gm_n_init=1,
        gm_max_iter=1000,
        gm_random_state=0,
        figsize=[0.94, 2.38],
        validation_subwrap="original_validation_v2",
        validation_loss_fn="epe",
        colors=None,
        task_error_colors=True,
        color_mask=None,
        color_substitute=(0.53, 0.53, 0.53, 0.1),
        fontsize=5,
        plot_mode="paper",
        cbar=False,
        fig=None,
        ax=None,
        task_error_sort_mode="mean",
        normalize_task_error=False,
        title="models clustered by\n{}'s sintel responses",
        # embedding=None,
        # mask=None,
        # **kwargs,
        err_x_offset=0.025,
        err_y_offset=-0.025,
    ):
        cluster = self.cluster(
            cell_type,
            cell_type_agnostic,
            naturalistic_responses_wrap,
            embedding_method,
            embedding_kwargs,
            range_n_clusters,
            gm_n_init,
            gm_max_iter,
            gm_random_state,
        )  # type: clustering.GaussianMixtureClustering

        if not cell_type_agnostic and task_error_colors and colors is None:
            task_error = self.task_error(
                validation_subwrap=validation_subwrap,
                loss_name=validation_loss_fn,
                normalize=normalize_task_error,
            )
            colors = task_error.colors
            cmap = task_error.cmap
            norm = task_error.norm
            task_error = task_error.values
        else:
            task_error = None
            cmap = None
            norm = None
        if cell_type_agnostic and colors is None:
            colors = plt_utils.get_colors(len(cell_type))
            colors = np.array(colors).repeat(len(self), axis=0)

        if colors is not None and color_mask is not None:
            colors[color_mask] = color_substitute

        embeddingplot = cluster.plot(
            task_error,
            colors,
            annotate=True,
            fig=fig,
            ax=ax,
            figsize=figsize,
            plot_mode=plot_mode,
            fontsize=fontsize,
            task_error_sort_mode=task_error_sort_mode,
            err_x_offset=err_x_offset,
            err_y_offset=err_y_offset,
        )
        embeddingplot.cmap = cmap
        embeddingplot.norm = norm

        if plot_mode != "paper":
            embeddingplot.fig.suptitle(
                title.format(cell_type),
                y=embeddingplot.ax.get_position().extents[-1],
                fontsize=5,
                va="bottom",
            )
        if task_error_colors and cbar:
            plots.plt_utils.add_colorbar(
                embeddingplot.fig,
                embeddingplot.ax,
                cmap=embeddingplot.cmap,
                norm=embeddingplot.norm,
                fontsize=5,
                tick_width=0.25,
                tick_length=1,
                label="task error",
            )
        return embeddingplot

    # def cluster_indices(
    #     self,
    #     node_type,
    #     model_mask=None,
    #     cluster_subwrap="umap",
    #     validation_subwrap="original_validation_v2",
    #     embedding=None,
    #     mask=None,
    # ):
    #     """To return references for clustering the ensemble while maintaining
    #     state.

    #     Returns: Dict[str, Array[int]]

    #     Note: requires arg 'validation_subwrap' to sort the label_ids according
    #     to the cluster average performance. We sort the labels such that the
    #     best performing cluster has label 0, the second best label 1 and so on.
    #     """

    #     if embedding is None and mask is None:
    #         umaps = Datawrap(self.path / cluster_subwrap)
    #         if not is_up_to_date(umaps.path, datetime(2021, 12, 13)):
    #             raise AssertionError(
    #                 "clustering not up to date. labels are not stored as performance sorted anymore. recompute clustering"
    #             )

    #         try:
    #             labels = umaps.sintel_responses_pre_nonlinearity.labels[
    #                 node_type
    #             ][:].astype(int)
    #             mask = umaps.sintel_responses_pre_nonlinearity.masks[node_type][
    #                 :
    #             ]
    #         except:
    #             print(
    #                 f"stored clusters not found for {node_type} - returning single cluster"
    #             )
    #             mask = np.ones(len(self), dtype=bool)
    #             labels = np.zeros(len(self), dtype=int)
    #     elif embedding is None or mask is None:
    #         raise ValueError("specify embedding and mask")
    #     else:
    #         labels, _, _ = clustering.gaussian_mixture(
    #             embedding,
    #             mask,
    #             range_n_clusters=[1, 2, 3, 4, 5],
    #             n_init=1,
    #             max_iter=1000,
    #         )

    #     if model_mask is not None:
    #         mask = mask & model_mask

    #     if self._model_mask is not None:
    #         mask = mask & self._model_mask

    #     return self._cluster_indices(labels, mask, validation_subwrap)

    def embedding(
        self,
        cell_type: Union[str, Iterable],
        cell_type_agnostic=False,
        naturalistic_responses_wrap="augmented_sintel",
        embedding_method=None,
        embedding_kwargs=None,
    ) -> clustering.Embedding:
        self.init_naturalistic_responses(subwrap=naturalistic_responses_wrap)
        return self.naturalistic_responses.embedding(
            cell_type, embedding_method, embedding_kwargs, cell_type_agnostic
        )

    def cluster(
        self,
        cell_type: Union[str, Iterable],
        cell_type_agnostic=False,
        naturalistic_responses_wrap="augmented_sintel",
        embedding_method=None,
        embedding_kwargs=None,
        range_n_clusters=None,
        gm_n_init=1,
        gm_max_iter=1000,
        gm_random_state=0,
    ) -> clustering.GaussianMixtureClustering:
        embedding = self.embedding(
            cell_type,
            cell_type_agnostic,
            naturalistic_responses_wrap,
            embedding_method,
            embedding_kwargs,
        )
        return embedding.cluster.gaussian_mixture(
            range_n_clusters, gm_n_init, gm_max_iter, gm_random_state
        )

    @dvs.utils.cache_utils.cache
    def cluster_indices(
        self,
        cell_type: Union[str, Iterable],
        cell_type_agnostic=False,
        naturalistic_responses_wrap="augmented_sintel",
        embedding_method=None,
        embedding_kwargs=None,
        range_n_clusters=None,
        gm_n_init=1,
        gm_max_iter=1000,
        gm_random_state=0,
        validation_subwrap="original_validation_v2",
        loss_name="epe",
        task_error_sort_mode="mean"
        # embedding=None,
        # model_mask=None,
        # cluster_subwrap="umap",
        # mask=None,
    ):

        cluster = self.cluster(
            cell_type,
            cell_type_agnostic,
            naturalistic_responses_wrap,
            embedding_method,
            embedding_kwargs,
            range_n_clusters,
            gm_n_init,
            gm_max_iter,
            gm_random_state,
        )
        task_error = self.task_error(
            validation_subwrap=validation_subwrap,
            loss_name=loss_name,
            normalize=False,
        )
        cluster.task_error_sort_labels(task_error.values, mode=task_error_sort_mode)
        return self._cluster_indices(
            cluster.labels,
            cluster.embedding.mask,
        )

    def _cluster_indices(self, labels, mask):

        # to remove labels that are invalid
        labels = labels[mask]
        # to remove models from the index that are invalid
        indices = np.arange(len(self))[mask]

        return {label_id: indices[labels == label_id] for label_id in np.unique(labels)}


class EnsembleViews(EnsembleEval):
    def __init__(self, tnns):
        super().__init__(tnns)
        self.ctome = self[next(iter(self))].ctome
        self.node_types_unsorted = self.ctome.unique_node_types[:].astype(str)
        self.node_types_sorted, self.node_index = utils.order_nodes_list(
            self.node_types_unsorted
        )
        self.node_types = self.node_types_sorted
        self.output_node_types = self.ctome.output_node_types[:].astype(str)
        self.input_node_types = self.ctome.input_node_types[:].astype(str)
        self._initialized = dict(
            movingbar="",
            dots="",
            augmented_sintel="",
            flashes="",
            oriented_bar="",
            rectangular_flicker="",
            normalizing_movingbar="",
            naturalistic_response_cluster="",
            naturalistic_responses="",
        )
        self._best_ratio = 0.5
        self._worst_ratio = 0.5

    #  -- Context manager to allow to alter the names attribute and then
    # selectively compute metrics only for the names in the context. This way no
    # need to create new EnsembleView objects which requires reinitializing
    # NetworkView instances and stimulus dataset instances.
    @contextmanager
    def model_items(self, key=None):
        """To filter the ensemble temporarily by a list of items (int, slice,
        list, array) while maintaining state of the Ensemble instance."""
        # no-op
        if key is None:
            yield
            return
        _names = tuple(self.names)
        _initialized = deepcopy(self._initialized)

        if isinstance(key, (int, np.integer, slice)):
            _context_names = self.names[key]
        elif isinstance(key, (list, np.ndarray)):
            if np.array(key).dtype == np.array(self.names).dtype:
                _context_names = key
            elif np.array(key).dtype == np.int64:
                _context_names = np.array(self.names)[key]
            else:
                raise ValueError(f"{key}")
        else:
            raise ValueError(f"{key}")
        self.names = _context_names
        self._model_index = np.array(
            [i for i, name in enumerate(_names) if name in self.names]
        )
        self._model_mask = np.zeros(len(_names), dtype=bool)
        self._model_mask[self._model_index] = True
        try:
            yield
        finally:
            self.names = list(_names)
            self._initialized = _initialized
            self._model_mask = np.ones(len(self)).astype(bool)
            self._model_index = np.arange(len(self))

    def _recover_names(self):
        # to reset names if any of the context managers failed and corrupted the
        # names
        self.names, _ = tnn_path_names(self.tnn_paths)

    @contextmanager
    def model_ratio(
        self,
        best=None,
        worst=None,
        validation_subwrap="original_validation_v2",
        validation_loss_fn="epe",
        sort_mode="min",
    ):
        """To sort and filter the ensemble temporarily by a ratio of models that
        are performing good or bad based on a type of task error."""
        # no-op
        if best is None and worst is None:
            yield
            return

        _names = tuple(self.names)
        # because even if new stimuli are initialized, these are only for a
        # subset of the models. but then these are not necessarily true anymore.
        # needs to track which field changes and null it.
        _initialized = deepcopy(self._initialized)

        with self.sort(
            sort_mode,
            validation_subwrap=validation_subwrap,
            loss_name=validation_loss_fn,
        ):
            if best is not None and worst is not None and best + worst > 1:
                raise ValueError("best and worst must add up to 1")

            if best is not None or worst is not None:
                _context_best_names, _context_worst_names = [], []
                if best is not None:
                    _context_best_names = list(self.names[: int(best * len(self))])
                    self._best_ratio = best
                else:
                    self._best_ratio = 0
                if worst is not None:
                    _context_worst_names = list(self.names[-int(worst * len(self)) :])
                    self._worst_ratio = worst
                else:
                    self._worst_ratio = 0
                self.names = [*_context_best_names, *_context_worst_names]

                if self.names:  # to prevent an empty index
                    self._model_index = np.array(
                        [i for i, name in enumerate(_names) if name in self.names]
                    )
                    self._model_mask = np.zeros(len(_names), dtype=bool)
                    self._model_mask[self._model_index] = True
            try:
                yield
            finally:
                self.names = list(_names)
                # a subset of models can initialize stimuli
                # responses of one type from different configs. this will
                # change self._initialized. if different models have differently
                # configured responses of the same type loaded, this would lead
                # to inconsistent analyses. this is to reset the initialization.
                for key in _initialized:
                    if _initialized[key] != self._initialized[key]:
                        _initialized[key] = ""
                self._initialized = _initialized
                self._model_mask = np.ones(len(self)).astype(bool)
                self._model_index = np.arange(len(self))
                self._best_ratio = 0.5
                self._worst_ratio = 0.5

    def _rm_unstable(self, threshold=100, soft_rm=True):
        """To pop unstable models from the ensemble.

        Note: this condition for unstable models is limited to a heuristic.
        """
        for name, nnv in self.items():
            best_chkpt = np.argmin(nnv.tnn.kfold_validation.loss[:])
            iteration_at_best_chkpt = nnv.tnn.chkpt_iter[:][best_chkpt]
            activity = nnv.tnn.activity[iteration_at_best_chkpt]
            if np.abs(activity) > threshold:
                if soft_rm:
                    self.names.remove(name)
                else:
                    self.pop(name)
                logging.info(f"removing {name} from analysis")

    @contextmanager
    def filter(
        self,
        items=None,
        best_ratio=None,
        worst_ratio=None,
        validation_subwrap="original_validation_v2",
        sort_mode="min",
    ):
        """Filter by items, best ratios, worst ratios.

        Can filter by items and ratios as long as there are
        """

        try:
            if items is None and best_ratio is None and worst_ratio is None:
                yield
                return
            elif items is None:
                with self.model_ratio(
                    best_ratio, worst_ratio, validation_subwrap, sort_mode
                ):
                    yield
            elif best_ratio is None and worst_ratio is None:
                with self.model_items(items):
                    yield
            else:
                with self.model_items(items):
                    with self.model_ratio(
                        best_ratio, worst_ratio, validation_subwrap, sort_mode
                    ):
                        yield
        finally:
            pass

    @contextmanager
    def filter_unstable(self, threshold=100):
        """To temporarily remove clearly unstable models from the ensemble.

        Note: the condition for unstable models is limited to a heuristic.
        """
        _names = deepcopy(self.names)
        self._rm_unstable(threshold=threshold, soft_rm=True)
        try:
            yield
        finally:
            self.names = _names

    def _min_losses(
        self,
        normalize=False,
        validation_subwrap="original_validation_v2",
        loss_name="epe",
    ):
        return self.task_error(
            validation_subwrap=validation_subwrap,
            normalize=normalize,
            loss_name=loss_name,
        )

    def _argmin_losses(
        self,
        normalize=False,
        validation_subwrap="original_validation_v2",
        loss_name="epe",
    ):
        """To return the index over chkpts or iterations where losses are
        minimal as array of shape (#models)."""

        argmin = np.array(
            [
                np.argmin(nnv.tnn[validation_subwrap][loss_name][:])
                for nnv in self.values()
            ]
        )
        return argmin

    #  -- Visualizations

    def loss_histogramm(
        self,
        normalize=False,
        other_ensemble: str = None,
        figsize=[1, 1],
        validation_subwrap="original_validation_v2",
        cmap=utils.color_utils.single_blue_cmap,
        other_cmap=utils.color_utils.single_orange_cmap,
        legend=True,
        legend_labels=None,
        bar_width=None,
        bins=None,
        losses=None,
        other_losses=None,
        bbox_legend=(0.8, 0.8),  # (x0, y0, width, height)
        legend_loc="lower left",  # edge to put at (x0, y0)
        edge_color="k",
        fontsize=5,
        fill=False,
        histtype="step",
        fig=None,
        ax=None,
        loss_name="epe",
        annotate=True,
        **kwargs,
    ):
        """To visualize the minimal task errors of one or two ensembles."""

        if other_ensemble is None:
            fig, ax = plt_utils.init_plot(
                figsize=figsize, fontsize=fontsize, fig=fig, ax=ax
            )
            losses = (
                losses
                if losses is not None
                else self.task_error(
                    normalize=normalize,
                    validation_subwrap=validation_subwrap,
                    loss_name=loss_name,
                ).values
            )
            _, _, patches0 = ax.hist(
                losses,
                bins=bins if bins is not None else len(self),
                # zorder=5, alpha=0.5, edgecolor=edge_color,
                linewidth=0.5,
                # rwidth=bar_width,
                fill=fill,
                histtype=histtype,
            )
            # for i, patch in enumerate(patches0):
            #     patch.set_facecolor(colors[i])

            ax.set_ylabel("number models", fontsize=fontsize)
            ax.set_xlabel(
                f"task error\n{validation_subwrap.replace('_', ' ')}",
                fontsize=fontsize,
            )
            if annotate:
                ax.annotate(
                    f"min:{losses.min():.3f}\nmedian:{np.median(losses):.3f}\nmean:{losses.mean():.3f}\nstd:{losses.std():.2f}",
                    xy=(1, 1),
                    ha="right",
                    va="top",
                    xycoords="axes fraction",
                    fontsize=fontsize,
                )
            return fig, ax

        elif other_ensemble is not None:
            # figsize = [max(figsize[0], 2), max(figsize[1], 2)]
            if other_ensemble == self.name:
                kwargs = vars()
                kwargs.update(other_ensemble=None)
                kwargs.pop("self")
                return self.loss_histogramm(**kwargs)
            losses = (
                losses
                if losses is not None
                else self.task_error(
                    normalize=normalize,
                    validation_subwrap=validation_subwrap,
                    loss_name=loss_name,
                ).values
            )

            other_ensemble = EnsembleViews(other_ensemble)

            other_losses = (
                other_losses
                if other_losses is not None
                else other_ensemble.task_error(
                    normalize=normalize,
                    validation_subwrap=validation_subwrap,
                    loss_name=loss_name,
                ).values
            )
            fig, axes = plots.histogram_grid(
                [other_losses, losses],
                labels=legend_labels
                if legend_labels is not None
                else [other_ensemble.name, self.name],
                share_lims=True,
                nbins=bins,
                annotate=annotate,
                fontsize=fontsize,
            )
            # # if len(other_losses) != len(losses):
            # #     raise NotImplementedError(
            # #         "TODO: implement comparison with unequally sized ensembles."
            # #     )
            # all_losses = np.stack((losses, other_losses), axis=1)
            # # all_losses = [losses, other_losses]

            # limits = plt_utils.get_lims((losses, other_losses), 0.1) fig, axes
            # = plots.pairplot( [all_losses, all_losses], upper="hist",
            # diag="hist", labels=legend_labels if legend_labels is not None
            # else [self.name, other_ensemble.name], limits=[limits, limits],
            # figsize=figsize, yhide_diag_no_lower=False,
            # )

            # all_losses = (losses, other_losses) for i, ax in
            # enumerate(np.diag(axes)): ax.tick_params(axis="both",
            # which="major", labelsize=fontsize)
            # ax.xaxis.label.set_size(fontsize) if i == 0: ax.set_ylabel("number
            # models", fontsize=fontsize) else: ax.set_yticks([])
            # ax.spines["left"].set_visible(False) ax.annotate(
            # f"min:{all_losses[i].min():.3G}\nmean:{all_losses[i].mean():.3G}\nstd:{all_losses[i].std():.2G}",
            # xy=(1, 1), ha="right", va="top", xycoords="axes fraction",
            # fontsize=fontsize,
            #     )
            # ax.yaxis.label.set_size(fontsize)
            fig.suptitle(
                f"task error\n{validation_subwrap.replace('_', ' ')}",
                y=0.96,
                fontsize=fontsize,
                va="bottom",
            )
            return fig, axes
            # _, _, patches = ax.hist( [losses, other_losses], bins=bins if bins
            #     is not None else len(self), zorder=0, alpha=0.5,
            #     edgecolor=edge_color, linewidth=0.1, rwidth=bar_width,
            #     fill=fill, histtype=histtype,
            # )
            # for i, patch in enumerate(patches[0]):
            #     patch.set_facecolor(colors[i]) for i, patch in
            #     enumerate(patches[1]): patch.set_facecolor(other_colors[i])

            # if legend: plt_utils.add_legend( ax=ax, labels=legend_labels if
            #     legend_labels is not None else [self.name,
            #     other_ensemble.name], colors=[ colors[len(colors) // 2],
            #     other_colors[len(other_colors) // 2],
            #         ],
            #         fontsize=4,
            #         handlelength=1,
            #         bbox_to_anchor=bbox_legend,
            #         loc=legend_loc,
            #         edgecolor="k",
            #         alpha=0.5,
            #         edgewidth=0.1,
            #     )
        else:
            raise ValueError

    def loss_histogramm_with_highlights(
        self,
        best=None,
        worst=None,
        items=None,
        fig=None,
        ax=None,
        bins=None,
        validation_subwrap="original_validation_v2",
        loss_name="epe",
    ):
        min_losses = self.task_error(
            normalize=True,
            validation_subwrap=validation_subwrap,
            loss_name=loss_name,
        ).values
        bins = bins or np.linspace(*plt_utils.get_lims(min_losses, 0.1), 50)
        fig, ax = self.loss_histogramm(
            normalize=False,
            other_ensemble=None,
            validation_subwrap=validation_subwrap,
            bins=bins,
            fill=True,
            hisstype="bar",
            fig=fig,
            ax=ax,
        )
        if best is not None or worst is not None:
            with self.model_ratio(
                best=best, worst=worst, validation_subwrap=validation_subwrap
            ):
                fig, ax = self.loss_histogramm(
                    normalize=False,
                    other_ensemble=None,
                    validation_subwrap=validation_subwrap,
                    fig=fig,
                    ax=ax,
                    bins=bins,
                    fill=True,
                    histtype="bar",
                )
        if items is not None:
            with self.model_items(items):
                fig, ax = self.loss_histogramm(
                    normalize=False,
                    other_ensemble=None,
                    validation_subwrap=validation_subwrap,
                    fig=fig,
                    ax=ax,
                    bins=bins,
                    fill=True,
                    histtype="bar",
                )
        return fig, ax

    def validation_error_ranking(
        self, validation_subwrap="original_validation_v2", loss_name="epe"
    ):
        # self.sort("min", subwrap=subwrap)

        min_losses = self.task_error(
            normalize=True,
            validation_subwrap=validation_subwrap,
            loss_name=loss_name,
        ).values
        fig, ax, _ = plots.bars(
            self.names, min_losses, figsize=[5, 2], fontsize=5, grid=False
        )
        ax.set_title(f"{validation_subwrap.replace('_', ' ')} ranking", fontsize=5)
        # ax.set_ylim(5.05, 5.9)
        plt_utils.trim_axis(ax)
        return fig, ax

    def arg_df(
        self,
        drop_columns_with=["movingedge", "movingbar", "gratings", "flashes"],
    ):

        parser = pd.io.parsers.ParserBase({"usecols": None})

        args = []
        for nnv in self.values():
            args.append(nnv.arg_df())

        for df in args:
            df.columns = parser._maybe_dedup_names(df.columns)

        df = pd.concat(args, ignore_index=True)

        if drop_columns_with:
            df.drop(
                columns=[
                    column
                    for column in df.columns
                    if any([word in column for word in drop_columns_with])
                ],
                inplace=True,
            )
        return df

    # def node_param_stats(self, parameter_name, title="", **kwargs):
    #     """Plots node statistics."""
    #     # Collect all parameters.
    #     all_params = []

    #     for nnv in self.values():
    #         all_params.append(
    #             nnv.tnn.best_network_parameters[parameter_name][:]
    #         )
    #     all_params = np.array(all_params)[:, self.node_index]  # Order.

    #     return plots.param_stats(
    #         self.node_types_sorted, all_params, title=title, **kwargs
    #     )

    # def filter_stats(self, **kwargs):
    #     """Plots edge statistics."""
    #     return edge_statistics(
    #         self.names, **kwargs
    #     )  # Collecting rf indices edge dataframe is still faster than from ctome wrap.

    # def similarity_per_node_grid(
    #     self,
    #     node_types=None,
    #     aspect_ratio=1,
    #     size_scale=100,
    #     fontsize=10,
    #     figsize=None,
    # ):
    #     """Similarity over models per node."""
    #     (
    #         similarity_dict,
    #         models,
    #     ) = self.central_node_gratings_response_similarity()
    #     ordered_node_types = node_types or self.node_types_sorted
    #     fig, axes, (gw, gh) = plt_utils.get_axis_grid(
    #         ordered_node_types, figsize=figsize, aspect_ratio=aspect_ratio
    #     )
    #     for i, node_type in enumerate(ordered_node_types):
    #         similarity_matrix = similarity_dict[node_type]
    #         plots.heatmap(
    #             similarity_matrix,
    #             xlabels=[],
    #             fig=fig,
    #             ax=axes[i],
    #             cbar=(True if i + 1 == len(ordered_node_types) else False),
    #             size_scale=size_scale,
    #             title=node_type,
    #             cmap=cm.get_cmap("Greens"),
    #             vmin=0,
    #             vmax=1,
    #             fontsize=fontsize,
    #         )
    #     return fig, axes

    # def similarity_of_node_type(
    #     self,
    #     node_type,
    #     title="",
    #     size_scale=100,
    #     figsize=[1, 1],
    #     fontsize=10,
    #     cbar=True,
    # ):
    #     (
    #         similarity_dict,
    #         models,
    #     ) = self.central_node_gratings_response_similarity()
    #     ordered_node_types = utils.order_nodes_list(
    #         [nt for nt in similarity_dict]
    #     )[0]
    #     similarity_matrix = similarity_dict[node_type]
    #     fig, ax, _, _ = plots.heatmap(
    #         similarity_matrix,
    #         xlabels=[],
    #         cbar=cbar,
    #         figsize=figsize,
    #         fontsize=fontsize,
    #         size_scale=size_scale,
    #         cmap=cm.get_cmap("Greens"),
    #         vmin=0,
    #         vmax=1,
    #     )
    #     return fig, ax

    # def similarity_per_node_bars(self, fig=None, ax=None, fontsize=10):
    #     _, (node_types, similarity) = self.total_similarity()
    #     fig, ax, _ = plots.bars(
    #         node_types, similarity, fig=fig, ax=ax, fontsize=fontsize
    #     )
    #     return fig, ax

    # def filter_input_mean(self, **kwargs):
    #     means, _ = self.filter_statistics()
    #     fig, ax, _, _ = plots.heatmap(
    #         means,
    #         self.node_types_sorted,
    #         cmap=cm.get_cmap("seismic"),
    #         symlog=1e-5,
    #         size_scale=1,
    #         **kwargs,
    #     )
    #     return fig, ax

    # def filter_input_variation(self, **kwargs):
    #     _, variation = self.filter_statistics()
    #     fig, ax, _, _ = plots.heatmap(
    #         variation,
    #         self.node_types_sorted,
    #         cmap=cm.get_cmap("seismic"),
    #         midpoint=0,
    #         size_scale=20,
    #         **kwargs,
    #     )
    #     return fig, ax

    def loss(
        self,
        mode="kfold_validation",
        fontsize=10,
        fig=None,
        ax=None,
        figsize=[4, 3],
        **kwargs,
    ):

        ymin = 1e15
        ymax = -1e15
        for tnn, nnv in self.items():
            fig, ax, loss = nnv.loss(
                mode,
                color=self.colors[tnn],
                fig=fig,
                ax=ax,
                figsize=figsize,
                fontsize=fontsize,
                mark_chkpts=False,
                **kwargs,
            )
            ymin = min(ymin, loss.min())
            ymax = max(ymax, loss.max())
        assert len(ax.lines) == len(self.names)
        ax.legend(ax.lines, self.names, fontsize=10)
        ax.set_ylim(ymin, ymax)
        return fig, ax

    def training_loss(
        self,
        smooth=0.05,
        fig=None,
        ax=None,
        subsample=1,
        mean=False,
        normalize=False,
        grid=True,
        validation_subwrap="original_validation_v2",
        cbar=True,
    ):

        task_error = self.task_error(validation_subwrap=validation_subwrap)
        colors, cmap, norm = task_error.colors, task_error.cmap, task_error.norm

        losses = [nnv.tnn.loss[:] for nnv in self.values()]

        if normalize:
            losses = losses / np.mean(
                [np.mean(nnv.tnn.loss[0:10_000]) for nnv in self.values()]
            )

        losses = np.array([loss[::subsample] for loss in losses])

        # chkpt_iters = np.array([nnv.tnn.chkpt_iter[:] for nnv in
        # self.values()])
        max_n_iters = max([len(loss) for loss in losses])

        _losses = np.zeros([len(losses), max_n_iters]) * np.nan
        for i, loss in enumerate(losses):
            n_iters = len(loss)
            _losses[i, :n_iters] = loss[:]
        fig, ax, smoothed, _ = plots.traces(
            _losses[::-1],
            x=np.arange(max_n_iters) * subsample,
            fontsize=5,
            figsize=[1, 1],
            smooth=smooth,
            fig=fig,
            ax=ax,
            color=colors[::-1],
            linewidth=0.5,
            highlight_mean=mean,
        )
        # if mean: mean = np.nanmean(smoothed, axis=0) ax.plot(
        #     np.arange(max_n_iters) * subsample, mean, color="k",
        #     linewidth=0.5,
        #     )
        # best = _losses[0] ax.plot( np.arange(max_n_chkpts) * iters_per_chkpts,
        # best, color="#8da0cb", linewidth=0.5,
        # )
        # worst = _losses[-1] ax.plot( np.arange(max_n_chkpts) *
        # iters_per_chkpts, worst, color="#fc8d62", linewidth=0.5,
        # )
        if normalize:
            ax.set_ylabel("training error", fontsize=5)
        else:
            ax.set_ylabel("training loss", fontsize=5)
        ax.set_xlabel("iterations", fontsize=5)
        # mean = losses.mean(axis=0) std = losses.std(axis=0)
        # ax.plot(losses.mean(axis=0)) ax.fill_between(np.arange(n_chkpts), mean
        # - std, mean + std, alpha=0.5)

        if cbar:
            plt_utils.add_colorbar(
                fig,
                ax,
                cmap=cmap,
                norm=norm,
                label="min task error",
                fontsize=5,
                tick_length=1,
                tick_width=0.5,
            )

        # self.sort("min", reverse=False)

        if grid:
            ax.set_yticks(np.linspace(*ax.get_ylim(), 10))
            ax.grid(True, linewidth=0.5)

        return fig, ax

    def validation_loss(
        self,
        normalize=False,
        grid=True,
        validation_subwrap="original_validation_v2",
        loss_name="epe",
        fig=None,
        ax=None,
        mean=False,
        cbar=True,
    ):

        # self.sort("min", reverse=True)

        fig, ax = plt_utils.init_plot(figsize=[1, 1], fontsize=5, fig=fig, ax=ax)

        task_error = self.task_error(validation_subwrap=validation_subwrap)
        colors, cmap, norm = task_error.colors, task_error.cmap, task_error.norm

        loss_name = dvs.analysis.validation_error._check_loss_name(
            self[0].tnn[validation_subwrap], loss_name
        )

        losses = np.array(
            [nnv.tnn[validation_subwrap][loss_name][:] for nnv in self.values()]
        )

        if normalize:
            losses = losses / np.mean(
                [nnv.tnn[validation_subwrap][loss_name][0] for nnv in self.values()]
            )

        chkpt_iter = self[0].tnn.chkpt_iter
        iters_per_chkpts = (chkpt_iter[1:] - chkpt_iter[:-1])[len(chkpt_iter) // 2]
        # chkpt_iters = np.array([nnv.tnn.chkpt_iter[:] for nnv in
        # self.values()])
        max_n_chkpts = max(len(loss) for loss in losses)

        _losses = np.zeros([len(losses), max_n_chkpts]) * np.nan
        for i, loss in enumerate(losses):
            n_chkpts = len(loss)
            _losses[i, :n_chkpts] = loss[:]
            ax.plot(
                np.arange(n_chkpts) * iters_per_chkpts,
                loss,
                color=colors[i],
                linewidth=0.5,
                alpha=0.8,
            )

        if mean:
            mean = np.nanmean(_losses, axis=0)
            ax.plot(
                np.arange(max_n_chkpts) * iters_per_chkpts,
                mean,
                color="k",
                ls="-.",
                linewidth=0.5,
            )
        # best = _losses[0] ax.plot( np.arange(max_n_chkpts) * iters_per_chkpts,
        # best, color="#8da0cb", linewidth=0.5,
        # )
        # worst = _losses[-1] ax.plot( np.arange(max_n_chkpts) *
        # iters_per_chkpts, worst, color="#fc8d62", linewidth=0.5,
        # )
        if normalize:
            ax.set_ylabel("task error", fontsize=5)
        else:
            ax.set_ylabel("validation loss", fontsize=5)
        ax.set_xlabel("iterations", fontsize=5)
        # mean = losses.mean(axis=0) std = losses.std(axis=0)
        # ax.plot(losses.mean(axis=0)) ax.fill_between(np.arange(n_chkpts), mean
        # - std, mean + std, alpha=0.5)

        if cbar:
            plt_utils.add_colorbar(
                fig,
                ax,
                cmap=cmap,
                norm=norm,
                label="min task error",
                fontsize=5,
                tick_length=1,
                tick_width=0.5,
            )
        # self.sort("min", reverse=False)
        if normalize:
            ymin, _ = plt_utils.get_lims((*losses,), 0.1)
            ax.set_ylim(max(ymin, 0), 1.1)

        if grid:
            ax.set_yticks(np.linspace(*ax.get_ylim(), 10))
            ax.grid(True, linewidth=0.5)

        return fig, ax

    def gradients_over_checkpoint(
        self, param, validation_subwrap="original_validation_v2"
    ):
        """
        nodes_time_cont, nodes_bias, edges_syn_strength, edges_syn_count
        """

        # self.sort("min", reverse=True)

        fig, ax = plt_utils.init_plot(figsize=[1, 1], fontsize=5)

        task_error = self.task_error(validation_subwrap=validation_subwrap)
        colors, cmap, norm = task_error.colors, task_error.cmap, task_error.norm
        gradient = []
        for nnv in self.values():
            try:
                gradient.append(nnv.tnn.gradients[param][:])
            except AssertionError as e:
                logging.warning(e)
        n_checkpoints, n_params = gradient[0].shape
        chkpt_iter = self[0].tnn.chkpt_iter
        iters_per_chkpts = (chkpt_iter[1:] - chkpt_iter[:-1])[len(chkpt_iter) // 2]
        # chkpt_iters = np.array([nnv.tnn.chkpt_iter[:] for nnv in
        # self.values()])
        max_n_chkpts = max([len(grad) for grad in gradient])

        # n_models, n_checkpoints, n_params
        _gradient = np.zeros([len(gradient), max_n_chkpts, n_params]) * np.nan
        for i, grad in enumerate(gradient):
            n_chkpts = len(grad)
            _gradient[i, :n_chkpts] = grad

        for model in range(_gradient.shape[0]):
            for iteration in range(_gradient.shape[1]):
                _y = _gradient[model, iteration]
                ax.scatter(np.ones(len(_y)) * iteration, _y, c=colors[model], s=0.5)
        #     for j in range(grad.shape[1]):
        #     ax.scatter()
        # ax.plot( np.arange(n_chkpts) * iters_per_chkpts, _gradient[i,
        #     :n_chkpts], color=colors[i], linewidth=0.5, alpha=0.8,
        # )

        # best = _gradient[0] ax.plot( np.arange(max_n_chkpts) *
        # iters_per_chkpts, best, color="#8da0cb", linewidth=0.5,
        # )
        # worst = _gradient[-1] ax.plot( np.arange(max_n_chkpts) *
        # iters_per_chkpts, worst, color="#fc8d62", linewidth=0.5,
        # )
        ax.set_ylabel(f"gradient {param_to_tex[param]}", fontsize=5)
        ax.set_xlabel("iterations", fontsize=5)
        # mean = gradient.mean(axis=0) std = gradient.std(axis=0)
        # ax.plot(gradient.mean(axis=0)) ax.fill_between(np.arange(n_chkpts),
        # mean - std, mean + std, alpha=0.5)

        plt_utils.add_colorbar(
            fig,
            ax,
            cmap=cmap,
            norm=norm,
            label="task error",
            fontsize=5,
            tick_length=1,
            tick_width=0.5,
        )

        # self.sort("min", reverse=False)

        return fig, ax

    def activity(self, validation_subwrap="original_validation_v2"):

        # self.sort("min", reverse=True)

        fig, ax = plt_utils.init_plot(figsize=[1, 1], fontsize=5)

        task_error = self.task_error(validation_subwrap=validation_subwrap)
        colors, cmap, norm = task_error.colors, task_error.cmap, task_error.norm

        activities = np.array([nnv.tnn.activity[:] for nnv in self.values()])
        # chkpt_iter = self[0].tnn.chkpt_iter iters_per_chkpts = (chkpt_iter[1:]
        # - chkpt_iter[:-1])[0] chkpt_iters = np.array([nnv.tnn.chkpt_iter[:]
        #   for nnv in self.values()])
        max_n_iters = max([len(activity) for activity in activities])

        _activities = np.zeros([len(activities), max_n_iters]) * np.nan
        for i, activity in enumerate(activities):
            n_chkpts = len(activity)
            _activities[i, :n_chkpts] = activity[:]
            ax.plot(
                np.arange(n_chkpts),
                activity,
                color=colors[i],
                linewidth=0.5,
                alpha=0.8,
            )
        mean = np.nanmean(_activities, axis=0)
        ax.plot(
            np.arange(max_n_iters),
            mean,
            color="k",
            linewidth=0.5,
        )
        # best = _activities[0] ax.plot( np.arange(max_n_iters), best,
        # color="#8da0cb", linewidth=0.5,
        # )
        # worst = _activities[-1] ax.plot( np.arange(max_n_iters), worst,
        # color="#fc8d62", linewidth=0.5,
        # )
        ax.set_ylabel("network activity", fontsize=5)
        ax.set_xlabel("iterations", fontsize=5)
        # mean = losses.mean(axis=0) std = losses.std(axis=0)
        # ax.plot(losses.mean(axis=0)) ax.fill_between(np.arange(n_chkpts), mean
        # - std, mean + std, alpha=0.5)

        # self.sort("min", reverse=False)

        return fig, ax

    def paper_direction_tuning(
        self,
        node_types=None,
        fig=None,
        axes=None,
        subwrap="movingedge_chkpt_best_v4",
    ):
        if fig is None or axes is None:
            fig, axes, _ = plt_utils.get_axis_grid(
                gridwidth=4,
                gridheight=1,
                projection="polar",
                aspect_ratio=1,
                figsize=[2.95, 0.83],
                wspace=0.25,
            )

        on_color_cmap = utils.color_utils.get_alpha_colormap(
            utils.color_utils.ON, len(self)
        )
        off_color_cmap = utils.color_utils.get_alpha_colormap(
            utils.color_utils.OFF, len(self)
        )

        for i in range(len(self)):
            self[i].paper_direction_tuning(
                node_types,
                fig=fig,
                axes=axes,
                on_color=on_color_cmap(i),
                off_color=off_color_cmap(i),
                subwrap=subwrap,
            )
        if len(self) <= 4:
            plt_utils.add_legend(
                axes[-1],
                labels=[
                    *[f"{i+1}. model - on responses" for i in range(len(self))],
                    *[f"{i+1}. model - off responses" for i in range(len(self))],
                ],
                colors=[
                    *[on_color_cmap(i) for i in range(len(self))],
                    *[off_color_cmap(i) for i in range(len(self))],
                ],
                fontsize=5,
                lw=1,
                loc="center left",
            )
        return fig, axes

    def paper_direction_tuning_v2(
        self,
        node_types,
        subwrap="movingedge_chkpt_best_v4",
        zorder={0: 50, 1: 100},
        **kwargs,
    ):
        """to be able to normalize by peak on and peak off to see scaling
        invariant tuning behaviour.
        """
        # TODO: looks weird needs to be checked, very low prio
        raise NotImplementedError
        if isinstance(node_types, list):
            pass
        elif node_types == "T4":
            node_types = ["T4a", "T4b", "T4c", "T4d"]
            zorder = {0: 50, 1: 100}
        elif node_types == "T5":
            node_types = ["T5a", "T5b", "T5c", "T5d"]
            zorder = {0: 100, 1: 50}
        elif node_types == "TmY":
            node_types = ["TmY3", "TmY4", "TmY13", "TmY18"]
        else:
            raise ValueError(f"{node_types}")

        self.init_movingbar(subwrap=subwrap)

        # to plot on / off normalized separately for all node_types and alls
        # models get responses into datastructure that allows the separation
        model_peak_responses = {
            intensity: {node_type: [] for node_type in node_types}
            for intensity in [0, 1]
        }
        for intensity in [0, 1]:
            for node_type in node_types:
                for name, nnv in self.items():
                    _, _, _, (theta, r), _ = nnv.movingbar.dsi(
                        node_type, intensity=intensity, round_angle=True
                    )
                    model_peak_responses[intensity][node_type].append(r)

        # now normalize by the peak response per neuron over both model and
        # intensity
        for node_type in node_types:
            # should be (#intensities, #models/ len(self), #angles)
            peak_response_on_off = np.array(
                [model_peak_responses[intensity][node_type] for intensity in [0, 1]]
            )
            for intensity in [0, 1]:
                peak_responses = np.array(model_peak_responses[intensity][node_type])
                model_peak_responses[intensity][node_type] = peak_responses / (
                    np.abs(peak_response_on_off).max(axis=(0, 1)) + 1e-12
                )

        fig, axes, _ = plt_utils.get_axis_grid(
            gridwidth=len(node_types),
            gridheight=1,
            projection="polar",
            aspect_ratio=1,
            figsize=[2.95 / 4 * len(node_types), 0.83],
            wspace=0.25,
        )

        on_color_cmap = utils.color_utils.get_alpha_colormap(
            utils.color_utils.ON, len(self)
        )
        off_color_cmap = utils.color_utils.get_alpha_colormap(
            utils.color_utils.OFF, len(self)
        )

        for node_type_index, node_type in enumerate(node_types):
            for intensity in [0, 1]:
                for model_index in range(len(self)):
                    r = model_peak_responses[intensity][node_type][model_index]
                    fig, ax = plots.polar(
                        theta,
                        r,
                        xlabel="",
                        color=on_color_cmap(model_index)
                        if intensity == 1
                        else off_color_cmap(model_index),
                        fontweight="normal",
                        stroke_kwargs={},
                        fig=fig,
                        ax=axes[node_type_index],
                        zorder=zorder[intensity],
                        **kwargs,
                    )

        if len(self) <= 4:
            plt_utils.add_legend(
                axes[-1],
                labels=[
                    *[f"{i+1}. model - on responses" for i in range(len(self))],
                    *[f"{i+1}. model - off responses" for i in range(len(self))],
                ],
                colors=[
                    *[on_color_cmap(i) for i in range(len(self))],
                    *[off_color_cmap(i) for i in range(len(self))],
                ],
                fontsize=5,
                lw=1,
                loc="center left",
            )

        return fig, ax

    def predictions_of_eyal_recordings(self):

        if not self[0]._initialized["eyal_data"]:
            self[0].init_eyal_data()

        def plot(sample, node_type):
            fig, ax, ylim, _ = self[0].prediction_of_eyal_recording(sample, node_type)

            if fig is not None and ax is not None:
                ymin = ylim[0]
                ymax = ylim[1]
                for tnn in self.names[1:]:
                    nnv = self[tnn]

                    response = nnv.tnn.full_data_eval.y_est.eyal[node_type][sample]
                    time = np.arange(len(response)) * nnv.tnn.full_data_eval.dt[()]
                    ax.plot(
                        time,
                        response,
                        label=f"{nnv.tnn.path.name}",
                        color=self.colors[tnn],
                        linewidth=2,
                    )
                    ymin = min(ymin, response.min())
                    ymax = max(ymax, response.max())
                ax.set_ylim(ymin, ymax)
                ax.legend(fontsize=10)
            return fig, ax

        for sample in np.concatenate(
            (self[0].tnn.val_data_index[:], self[0].tnn.train_data_index[:])
        ):
            for node_type in sorted(self[0].tnn.validation.y.eyal):
                yield plot(sample, node_type)

    def predictions_of_eyal_recordings_with_input(self):

        if not self[0]._initialized["eyal_data"]:
            self[0].init_eyal_data()

        def plot(sample, node_type):
            fig, axes, ylim, _ = self[0].prediction_of_eyal_recording_with_input(
                sample, node_type
            )

            if fig is not None and axes is not None:

                ymin = ylim[0]
                ymax = ylim[1]

                for tnn in self.names[1:]:
                    nnv = self[tnn]

                    response = nnv.tnn.full_data_eval.y_est.eyal[node_type][sample]
                    time = np.arange(len(response)) * nnv.tnn.full_data_eval.dt[()]
                    axes[1].plot(
                        time,
                        response,
                        label=f"{nnv.tnn.path.name}",
                        color=self.colors[tnn],
                        linewidth=2,
                    )
                    ymin = min(ymin, response.min())
                    ymax = max(ymax, response.max())
                axes[1].set_ylim(ymin, ymax)
                axes[1].legend(fontsize=10)
            return fig, axes

        for sample in np.concatenate(
            (self[0].tnn.val_data_index[:], self[0].tnn.train_data_index[:])
        ):
            for node_type in sorted(self[0].tnn.validation.y.eyal):
                yield plot(sample, node_type)

    def pred_eyal_rec_grid(self, node_type, width, intensity, t_stim):

        if not self[0]._initialized["eyal_data"]:
            self[0].init_eyal_data()

        fig, axes, ylim, samples = self[0].pred_eyal_rec_grid(
            node_type, width, intensity, t_stim
        )

        def plot(sample, node_type, ax):

            ymin = ylim[0]
            ymax = ylim[1]

            for tnn in self.names[1:]:
                nnv = self[tnn]

                response = nnv.tnn.full_data_eval.y_est.eyal[node_type][sample]
                time = np.arange(len(response)) * nnv.tnn.full_data_eval.dt[()]
                ax.plot(
                    time,
                    response,
                    label=f"{nnv.tnn.path.name}",
                    color=self.colors[tnn],
                    linewidth=2,
                )
                ymin = min(ymin, response.min())
                ymax = max(ymax, response.max())

            return fig, axes

        if fig is not None:
            for i, sample in enumerate(samples):
                plot(sample, node_type, axes[i])

            axes[-1].legend(fontsize=10)

        return fig, axes

    def pred_eyal_rec_grid_v2(self, node_type, width, intensity, t_stim):

        if not self[0]._initialized["eyal_data"]:
            self[0].init_eyal_data()

        fig, axes, ylim, samples = self[0].pred_eyal_rec_grid_v2(
            node_type, width, intensity, t_stim
        )

        def plot(sample, node_type, ax):

            ymin = ylim[0]
            ymax = ylim[1]

            for tnn in self.names[1:]:
                nnv = self[tnn]

                response = nnv.tnn.recordings.full_data_eval.y_est.eyal[node_type][
                    sample
                ]
                time = (
                    np.arange(len(response)) * nnv.tnn.recordings.full_data_eval.dt[()]
                )
                ax.plot(
                    time,
                    response,
                    label=f"{nnv.tnn.path.name}",
                    color=self.colors[tnn],
                    linewidth=2,
                )
                ymin = min(ymin, response.min())
                ymax = max(ymax, response.max())

            return fig, axes

        if fig is not None:
            for i, sample in enumerate(samples):
                plot(sample, node_type, axes[i])

            axes[-1].legend(fontsize=10)

        return fig, axes

    def pred_eyal_rec_with_input_by_params(
        self, angle, offset, t_stim, width, intensity, node_type
    ):

        if not self[0]._initialized["eyal_data"]:
            self[0].init_eyal_data()

        sample = (
            self[0]
            .eyal_dataset.get_params(
                angle=angle,
                offset=offset,
                t_stim=t_stim,
                width=width,
                intensity=intensity,
            )
            .index.values.item()
        )

        def plot(sample, node_type):
            fig, axes, ylim, _ = self[0].prediction_of_eyal_recording_with_input(
                sample, node_type
            )

            if fig is not None and axes is not None:

                ymin = ylim[0]
                ymax = ylim[1]

                for tnn in self.names[1:]:
                    nnv = self[tnn]

                    response = nnv.tnn.full_data_eval.y_est.eyal[node_type][sample]
                    time = np.arange(len(response)) * nnv.tnn.full_data_eval.dt[()]
                    axes[1].plot(
                        time,
                        response,
                        label=f"{nnv.tnn.path.name}",
                        color=self.colors[tnn],
                        linewidth=2,
                    )
                    ymin = min(ymin, response.min())
                    ymax = max(ymax, response.max())
                axes[1].set_ylim(ymin, ymax)
                axes[1].legend(fontsize=10)
            return fig, axes

        return plot(sample, node_type)

    def peak_response_angular(self, node_type, width=None, speed=None, intensity=None):
        rs = []
        for key, nnv in self.items():
            if not nnv._initialized["movingbar"]:
                nnv.init_movingbar()
            dsi, theta_pref, _, (theta, r), dsi_table = nnv.movingbar.dsi(
                node_type,
                round_angle=True,
                width=width,
                speed=speed,
                intensity=intensity,
            )
            rs.append(r)
        rs = np.array(rs)
        return theta, rs

    def dsi_ensemble_averaged(
        self,
        widths=None,
        intensities=None,
        speeds=None,
        subwrap="movingedge_chkpt_best_v4",
    ):
        class DSI(Namespace):
            pass

        dsi = DSI()

        self.init_movingbar(subwrap=subwrap)
        peak_responses_angular = []
        for nnv in self.values():
            peak_response_angular, (
                all_widths,
                all_intensities,
                all_speeds,
            ) = nnv.movingbar.peak_response_angular()
            peak_responses_angular.append(peak_response_angular)

        # n_models, n_angles, n_widths, n_intensities, n_speeds, n_cell_types
        peak_responses_angular = np.stack(peak_responses_angular, axis=0)
        # normalize over all stimuli parameter and all models to get
        # R(model, theta, width, intensity, speed, node_type) * exp(i*theta) / |sum_{model, theta, width, intensity, speed} R(model, theta, width, intensity, speed, node_type)|
        # that lies within the unit circle
        peak_responses_angular /= (
            np.nansum(
                np.abs(peak_responses_angular),
                axis=(0, 1, 2, 3, 4),
                keepdims=True,
            )
            + 1e-15
        )
        widths_index = nnv.movingbar.param_indices(all_widths, widths)
        intensities_index = nnv.movingbar.param_indices(all_intensities, intensities)
        speeds_index = nnv.movingbar.param_indices(all_speeds, speeds)
        n_models, n_angles, _, _, _, n_node_types = peak_responses_angular.shape
        n_widths, n_intensities, n_speeds = (
            len(widths_index),
            len(intensities_index),
            len(speeds_index),
        )
        peak_responses_angular = peak_responses_angular[
            :, :, widths_index, intensities_index, speeds_index
        ].reshape(n_models, n_angles, n_widths, n_intensities, n_speeds, n_node_types)
        dsis = np.abs(np.nansum(peak_responses_angular, axis=(0, 1, 2, 3, 4)))
        angular_traces = np.abs(np.nansum(peak_responses_angular, axis=(2, 3, 4)))

        dsi.angles = nnv.movingbar.angles
        dsi.angular_traces = angular_traces
        dsi.dsis = dsis
        dsi.peak_responses_angular = peak_responses_angular

        return dsi

    def polar_tuning(
        self,
        theta,
        peak_responses,
        colors=None,
        aggregate_models=None,
        norm=True,
        legend=True,
        legend_kwargs=dict(fontsize=5, bbox_to_anchor=[1.0, 0.5], loc="center left"),
        fontsize=10,
        zorder=None,
        fig=None,
        ax=None,
        **kwargs,
    ):

        if colors is None:
            colors = list(self.colors.values())
        # breakpoint()
        if aggregate_models == "sum" or (aggregate_models == "mean" and norm):
            peak_responses = np.nansum(peak_responses, axis=0, keepdims=True)
            colors = None
            labels = ["normalized vector-sum"] if norm else ["vector-sum"]
        elif aggregate_models == "mean":
            peak_responses = np.nanmean(peak_responses, axis=0, keepdims=True)
            colors = None
            labels = ["vector-mean"]
        else:
            labels = self.names

        return plots.multi_polar(
            theta,
            peak_responses,
            norm=norm,
            color=colors,
            label=labels,
            legend=legend,
            fontsize=fontsize,
            zorder=zorder,
            fig=fig,
            ax=ax,
            legend_kwargs=legend_kwargs,
            **kwargs,
        )

    def half_polar_tuning(
        self,
        theta,
        peak_responses,
        colors=None,
        aggregate_models=None,
        norm=True,
        legend=True,
        legend_kwargs=dict(fontsize=5, bbox_to_anchor=[1.0, 0.5], loc="center left"),
        fontsize=10,
        zorder=None,
        fig=None,
        ax=None,
        **kwargs,
    ):

        if colors is None:
            colors = list(self.colors.values())
        # breakpoint()
        if aggregate_models == "sum" or (aggregate_models == "mean" and norm):
            peak_responses = np.nansum(peak_responses, axis=0, keepdims=True)
            colors = None
            labels = ["normalized vector-sum"] if norm else ["vector-sum"]
        elif aggregate_models == "mean":
            peak_responses = np.nanmean(peak_responses, axis=0, keepdims=True)
            colors = None
            labels = ["vector-mean"]
        else:
            labels = self.names

        return plots.half_multi_polar(
            theta,
            peak_responses,
            norm=norm,
            color=colors,
            label=labels,
            legend=legend,
            fontsize=fontsize,
            zorder=zorder,
            fig=fig,
            ax=ax,
            legend_kwargs=legend_kwargs,
            **kwargs,
        )

    def motion_tuning(
        self,
        node_type,
        dsi_mode="mean",
        aggregate_models=None,
        width=None,
        colors=None,
        speed=None,
        intensity=None,
        round_angle=True,
        legend=True,
        fontsize=10,
        zorder=None,
        fig=None,
        ax=None,
        norm=True,
        subwrap="movingedge_chkpt_best_v4",
        legend_kwargs=dict(fontsize=5, bbox_to_anchor=[1.0, 0.5], loc="center left"),
        **kwargs,
    ):

        self.init_movingbar(subwrap=subwrap)

        rs = []
        for key, nnv in self.items():
            if dsi_mode == "max":
                dsi_fn = nnv.movingbar.max_dsi
            elif dsi_mode == "mean":
                dsi_fn = nnv.movingbar.dsi
            _, _, _, (theta, r), _ = dsi_fn(
                node_type,
                round_angle=round_angle,
                width=width,
                speed=speed,
                intensity=intensity,
            )
            rs.append(r)
        rs = np.array(rs)

        return self.polar_tuning(
            theta,
            rs,
            aggregate_models=aggregate_models,
            norm=norm,
            colors=colors,
            legend=legend,
            fontsize=fontsize,
            zorder=zorder,
            fig=fig,
            ax=ax,
            legend_kwargs=legend_kwargs,
            **kwargs,
        )

    def orientation_tuning(
        self,
        node_type,
        osi_mode="mean",
        aggregate_models=None,
        width=None,
        intensity=None,
        round_angle=True,
        legend=True,
        fontsize=10,
        colors=None,
        zorder=None,
        fig=None,
        ax=None,
        norm=True,
        subwrap="oriented_edge",
        legend_kwargs=dict(fontsize=5, bbox_to_anchor=[1.0, 0.5], loc="center left"),
        **kwargs,
    ):

        self.init_oriented_bar(subwrap=subwrap)

        rs = []
        for key, nnv in self.items():
            if osi_mode == "max":
                raise NotImplementedError
                # osi_fn = nnv.movingbar.max_osi
            elif osi_mode == "mean":
                osi_fn = nnv.oriented_bar.osi
            _, _, _, (theta, r), _ = osi_fn(
                node_type,
                round_angle=round_angle,
                width=width,
                intensity=intensity,
            )
            rs.append(r)
        rs = np.array(rs)

        return self.polar_tuning(
            theta,
            rs,
            aggregate_models=aggregate_models,
            norm=norm,
            colors=colors,
            legend=legend,
            fontsize=fontsize,
            zorder=zorder,
            fig=fig,
            ax=ax,
            legend_kwargs=legend_kwargs,
            **kwargs,
        )

    def half_orientation_tuning(
        self,
        node_type,
        osi_mode="mean",
        aggregate_models=None,
        width=None,
        intensity=None,
        round_angle=True,
        legend=True,
        fontsize=10,
        colors=None,
        zorder=None,
        fig=None,
        ax=None,
        norm=True,
        subwrap="oriented_bar",
        legend_kwargs=dict(fontsize=5, bbox_to_anchor=[1.0, 0.5], loc="center left"),
        **kwargs,
    ):

        self.init_oriented_bar(subwrap=subwrap)

        rs = []
        for key, nnv in self.items():
            if osi_mode == "max":
                raise NotImplementedError
                # osi_fn = nnv.movingbar.max_osi
            elif osi_mode == "mean":
                osi_fn = nnv.oriented_bar.osi
            _, _, _, (theta, r), _ = osi_fn(
                node_type,
                round_angle=round_angle,
                width=width,
                intensity=intensity,
            )
            rs.append(r)
        rs = np.array(rs)

        return self.half_polar_tuning(
            theta,
            rs,
            aggregate_models=aggregate_models,
            norm=norm,
            colors=colors,
            legend=legend,
            fontsize=fontsize,
            zorder=zorder,
            fig=fig,
            ax=ax,
            legend_kwargs=legend_kwargs,
            **kwargs,
        )

    def motion_tuning_mean(
        self,
        node_type,
        width=None,
        speed=None,
        intensity=None,
        round_angle=True,
        **kwargs,
    ):
        """Ensemble averaged motion tuning."""
        rs = []
        for key, nnv in self.items():
            if not nnv._initialized["movingbar"]:
                nnv.init_movingbar()
            dsi, theta_pref, _, (theta, r), dsi_table = nnv.movingbar.dsi(
                node_type,
                round_angle=round_angle,
                width=width,
                speed=speed,
                intensity=intensity,
            )
            rs.append(r)
        rs = np.array(rs).mean(0)

        return plots.polar(theta, rs, **kwargs)

    def motion_tuning_quartett(
        self,
        node_types=None,
        aspect_ratio=4,
        fontsize=10,
        figsize=[8, 0.5],
        **kwargs,
    ):
        """Good defaults for a grid of 4 node types."""
        node_types = node_types or self.node_types_sorted

        fig, axes, _ = plt_utils.get_axis_grid(
            node_types,
            projection="polar",
            aspect_ratio=aspect_ratio,
            figsize=figsize,
        )

        cmap = (
            cm.get_cmap("tab10") if len(node_types) <= 10 else plt_utils.cm_uniform_2d
        )
        for i, node_type in enumerate(node_types):
            self.motion_tuning_mean(
                node_type,
                xlabel=node_type,
                color=cmap(i),
                fig=fig,
                ax=axes[i],
                fontsize=fontsize,
                **kwargs,
            )
        return fig, axes

    def motion_tuning_all(
        self,
        node_type,
        figsize=[3, 6],
        fontsize=10,
        widths=None,
        nonlinearity=False,
        cbar=True,
        fig=None,
        axes=None,
        **kwargs,
    ):
        """
        Motion tuning for all stimuli conditions, averaged over ensemble.
        """

        widths = widths or self[0].movingbar.widths
        intensities = self[0].movingbar.intensities
        angles = self[0].movingbar.angles
        speeds = self[0].movingbar.speeds
        if fig is None and axes is None:
            fig, axes, _ = plt_utils.get_axis_grid(
                gridwidth=len(widths),
                gridheight=len(intensities),
                projection="polar",
                as_matrix=True,
                figsize=figsize,
            )
        # response = np.nanmax(self.movingbar.response(node_type=node_type),
        # axis=-1) if nonlinearity: response[np.isnan(response)] = 0 response =
        # np.maximum(response, 0)
        ymax = -1e15  # np.nanmax(response)

        for i, intensity in enumerate(intensities[::-1]):
            for j, width in enumerate(widths):

                _prs = []
                for nnv in self.values():
                    _prs.append(
                        np.abs(
                            nnv.movingbar.peak_response_angular(
                                node_type=node_type,
                                width=width,
                                intensity=intensity,
                                normalize=True,
                            )[0].reshape(len(angles), len(speeds))
                        )
                    )
                peak_response = np.mean(np.array(_prs), axis=0)
                ymax = max(ymax, np.max(peak_response))

                _cbar = False
                if j + 1 == len(widths) and cbar:
                    _cbar = True

                ylabel = ""
                if intensity == 0 and j % len(widths) == 0:
                    ylabel = "off-edges"
                elif intensity == 1 and j % len(widths) == 0:
                    ylabel = "on-edges"

                xlabel = ""
                if i + 1 == len(intensities):
                    xlabel = f"width: {width} col."

                plots.speed_polar(
                    angles,
                    peak_response.T,
                    speeds,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    ax=axes[i, j],
                    fig=fig,
                    cbar=_cbar,
                    fontsize=fontsize,
                    **kwargs,
                )

        for ax in axes.flatten():
            ax.set_ylim(0, ymax)
        return fig, axes, peak_response

    def motion_tuning_quartett_all(
        self,
        node_types=["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"],
        cbar=True,
        fontsize=10,
        figsize=[8, 0.2],
        **kwargs,
    ):
        """Good defaults for a grid of 4 node types."""
        node_types = node_types or self.node_types_sorted

        fig, axes, _ = plt_utils.get_axis_grid(
            gridwidth=len(node_types),
            gridheight=2,
            projection="polar",
            figsize=figsize,
            as_matrix=True,
        )

        # cmap = cm.get_cmap("tab10") if len(node_types) <= 10 else
        # plt_utils.cm_uniform_2d
        _cbar = False
        for i, node_type in enumerate(node_types):
            if i + 1 == len(node_types) and cbar:
                _cbar = True
            self.motion_tuning_all(
                node_type,
                fig=fig,
                axes=axes[:, i].reshape(2, 1),
                cbar=_cbar,
                cbar_offset=(1.3, 0),
                fontsize=fontsize,
                **kwargs,
            )
            if i != 0:
                for ax in axes[:, i].flatten():
                    ax.set_ylabel("")

        return fig, axes

    def paper_cluster_motion_tuning(self, *args, **kwargs):
        return self.cluster_motion_tuning(*args, **kwargs)

    def cluster_polar_tuning(
        self,
        node_type,
        polar_tuning_function,
        subwrap,
        cluster_indices=None,
        aggregate_models=None,
        normalize_peak_responses=True,
        equal_ylims=False,
        intensity=[0, 1],
        figsize=[0.94, 2.38],
        ax_title_color="k",
        validation_subwrap="original_validation_v2",
        # model_mask=None,
        colors=None,
        color_mask=None,
        color_substitute=(0.53, 0.53, 0.53, 0.1),
        # best=None,
        fontsize=5,
        marker_size=20,
        fig=None,
        axes=None,
        cell_type_agnostic=False,
        naturalistic_responses_wrap="augmented_sintel",
        embedding_method=None,
        embedding_kwargs=None,
        range_n_clusters=None,
        gm_n_init=1,
        gm_max_iter=1000,
        gm_random_state=0,
        loss_name="epe",
        **kwargs,
    ):
        """
        cluster_indices Dict[str, array
            from the one returned by self.cluster_indices. Optional.

        equal_ylims (Union[bool, str]): in combination with
            normalize_peak_response=False, to look at the different magnitudes
            of tuning across intensities, 'row', clusters, 'column', both
            intensities and clusters, 'all' or 'none'. Options include
            ['none', 'row', 'column', 'all'].
        """
        # to show the tuning only for the best models
        # if model_mask is None and best is not None:
        #     model_mask = self.best_models_mask(best)

        cluster_indices = cluster_indices or self.cluster_indices(
            node_type,
            cell_type_agnostic,
            naturalistic_responses_wrap,
            embedding_method,
            embedding_kwargs,
            range_n_clusters,
            gm_n_init,
            gm_max_iter,
            gm_random_state,
            validation_subwrap=validation_subwrap,
            loss_name=loss_name,
        )

        if colors is None:
            task_error = self.task_error(validation_subwrap=validation_subwrap)
            colors, cmap, norm = (
                task_error.colors,
                task_error.cmap,
                task_error.norm,
            )
        if color_mask is not None:
            colors[color_mask] = color_substitute
        n_clusters = len(cluster_indices)

        if intensity == [0, 1]:

            if fig is None or axes is None:
                fig, axes, _ = plt_utils.get_axis_grid(
                    gridwidth=2,
                    gridheight=n_clusters,
                    figsize=[2, n_clusters],
                    projection="polar",
                    as_matrix=True,
                    wspace=0.3,
                )

            axes[0, 0].annotate(
                "cluster",
                xy=(-0.4, 1),
                ha="center",
                va="top",
                xycoords=axes[0, 0].transAxes,
                fontsize=fontsize,
            )
            axes[0, 0].set_ylabel(
                "activity (a.u.)",
                fontsize=fontsize,
            )

            for i, indices in cluster_indices.items():
                with self.model_items(indices):
                    fig, ax = polar_tuning_function(
                        node_type,
                        aggregate_models=aggregate_models,
                        norm=normalize_peak_responses,
                        intensity=1,
                        colors=colors[indices],
                        fontsize=fontsize,
                        zorder=np.arange(len(self))[::-1],
                        fig=fig,
                        ax=axes[i, 0],
                        linewidth=1,
                        anglepad=-6,
                        legend=False,
                        alpha=0.7,
                        subwrap=subwrap,
                    )
                    axes[i, 0].annotate(
                        f"{i+1}",
                        xy=(-0.4, 0.5),
                        ha="center",
                        va="center",
                        fontsize=fontsize,
                        rotation=0,
                        xycoords=axes[i, 0].transAxes,
                    )

                    # ax.set_ylabel(f"cluster {i+1}", fontsize=5)
                    if i == 0:
                        ax.set_title("on", fontsize=5)
                    fig, ax = polar_tuning_function(
                        node_type,
                        aggregate_models=aggregate_models,
                        norm=normalize_peak_responses,
                        intensity=0,
                        colors=colors[indices],
                        fontsize=fontsize,
                        zorder=np.arange(len(self))[::-1],
                        fig=fig,
                        ax=axes[i, 1],
                        linewidth=1,
                        anglepad=-6,
                        legend=False,
                        alpha=0.7,
                        subwrap=subwrap,
                    )

                    if i == 0:
                        ax.set_title("off", fontsize=5)
        else:

            if fig is None or axes is None:
                fig, axes, _ = plt_utils.get_axis_grid(
                    gridwidth=1,
                    gridheight=n_clusters,
                    figsize=figsize,
                    projection="polar",
                    hspace=0.05,
                    as_matrix=True,
                )

            # axes[0].annotate( "cluster", xy=(-0.4, 1), ha="center", va="top",
            #     xycoords=axes[0].transAxes, fontsize=fontsize,
            # )

            for i, indices in cluster_indices.items():
                with self.model_items(indices):
                    fig, ax = polar_tuning_function(
                        node_type,
                        aggregate_models=aggregate_models,
                        norm=normalize_peak_responses,
                        intensity=intensity,
                        colors=colors[indices],
                        fontsize=fontsize,
                        zorder=np.arange(len(self))[::-1],
                        fig=fig,
                        ax=axes[i, 0],
                        linewidth=1.0,
                        anglepad=-6,
                        legend=False,
                        alpha=0.7,
                        subwrap=subwrap,
                    )

                    # axes[i].annotate( f"{i+1}", xy=(-0.4, 0.5), ha="center",
                    #     va="center", fontsize=5, rotation=0,
                    #     xycoords=axes[i].transAxes,
                    # )

                    #             ax.set_ylabel(f"cluster {i+1}", fontsize=5)
                    if i == 0:
                        if intensity == 0:
                            ax.set_title(
                                f"off-edge responses",
                                fontsize=fontsize,
                                color=ax_title_color,
                                y=1.25,
                            )
                        if intensity == 1:
                            ax.set_title(
                                f"on-edge responses",
                                fontsize=fontsize,
                                color=ax_title_color,
                                y=1.25,
                            )

        for row in range(axes.shape[0]):
            for column in range(axes.shape[1]):
                if (row, column) == (0, 0):
                    axes[row, column].set_ylabel("activity (a.u.)", fontsize=fontsize)
                else:
                    axes[row, column].set_xticklabels([])

        axes = np.atleast_2d(axes)
        MARKERS = clustering._check_markers(n_clusters)
        for row in range(axes.shape[0]):
            marker = MARKERS[row]
            plt_utils.add_cluster_marker(
                fig,
                axes[row, 0],
                marker=marker,
                color="#4F73AE",
                marker_size=marker_size,
            )

        # sync ylims normalize_peak_responses=False
        if equal_ylims and not normalize_peak_responses:
            ymins = np.zeros(axes.shape)
            ymaxs = np.zeros(axes.shape)

            for row in range(axes.shape[0]):
                for column in range(axes.shape[1]):
                    ymin, ymax = axes[row, column].get_ylim()
                    ymins[row, column] = ymin
                    ymaxs[row, column] = ymax

            if equal_ylims == "row":
                for row in range(axes.shape[0]):
                    ylims = plt_utils.get_lims([ymins[row], ymaxs[row]], 0.01)
                    for ax in axes[row]:
                        ax.set_ylim(*ylims)
            elif equal_ylims == "column":
                for column in range(axes.shape[1]):
                    ylims = plt_utils.get_lims(
                        [ymins[:, column], ymaxs[:, column]], 0.01
                    )
                    for ax in axes[:, column]:
                        ax.set_ylim(*ylims)
            elif equal_ylims == "all":
                ylims = plt_utils.get_lims([ymins, ymaxs], 0.01)
                for ax in axes.flatten():
                    ax.set_ylim(*ylims)
            else:
                raise ValueError

        return fig, axes

    def cluster_polar_tuning_horizontal(
        self,
        node_type,
        polar_tuning_function,
        subwrap,
        cluster_indices=None,
        aggregate_models=None,
        normalize_peak_responses=True,
        equal_ylims=False,
        intensity=[0, 1],
        figsize=[0.94, 2.38],
        figsizecm=[3.6, 6],
        ax_title_color="k",
        validation_subwrap="original_validation_v2",
        # model_mask=None,
        colors=None,
        color_mask=None,
        color_substitute=(0.53, 0.53, 0.53, 0.1),
        # best=None,
        fontsize=5,
        marker_size=20,
        fig=None,
        axes=None,
        cell_type_agnostic=False,
        naturalistic_responses_wrap="augmented_sintel",
        embedding_method=None,
        embedding_kwargs=None,
        range_n_clusters=None,
        gm_n_init=1,
        gm_max_iter=1000,
        gm_random_state=0,
        loss_name="epe",
        wspace=0.3,
        hspace=0.1,
        **kwargs,
    ):
        """
        cluster_indices Dict[str, array
            from the one returned by self.cluster_indices. Optional.

        equal_ylims (Union[bool, str]): in combination with
            normalize_peak_response=False, to look at the different magnitudes
            of tuning across intensities, 'row', clusters, 'column', both
            intensities and clusters, 'all' or 'none'. Options include
            ['none', 'row', 'column', 'all'].
        """
        # to show the tuning only for the best models
        # if model_mask is None and best is not None:
        #     model_mask = self.best_models_mask(best)

        cluster_indices = cluster_indices or self.cluster_indices(
            node_type,
            cell_type_agnostic,
            naturalistic_responses_wrap,
            embedding_method,
            embedding_kwargs,
            range_n_clusters,
            gm_n_init,
            gm_max_iter,
            gm_random_state,
            validation_subwrap=validation_subwrap,
            loss_name=loss_name,
        )

        if colors is None:
            task_error = self.task_error(validation_subwrap=validation_subwrap)
            colors, cmap, norm = (
                task_error.colors,
                task_error.cmap,
                task_error.norm,
            )
        if color_mask is not None:
            colors[color_mask] = color_substitute
        n_clusters = len(cluster_indices)

        if intensity == [0, 1]:

            if fig is None or axes is None:
                figsize = figsize or plt_utils.cm_to_inch(figsizecm)
                figsize = [figsize[0] * n_clusters, figsize[1]]
                fig, axes, _ = plt_utils.get_axis_grid(
                    gridwidth=n_clusters,
                    gridheight=2,
                    figsize=figsize,
                    projection="polar",
                    as_matrix=True,
                    wspace=wspace,
                    hspace=hspace,
                )

            axes[0, 0].annotate(
                "cluster",
                xy=(0, 1.4),
                ha="center",
                va="top",
                xycoords=axes[0, 0].transAxes,
                fontsize=fontsize,
            )
            axes[0, 0].set_ylabel(
                "response (a.u.)",
                fontsize=fontsize,
            )

            for i, indices in cluster_indices.items():
                with self.model_items(indices):
                    fig, ax = polar_tuning_function(
                        node_type,
                        aggregate_models=aggregate_models,
                        norm=normalize_peak_responses,
                        intensity=1,
                        colors=colors[indices],
                        fontsize=fontsize,
                        zorder=np.arange(len(self))[::-1],
                        fig=fig,
                        ax=axes[0, i],
                        linewidth=1,
                        anglepad=-6,
                        legend=False,
                        alpha=0.7,
                        subwrap=subwrap,
                    )
                    ax.annotate(
                        f"{i+1}",
                        xy=(0.5, 1.4),
                        ha="center",
                        va="top",
                        fontsize=fontsize,
                        rotation=0,
                        xycoords=ax.transAxes,
                    )

                    # ax.set_ylabel(f"cluster {i+1}", fontsize=5)
                    if i == 0:
                        ax.annotate(
                            "ON",
                            xy=(-0.4, 0.5),
                            ha="right",
                            va="center",
                            fontsize=fontsize,
                            rotation=0,
                            xycoords=ax.transAxes,
                        )
                    fig, ax = polar_tuning_function(
                        node_type,
                        aggregate_models=aggregate_models,
                        norm=normalize_peak_responses,
                        intensity=0,
                        colors=colors[indices],
                        fontsize=fontsize,
                        zorder=np.arange(len(self))[::-1],
                        fig=fig,
                        ax=axes[1, i],
                        linewidth=1,
                        anglepad=-6,
                        legend=False,
                        alpha=0.7,
                        subwrap=subwrap,
                    )

                    if i == 0:
                        ax.annotate(
                            "OFF",
                            xy=(-0.4, 0.5),
                            ha="right",
                            va="center",
                            fontsize=fontsize,
                            rotation=0,
                            xycoords=ax.transAxes,
                        )
        else:

            if fig is None or axes is None:
                figsize = figsize or plt_utils.cm_to_inch(figsizecm)
                figsize = [figsize[0] * n_clusters, figsize[1]]
                fig, axes, _ = plt_utils.get_axis_grid(
                    gridwidth=n_clusters,
                    gridheight=1,
                    figsize=figsize,
                    projection="polar",
                    hspace=hspace,
                    wspace=wspace,
                    as_matrix=True,
                )

            # axes[0].annotate( "cluster", xy=(-0.4, 1), ha="center", va="top",
            #     xycoords=axes[0].transAxes, fontsize=fontsize,
            # )

            for i, indices in cluster_indices.items():
                with self.model_items(indices):
                    fig, ax = polar_tuning_function(
                        node_type,
                        aggregate_models=aggregate_models,
                        norm=normalize_peak_responses,
                        intensity=intensity,
                        colors=colors[indices],
                        fontsize=fontsize,
                        zorder=np.arange(len(self))[::-1],
                        fig=fig,
                        ax=axes[0, i],
                        linewidth=1.0,
                        anglepad=-6,
                        legend=False,
                        alpha=0.7,
                        subwrap=subwrap,
                    )

                    # axes[i].annotate( f"{i+1}", xy=(-0.4, 0.5), ha="center",
                    #     va="center", fontsize=5, rotation=0,
                    #     xycoords=axes[i].transAxes,
                    # )

                    #             ax.set_ylabel(f"cluster {i+1}", fontsize=5)
                    if i == 0:
                        if intensity == 0:
                            ax.set_title(
                                f"OFF responses",
                                fontsize=fontsize,
                                color=ax_title_color,
                                y=1.25,
                            )
                        if intensity == 1:
                            ax.set_title(
                                f"ON responses",
                                fontsize=fontsize,
                                color=ax_title_color,
                                y=1.25,
                            )

        for row in range(axes.shape[0]):
            for column in range(axes.shape[1]):
                if (row, column) == (0, 0):
                    axes[row, column].set_ylabel("response (a.u.)", fontsize=fontsize)
                else:
                    axes[row, column].set_xticklabels([])

        axes = np.atleast_2d(axes)
        MARKERS = clustering._check_markers(n_clusters)
        for column in range(axes.shape[1]):
            marker = MARKERS[column]
            plt_utils.add_cluster_marker(
                fig,
                axes[-1, column],
                marker=marker,
                color="#4F73AE",
                marker_size=marker_size,
            )

        # sync ylims normalize_peak_responses=False
        if equal_ylims and not normalize_peak_responses:
            ymins = np.zeros(axes.shape)
            ymaxs = np.zeros(axes.shape)

            for row in range(axes.shape[0]):
                for column in range(axes.shape[1]):
                    ymin, ymax = axes[row, column].get_ylim()
                    ymins[row, column] = ymin
                    ymaxs[row, column] = ymax

            if equal_ylims == "row":
                for row in range(axes.shape[0]):
                    ylims = plt_utils.get_lims([ymins[row], ymaxs[row]], 0.01)
                    for ax in axes[row]:
                        ax.set_ylim(*ylims)
            elif equal_ylims == "column":
                for column in range(axes.shape[1]):
                    ylims = plt_utils.get_lims(
                        [ymins[:, column], ymaxs[:, column]], 0.01
                    )
                    for ax in axes[:, column]:
                        ax.set_ylim(*ylims)
            elif equal_ylims == "all":
                ylims = plt_utils.get_lims([ymins, ymaxs], 0.01)
                for ax in axes.flatten():
                    ax.set_ylim(*ylims)
            else:
                raise ValueError

        return fig, axes

    def cluster_motion_tuning(
        self,
        node_type,
        subwrap="movingedge_chkpt_best_v4",
        cluster_indices=None,
        aggregate_models=None,
        normalize_peak_responses=True,
        equal_ylims=False,
        intensity=[0, 1],
        figsize=[0.94, 2.38],
        ax_title_color="k",
        validation_subwrap="original_validation_v2",
        # model_mask=None,
        colors=None,
        color_mask=None,
        color_substitute=(0.53, 0.53, 0.53, 0.1),
        # best=None,
        # cluster_subwrap="umap",
        fontsize=5,
        marker_size=20,
        fig=None,
        axes=None,
        # embedding=None,
        # mask=None,
        cell_type_agnostic=False,
        naturalistic_responses_wrap="augmented_sintel",
        embedding_method=None,
        embedding_kwargs=None,
        range_n_clusters=None,
        gm_n_init=1,
        gm_max_iter=1000,
        gm_random_state=0,
        loss_name="epe",
        horizontal=False,
        figsizecm=None,
        **kwargs,
    ):
        kwargs.update(locals())
        kwargs.pop("self")
        if horizontal:
            return self.cluster_polar_tuning_horizontal(
                kwargs.pop("node_type"), self.motion_tuning, **kwargs
            )
        return self.cluster_polar_tuning(
            kwargs.pop("node_type"), self.motion_tuning, **kwargs
        )

    def cluster_orientation_tuning(
        self,
        node_type,
        subwrap="oriented_edge",
        cluster_indices=None,
        aggregate_models=None,
        normalize_peak_responses=True,
        equal_ylims=False,
        intensity=[0, 1],
        figsize=[0.94, 2.38],
        ax_title_color="k",
        validation_subwrap="original_validation_v2",
        # model_mask=None,
        colors=None,
        color_mask=None,
        color_substitute=(0.53, 0.53, 0.53, 0.1),
        # best=None,
        # cluster_subwrap="umap",
        fontsize=5,
        marker_size=20,
        fig=None,
        axes=None,
        # embedding=None,
        # mask=None,
        cell_type_agnostic=False,
        naturalistic_responses_wrap="augmented_sintel",
        embedding_method=None,
        embedding_kwargs=None,
        range_n_clusters=None,
        gm_n_init=1,
        gm_max_iter=1000,
        gm_random_state=0,
        loss_name="epe",
        horizontal=False,
        figsizecm=None,
    ):
        kwargs = locals()
        kwargs.pop("self")
        if horizontal:
            return self.cluster_polar_tuning_horizontal(
                kwargs.pop("node_type"), self.orientation_tuning, **kwargs
            )
        return self.cluster_polar_tuning(
            kwargs.pop("node_type"), self.orientation_tuning, **kwargs
        )

    def cluster_half_orientation_tuning(
        self,
        node_type,
        subwrap="oriented_bar",
        cluster_indices=None,
        aggregate_models=None,
        normalize_peak_responses=True,
        equal_ylims=False,
        intensity=[0, 1],
        figsize=[0.94, 2.38],
        ax_title_color="k",
        validation_subwrap="original_validation_v2",
        # model_mask=None,
        colors=None,
        color_mask=None,
        color_substitute=(0.53, 0.53, 0.53, 0.1),
        # best=None,
        # cluster_subwrap="umap",
        fontsize=5,
        marker_size=20,
        fig=None,
        axes=None,
        # embedding=None,
        # mask=None,
        cell_type_agnostic=False,
        naturalistic_responses_wrap="augmented_sintel",
        embedding_method=None,
        embedding_kwargs=None,
        range_n_clusters=None,
        gm_n_init=1,
        gm_max_iter=1000,
        gm_random_state=0,
        loss_name="epe",
        horizontal=False,
        figsizecm=None,
    ):
        kwargs = locals()
        kwargs.pop("self")
        if horizontal:
            return self.cluster_polar_tuning_horizontal(
                kwargs.pop("node_type"), self.half_orientation_tuning, **kwargs
            )
        return self.cluster_polar_tuning(
            kwargs.pop("node_type"), self.half_orientation_tuning, **kwargs
        )

    def movingbar_response(
        self,
        node_type,
        angle,
        width,
        intensity,
        speed,
        round_angle=True,
        fig=None,
        ax=None,
    ):
        """
        Response (temporal mean subtracted) to movingbar of all models.
        """
        rs = []
        ors = []
        for key, nnv in self.items():
            if not nnv._initialized["movingbar"]:
                nnv.init_movingbar()
            time, stimulus, response, opposite_resp = nnv.movingbar.traces(
                node_type,
                angle=angle,
                intensity=intensity,
                width=width,
                speed=speed,
            )
            rs.append(response)
            # ors.append(opposite_resp)

        rs = np.array(rs)
        # ors = np.array(ors)

        contour = stimulus
        zrs = rs - rs.mean(1, keepdims=True)
        fig, ax, _, _ = plots.traces(
            zrs,
            x=time,
            contour=contour,
            stim_line=True,
            color=[v for v in self.colors.values()],
            legend=[key for key in self.colors.keys()],
            xlabel="time in s",
            ylabel="voltage",
            fig=fig,
            ax=ax,
            legend_frame_alpha=1,
        )

        ax.set_title(f"{node_type} | $\\theta$={angle}", fontsize=10)
        return fig, ax

    def movingbar_response_mean(
        self,
        node_type,
        angle,
        width,
        intensity,
        speed,
        round_angle=True,
        fig=None,
        ax=None,
        figsize=[5, 3],
        **kwargs,
    ):
        """
        Ensemble average response (temporal mean subtracted) to movingbar of all
        models.
        """

        rs = []
        ors = []
        for key, nnv in self.items():
            if not nnv._initialized["movingbar"]:
                nnv.init_movingbar()
            time, stimulus, response, opposite_resp = nnv.movingbar.traces(
                node_type,
                angle=angle,
                intensity=intensity,
                width=width,
                speed=speed,
            )
            rs.append(response)
            ors.append(opposite_resp)

        rs = (np.array(rs) - np.array(rs).mean(1, keepdims=True)).mean(0)
        ors = (np.array(ors) - np.array(ors).mean(1, keepdims=True)).mean(0)

        fig, ax, _, _ = plots.traces(
            [rs, ors],
            x=time,
            contour=stimulus,
            fig=fig,
            ax=ax,
            figsize=figsize,
            **kwargs,
        )

        ax.set_title(f"{node_type} | $\\theta$={angle}", fontsize=10)
        return fig, ax

    def time_constants_heatmap(
        self,
        mode="tau",
        node_types=[
            "Mi1",
            "Mi4",
            "Mi9",
            "Mi10",
            "CT1(M10)",
            "C3",
            "TmY15",
            "Tm3",
        ],
        figsize=[10, 5],
        checkpoint="best",
    ):

        if node_types is None:
            node_types = self[0].node_types_sorted

        unique_node_types = self.ctome.unique_node_types[:].astype(str)
        indices = []
        for nt in node_types:
            indices.append(
                np.arange(len(unique_node_types))[unique_node_types == nt].item()
            )

        indices = np.array(indices)

        tcs = []
        for key, nnv in self.items():
            nnv.init_network(checkpoint=checkpoint)
            time_constants = nnv.network.nodes_time_const.detach().cpu().numpy()
            tcs.append(time_constants[indices])

        tcs = np.array(tcs)
        tcs_normed = tcs  # / (tcs.max(axis=1, keepdims=True) + 1e-15)

        index = np.argsort(tcs_normed[0])

        return plots.heatmap(
            tcs_normed[:, index],
            np.array(node_types)[index],
            ylabels=[model for model in self.colors],
            origin="upper",
            cmap=plt.cm.plasma,
            figsize=figsize,
            size_scale=1000,
        )

    def network_graph_dsi(self, intensity=[0, 1], mode="mean", speed=None, **kwargs):
        """Plots the abstracted network graph."""
        layout = dict(self.ctome.layout[:].astype(str))

        edges = self.ctome.edges.to_df()
        edges = pd.DataFrame(
            dict(
                source_type=edges.source_type[:].astype(str),
                target_type=edges.target_type[:].astype(str),
            )
        ).drop_duplicates()
        edges = list(
            map(
                lambda x: x.split(","),
                (edges.source_type + "," + edges.target_type),
            )
        )
        dsis = []
        for key, nnv in self.items():
            nodes, _dsis, theta = nnv.gratings_dataset.dsi(
                "all", intensity=intensity, speed=speed
            )
            dsis.append(_dsis)
        dsis = np.array(dsis)
        if mode == "mean":
            colors = dsis.mean(0)
        elif mode == "std":
            colors = dsis.std(0)
        return plots.network_graph(nodes, edges, layout, node_color=colors, **kwargs)

    def network_graph_fri(
        self,
        cmap=cm.get_cmap("seismic"),
        fri_mode="transient",
        absolute=False,
        mode="mean",
        **kwargs,
    ):
        """Plots the abstracted network graph."""
        layout = dict(self.ctome.layout[:].astype(str))
        nodes = self[0].node_types_sorted
        edges = self.ctome.edges.to_df()
        edges = pd.DataFrame(
            dict(
                source_type=edges.source_type[:].astype(str),
                target_type=edges.target_type[:].astype(str),
            )
        ).drop_duplicates()
        edges = list(
            map(
                lambda x: x.split(","),
                (edges.source_type + "," + edges.target_type),
            )
        )

        def get_fris(nnv):
            _fris = []
            for nt in nodes:
                fri, _, _ = nnv.flashes_dataset.fri(nt, mode=fri_mode)
                _fris.append(fri)
            return np.array(_fris)

        fris = []
        for key, nnv in self.items():
            fris.append(get_fris(nnv))
        fris = np.array(fris)

        if mode == "mean":
            colors = fris.mean(0)
        elif mode == "std":
            colors = fris.std(0)

        if absolute:
            colors[colors < 0] = -1
            colors[colors > 0] = 1
        return plots.network_graph(
            nodes, edges, layout, node_color=colors, node_cmap=cmap, **kwargs
        )

    def _scatter_on_violins(
        self,
        data,
        ax,
        validation_subwrap,
        scatter_best,
        scatter_all,
        validation_loss_name="epe",
        xticks=None,
        facecolor="none",
        edgecolor="k",
        best_scatter_alpha=1.0,
        all_scatter_alpha=0.35,
        best_index=None,
        best_color=None,
        all_marker="o",
        best_marker="o",
        linewidth=0.5,
        best_linewidth=0.75,
        uniform=[-0.35, 0.35],
    ):
        """
        Ax patch to scatter data on top of violins.
        data (Array): n_samples, n_variables.

        Not necessary to be a class method here. Should be optional for
        violin plot.
        """
        if scatter_all and not scatter_best:
            plt_utils.scatter_on_violins_or_bars(
                data,
                ax,
                xticks=xticks,
                zorder=100,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=all_scatter_alpha,
                uniform=uniform,
                marker=all_marker,
                linewidth=linewidth,
            )
        elif scatter_all:
            # in this case remove the best model datapoint for scatter all
            # to not show that point twice
            # this is error prone, because best index is entirely decoupled from data!
            # TODO
            best_index = (
                best_index
                if best_index is not None
                else self.best_index(
                    validation_subwrap=validation_subwrap,
                    loss_name=validation_loss_name,
                )
            )
            indices = list(range(data.shape[0]))
            indices.remove(best_index)
            plt_utils.scatter_on_violins_or_bars(
                data,
                ax,
                xticks=xticks,
                indices=indices,
                zorder=10,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=all_scatter_alpha,
                uniform=uniform,
                marker=all_marker,
                linewidth=linewidth,
            )

        if scatter_best:
            best_index = (
                best_index
                if best_index is not None
                else self.best_index(
                    validation_subwrap=validation_subwrap,
                    loss_name=validation_loss_name,
                )
            )
            best_color = (
                best_color
                if best_color is not None
                else self.best_color(
                    validation_subwrap=validation_subwrap,
                    loss_name=validation_loss_name,
                )
            )

            plt_utils.scatter_on_violins_or_bars(
                data,
                ax,
                xticks=xticks,
                indices=[best_index],
                alpha=best_scatter_alpha,
                linewidth=best_linewidth,
                edgecolor=best_color,
                facecolor=best_color,
                uniform=[0, 0],
                s=7.5,
                zorder=11,
                marker=best_marker,
            )

    def dsi_violins(
        self,
        subwrap="movingbar",
        intensity=None,
        cmap=plt.cm.Greens_r,
        color="b",
        figsize=None,
        fontsize=6,
        dsis=None,
        node_types=None,
        showmeans=False,
        showmedians=True,
        fig=None,
        ax=None,
        sorted_type_list=None,
        sort_descending=False,
        scatter_best=False,
        validation_subwrap="original_validation_v2",
        scatter_all=False,
        nonlinearity=True,
        subtract_baseline=False,
        scatter_edge_width=0.5,
        scatter_best_edge_width=0.75,
        scatter_edge_color="k",
        scatter_face_color="none",
        scatter_alpha=0.35,
        scatter_best_alpha=1.0,
        scatter_all_marker="o",
        scatter_best_marker="o",
        scatter_best_color=None,
        **kwargs,
    ):

        if intensity is None:
            intensity = [0, 1]
        if figsize is None:
            figsize = [10, 1]
        if dsis is None or node_types is None:
            dsis, node_types = self.dsis(
                subwrap,
                intensity,
                nonlinearity=nonlinearity,
                subtract_baseline=subtract_baseline,
            )  # , speed=speed)

        if sorted_type_list is not None:
            dsis = utils.nodes_edges_utils.sort_by_mapping_lists(
                node_types, sorted_type_list, dsis, axis=1
            )
            node_types = np.array(sorted_type_list)

        if sort_descending:
            medians = np.median(dsis, axis=0)
            index = np.argsort(medians)[::-1]
            dsis = dsis[:, index]
            node_types = node_types[index]

        if intensity == [0, 1]:
            cmap = plt.cm.Greens_r
            colors = None
        elif intensity in [[0], [1]]:
            cmap = None
            colors = (color,)
        else:
            raise ValueError("intensity must be one of [0], [1], [0, 1]")

        fig, ax, colors = plots.violin_groups(
            dsis.T[:, None],
            node_types[:],
            rotation=90,
            scatter=False,
            cmap=cmap,
            colors=colors,
            fontsize=fontsize,
            figsize=figsize,
            width=0.7,
            scatter_edge_color="white",
            scatter_radius=5,
            scatter_edge_width=0.5,
            showmeans=showmeans,
            showmedians=showmedians,
            fig=fig,
            ax=ax,
            **kwargs,
        )
        self._scatter_on_violins(
            dsis,
            ax,
            validation_subwrap,
            scatter_best,
            scatter_all,
            linewidth=scatter_edge_width,
            best_linewidth=scatter_best_edge_width,
            edgecolor=scatter_edge_color,
            facecolor=scatter_face_color,
            all_scatter_alpha=scatter_alpha,
            best_scatter_alpha=scatter_best_alpha,
            all_marker=scatter_all_marker,
            best_marker=scatter_best_marker,
            best_color=scatter_best_color,
        )

        return fig, ax, colors

    def dsis_paper_figure(
        self,
        intensity: list,
        subwrap="movingedge_chkpt_best_v4",
        validation_subwrap="original_validation_v2",
        fig=None,
        ax=None,
        bold_output_type_labels=True,
        scatter_best=True,
        sorted_type_list=None,
        known_on_off_first=True,
        ylim=(0, 1),
        sort_descending=False,
        scatter_all=False,
        nonlinearity=True,
        subtract_baseline=False,
        dsis=None,
        node_types=None,
        violin_alpha=1,
        **kwargs,
    ):

        if known_on_off_first:
            sorted_type_list = utils.nodes_list_sorting_on_off_unknown()

        ON = utils.color_utils.ON
        OFF = utils.color_utils.OFF
        BOTH = "#abcdef"

        if intensity == [0]:
            _color = OFF
        elif intensity == [1]:
            _color = ON
        else:
            _color = BOTH

        fig, ax, _ = self.dsi_violins(
            subwrap,
            intensity=intensity,
            figsize=[10, 1],
            showmedians=True,
            showmeans=False,
            color=_color,
            violin_alpha=violin_alpha,
            fig=fig,
            ax=ax,
            sorted_type_list=sorted_type_list,
            sort_descending=sort_descending,
            scatter_best=scatter_best,
            validation_subwrap=validation_subwrap,
            scatter_all=scatter_all,
            nonlinearity=nonlinearity,
            subtract_baseline=subtract_baseline,
            dsis=dsis,
            node_types=node_types,
            **kwargs,
        )

        ax.grid(False)

        if bold_output_type_labels:
            plt_utils.boldify_labels(self.output_node_types, ax)

        ax.set_ylim(*ylim)
        plt_utils.trim_axis(ax)
        plt_utils.set_spine_tick_params(
            ax,
            tickwidth=0.5,
            ticklength=3,
            ticklabelpad=2,
            labelsize=6,
            spinewidth=0.5,
        )
        plt_utils.patch_type_texts(ax)
        plt_utils.color_labels(["T4a", "T4b", "T4c", "T4d"], ON, ax)
        plt_utils.color_labels(["T5a", "T5b", "T5c", "T5d"], OFF, ax)
        return fig, ax

    def stacked_on_off_violins(
        self,
        violin_plot_fn,
        sorted_type_list=None,
        sort_known=False,
        figsize=[10, 2],
        fig=None,
        axes=None,
        **kwargs,
    ):
        if fig is None or axes is None:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_axes([0, 0.525, 1, 0.475])
            plt_utils.rm_spines(ax1, spines=("bottom",))
            ax2 = fig.add_axes([0, 0, 1, 0.475])
        else:
            ax1 = axes[0]
            plt_utils.rm_spines(ax1, spines=("bottom",))
            ax2 = axes[1]

        if sort_known:
            sorted_type_list = utils.nodes_edges_utils.nodes_list_sorting_ds_unknown()

        violin_plot_fn(
            [1],
            fig=fig,
            ax=ax1,
            ylim=(0, 1.2),
            sorted_type_list=sorted_type_list,
            **kwargs,
        )
        violin_plot_fn(
            [0],
            fig=fig,
            ax=ax2,
            ylim=(0, 1.2),
            sorted_type_list=sorted_type_list,
            **kwargs,
        )
        ax1.set_xticks([])
        ax1.set_yticks(np.arange(0, 1.2, 0.2))
        ax2.set_yticks(np.arange(0, 1.2, 0.2))
        # ax1.set_yticks(ax1.get_yticks()[1:])
        ax2.invert_yaxis()
        return fig, (ax1, ax2)

    def stacked_on_off_two_figures(
        self,
        violin_plot_fn,
        sorted_type_list=None,
        sort_known=False,
        figsize=[10, 1],
        fig=None,
        axes=None,
        kwargs1=None,
        kwargs2=None,
    ):
        """To match the size after saving pdf e.g.
        to the [10, 1] sized flash response violins for post editing in illustrator."""

        if kwargs1 is None:
            kwargs1 = {}
        if kwargs2 is None:
            kwargs2 = {}

        fig1, ax1 = plt_utils.init_plot(figsize=figsize)
        fig2, ax2 = plt_utils.init_plot(figsize=figsize)
        plt_utils.rm_spines(ax1, spines=("bottom",))

        if sort_known:
            sorted_type_list = utils.nodes_edges_utils.nodes_list_sorting_ds_unknown()

        violin_plot_fn(
            [1],
            fig=fig1,
            ax=ax1,
            ylim=(0, 1.1),
            sorted_type_list=sorted_type_list,
            **kwargs1,
        )
        violin_plot_fn(
            [0],
            fig=fig2,
            ax=ax2,
            ylim=(0, 1.1),
            sorted_type_list=sorted_type_list,
            **kwargs2,
        )
        ax1.set_xticks([])
        ax1.set_yticks(np.arange(0, 1.5, 0.5))
        ax2.set_yticks(np.arange(0, 1.5, 0.5))
        # ax1.set_yticks(ax1.get_yticks()[1:])
        ax2.invert_yaxis()
        return (fig1, fig2), (ax1, ax2)

    def dsis_stacked_on_off(
        self,
        sorted_type_list=None,
        sort_known=False,
        figsize=[10, 2],
        fig=None,
        axes=None,
        **kwargs,
    ):
        if fig is None or axes is None:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_axes([0, 0.525, 1, 0.475])
            plt_utils.rm_spines(ax1, spines=("bottom",))
            ax2 = fig.add_axes([0, 0, 1, 0.475])
        else:
            ax1 = axes[0]
            plt_utils.rm_spines(ax1, spines=("bottom",))
            ax2 = axes[1]

        if sort_known:
            sorted_type_list = utils.nodes_edges_utils.nodes_list_sorting_ds_unknown()

        self.dsis_paper_figure(
            [1],
            fig=fig,
            ax=ax1,
            ylim=(0, 1.2),
            sorted_type_list=sorted_type_list,
            **kwargs,
        )
        self.dsis_paper_figure(
            [0],
            fig=fig,
            ax=ax2,
            ylim=(0, 1.2),
            sorted_type_list=sorted_type_list,
            **kwargs,
        )
        ax1.set_xticks([])
        ax1.set_yticks(np.arange(0, 1.2, 0.2))
        ax2.set_yticks(np.arange(0, 1.2, 0.2))
        # ax1.set_yticks(ax1.get_yticks()[1:])
        ax2.invert_yaxis()
        return fig, (ax1, ax2)

    def dsis_stacked_on_off_two_figures(
        self,
        sorted_type_list=None,
        sort_known=False,
        figsize=[10, 1],
        violin_alphas=[1, 1],
        **kwargs,
    ):
        """To match the size after saving pdf e.g.
        to the [10, 1] sized flash response violins for post editing in illustrator."""
        fig1, ax1 = plt_utils.init_plot(figsize=figsize)
        fig2, ax2 = plt_utils.init_plot(figsize=figsize)
        plt_utils.rm_spines(ax1, spines=("bottom",))

        if sort_known:
            sorted_type_list = utils.nodes_edges_utils.nodes_list_sorting_ds_unknown()

        self.dsis_paper_figure(
            [1],
            fig=fig1,
            ax=ax1,
            ylim=(0, 1.1),
            sorted_type_list=sorted_type_list,
            violin_alpha=violin_alphas[0],
            **kwargs,
        )
        self.dsis_paper_figure(
            [0],
            fig=fig2,
            ax=ax2,
            ylim=(0, 1.1),
            sorted_type_list=sorted_type_list,
            violin_alpha=violin_alphas[1],
            **kwargs,
        )
        ax1.set_xticks([])
        ax1.set_yticks(np.arange(0, 1.5, 0.5))
        ax2.set_yticks(np.arange(0, 1.5, 0.5))
        # ax1.set_yticks(ax1.get_yticks()[1:])
        ax2.invert_yaxis()
        return (fig1, fig2), (ax1, ax2)

    def dsis_paper_si(
        self,
        intensity: list,
        subwrap="movingedge_chkpt_best_v4",
        best_ratio=0.5,
        worst_ratio=0.5,
        nonlinearity=True,
        fig=None,
        ax=None,
        validation_subwrap="original_validation_v2",
        legend_kwargs=dict(
            fontsize=5,
            markerscale=10,
            loc="lower left",
            bbox_to_anchor=(0.88, 0.9),
        ),
        known_first=True,
        **kwargs,
    ):

        if intensity == [0]:
            cmap = plt.cm.Blues_r
        elif intensity == [1]:
            cmap = plt.cm.Reds_r
        else:
            raise ValueError

        with self.model_ratio(best=best_ratio, validation_subwrap=validation_subwrap):
            dsis_best_models, node_types = self.dsis(
                subwrap,
                intensity,
                pre_stim=False,
                post_stim=False,
                nonlinearity=nonlinearity,
            )

        with self.model_ratio(worst=worst_ratio, validation_subwrap=validation_subwrap):
            dsis_worst_models, node_types = self.dsis(
                subwrap,
                intensity,
                pre_stim=False,
                post_stim=False,
                nonlinearity=nonlinearity,
            )

        if known_first:
            sorted_type_list = utils.nodes_list_sorting_on_off_unknown()
            dsis_best_models = utils.nodes_edges_utils.sort_by_mapping_lists(
                node_types, sorted_type_list, dsis_best_models, axis=1
            )
            dsis_worst_models = utils.nodes_edges_utils.sort_by_mapping_lists(
                node_types, sorted_type_list, dsis_worst_models, axis=1
            )
            node_types = sorted_type_list

        dsis = np.stack((dsis_best_models, dsis_worst_models), axis=1)
        fig, ax, C = plots.plots.violin_groups(
            dsis.T,
            node_types[:],
            rotation=90,
            scatter=False,
            fontsize=6,
            figsize=[10, 1],
            width=0.7,
            scatter_edge_color="white",
            scatter_radius=5,
            scatter_edge_width=0.25,
            cdist=100,
            cmap=cmap,
            showmedians=True,
            showmeans=False,
            violin_marker_lw=0.25,
            legend=[
                f"best {best_ratio:.0%} models",
                f"worst {worst_ratio:.0%} models",
            ],
            legend_kwargs=legend_kwargs,
            fig=fig,
            ax=ax,
        )
        ax.set_ylim(-0.05, 1)
        # dvs.plots.rm_spines(ax, ('left',))
        ax.grid(False)
        plt_utils.trim_axis(ax)
        plt_utils.set_spine_tick_params(
            ax,
            tickwidth=0.5,
            ticklength=3,
            ticklabelpad=2,
            labelsize=6,
            spinewidth=0.5,
        )

        if intensity == [0]:
            plt_utils.color_labels(
                ["T5a", "T5b", "T5c", "T5d"], utils.color_utils.OFF, ax
            )
        elif intensity == [1]:
            plt_utils.color_labels(
                ["T4a", "T4b", "T4c", "T4d"], utils.color_utils.ON, ax
            )

        plt_utils.boldify_labels(self.output_node_types, ax)

        plt_utils.patch_type_texts(ax)
        ax.hlines(
            0,
            *ax.get_xlim(),
            linewidth=0.5,
            linestyles="dashed",
            color="0.5",
            zorder=0,
        )
        return fig, ax

    def dsis_si_stacked_on_off(
        self,
        # sort_known=False,
        figsize=[10, 2],
        nonlinearity=True,
        **kwargs,
    ):
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0, 0.525, 1, 0.475])
        plt_utils.rm_spines(ax1, spines=("bottom",))
        ax2 = fig.add_axes([0, 0, 1, 0.475])

        # if sort_known: sorted_type_list = (
        #     utils.nodes_edges_utils.nodes_list_sorting_ds_unknown()
        #     )

        self.dsis_paper_si(
            intensity=[1],
            best_ratio=0.5,
            worst_ratio=0.5,
            fig=fig,
            ax=ax1,
            nonlinearity=nonlinearity,
            **kwargs,
        )

        self.dsis_paper_si(
            intensity=[0],
            best_ratio=0.5,
            worst_ratio=0.5,
            fig=fig,
            ax=ax2,
            nonlinearity=nonlinearity,
            legend_kwargs=dict(
                fontsize=5,
                markerscale=10,
                loc="lower left",
                bbox_to_anchor=(0.9, 0.1),
            ),
            **kwargs,
        )

        ax1.set_xticks([])
        ax1.set_yticks(np.arange(0, 1.2, 0.2))
        ax2.set_yticks(np.arange(0, 1.2, 0.2))
        # ax1.set_yticks(ax1.get_yticks()[1:])
        ax2.invert_yaxis()
        return fig, (ax1, ax2)

    def pds_violins(
        self,
        subwrap="movingbar",
        intensity=[0, 1],
        cmap=plt.cm.Greens_r,
        color="b",
        figsize=[10, 1],
        fontsize=6,
        preferred_directions=None,
        node_types=None,
        showmeans=False,
        showmedians=True,
        fig=None,
        ax=None,
        sorted_type_list=None,
        scatter_best=False,
        validation_subwrap="original_validation_v2",
        scatter_all=False,
        **kwargs,
    ):

        if preferred_directions is None or node_types is None:
            preferred_directions, node_types = self.preferred_directions(
                subwrap, intensity
            )  # , speed=speed)

        if sorted_type_list is not None:
            preferred_directions = utils.nodes_edges_utils.sort_by_mapping_lists(
                node_types, sorted_type_list, preferred_directions, axis=1
            )
            node_types = np.array(sorted_type_list)

        if intensity == [0, 1]:
            cmap = plt.cm.Greens_r
            colors = None
        elif intensity in [[0], [1]]:
            cmap = None
            colors = (color,)
        else:
            raise ValueError("intensity must be one of [0], [1], [0, 1]")

        fig, ax, colors = plots.violin_groups(
            preferred_directions.T[:, None],
            node_types[:],
            rotation=90,
            scatter=False,
            cmap=cmap,
            colors=colors,
            fontsize=fontsize,
            figsize=figsize,
            width=0.7,
            scatter_edge_color="white",
            scatter_radius=5,
            scatter_edge_width=0.5,
            showmeans=showmeans,
            showmedians=showmedians,
            fig=fig,
            ax=ax,
            **kwargs,
        )
        self._scatter_on_violins(
            preferred_directions,
            ax,
            validation_subwrap,
            scatter_best,
            scatter_all,
        )
        ax.set_ylim(-180, 180)

        ax.hlines(180, *ax.get_xlim(), zorder=0, color="k", alpha=0.5, linestyle="-.")
        ax.hlines(90, *ax.get_xlim(), zorder=0, color="k", alpha=0.5, linestyle="-.")
        ax.hlines(0, *ax.get_xlim(), zorder=0, color="k", alpha=0.5, linestyle="-.")
        ax.hlines(-90, *ax.get_xlim(), zorder=0, color="k", alpha=0.5, linestyle="-.")
        ax.hlines(-180, *ax.get_xlim(), zorder=0, color="k", alpha=0.5, linestyle="-.")

        return fig, ax, colors

    def pds_paper_figure(
        self,
        intensity: list,
        subwrap="movingedge_chkpt_best_v4",
        validation_subwrap="original_validation_v2",
        fig=None,
        ax=None,
        bold_output_type_labels=True,
        scatter_best=True,
        sorted_type_list=None,
        ylim=(-180, 180),
        sort_descending=False,
        scatter_all=False,
    ):
        ON = utils.color_utils.ON
        OFF = utils.color_utils.OFF
        BOTH = "#abcdef"

        if intensity == [0]:
            _color = OFF
        elif intensity == [1]:
            _color = ON
        else:
            _color = BOTH

        fig, ax, _ = self.pds_violins(
            subwrap,
            intensity=intensity,
            figsize=[10, 1],
            showmedians=True,
            showmeans=False,
            color=_color,
            violin_alpha=1,
            fig=fig,
            ax=ax,
            sorted_type_list=sorted_type_list,
            sort_descending=sort_descending,
            scatter_best=scatter_best,
            validation_subwrap=validation_subwrap,
            scatter_all=scatter_all,
        )

        # dvs.plots.rm_spines(ax, ('left',))
        ax.grid(False)
        if bold_output_type_labels:
            for tick in ax.xaxis.get_major_ticks():
                if tick.label1.get_text() in self.output_node_types:
                    tick.label1.set_weight("bold")

        ax.set_ylim(*ylim)
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 60, 60))
        plots.plt_utils.trim_axis(ax)
        ax.tick_params(axis="both", width=0.5, length=3, pad=2, labelsize=6)
        [i.set_linewidth(0.5) for i in ax.spines.values()]

        on = ["T4a", "T4b", "T4c", "T4d"]
        off = ["T5a", "T5b", "T5c", "T5d"]

        for tick in ax.xaxis.get_major_ticks():
            if tick.label1.get_text() in on:
                tick.label1.set_color(ON)
            elif tick.label1.get_text() in off:
                tick.label1.set_color(OFF)
        return fig, ax

    def pds_stacked_on_off(
        self,
        sorted_type_list=None,
        sort_known=True,
        figsize=[10, 2],
        **kwargs,
    ):
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0, 0.525, 1, 0.475])
        plt_utils.rm_spines(ax1, spines=("bottom",))
        ax2 = fig.add_axes([0, 0, 1, 0.475])

        if sort_known:
            sorted_type_list = utils.nodes_edges_utils.nodes_list_sorting_ds_unknown()

        self.pds_paper_figure(
            [1],
            fig=fig,
            ax=ax1,
            ylim=(-180, 180),
            sorted_type_list=sorted_type_list,
            **kwargs,
        )
        self.pds_paper_figure(
            [0],
            fig=fig,
            ax=ax2,
            ylim=(-180, 180),
            sorted_type_list=sorted_type_list,
            **kwargs,
        )
        ax1.set_xticks([])
        # ax1.set_yticks(np.arange(-180, 240, 60))
        # ax2.set_yticks(np.arange(-180, 240, 60))
        # ax1.set_yticks(ax1.get_yticks()[1:])
        ax2.invert_yaxis()
        return fig, (ax1, ax2)

    def osi_violins(
        self,
        subwrap="oriented_bar",
        intensity=[0, 1],
        cmap=plt.cm.Greens_r,
        color="b",
        figsize=[10, 1],
        fig=None,
        ax=None,
        fontsize=6,
        osis=None,
        node_types=None,
        showmeans=False,
        showmedians=True,
        validation_subwrap="original_validation_v2",
        scatter_best=True,
        scatter_all=True,
        sorted_type_list=None,
        sort_descending=False,
        nonlinearity=True,
        **kwargs,
    ):

        if osis is None or node_types is None:
            osis, node_types = self.osis(
                subwrap, intensity, nonlinearity=nonlinearity
            )  # , speed=speed)

        if sorted_type_list is not None:
            osis = utils.nodes_edges_utils.sort_by_mapping_lists(
                node_types, sorted_type_list, osis, axis=1
            )
            node_types = np.array(sorted_type_list)

        if sort_descending:
            medians = np.median(osis, axis=0)
            index = np.argsort(medians)[::-1]
            osis = osis[:, index]
            node_types = node_types[index]

        if intensity == [0, 1]:
            cmap = plt.cm.Greens_r
            colors = None
        elif intensity in [[0], [1]]:
            cmap = None
            colors = (color,)
        else:
            raise ValueError("intensity must be one of [0], [1], [0, 1]")

        fig, ax, colors = plots.violin_groups(
            osis.T[:, None],
            node_types[:],
            rotation=90,
            scatter=False,
            cmap=cmap,
            colors=colors,
            fontsize=fontsize,
            figsize=figsize,
            width=0.7,
            scatter_edge_color="white",
            scatter_radius=5,
            scatter_edge_width=0.5,
            showmeans=showmeans,
            showmedians=showmedians,
            fig=fig,
            ax=ax,
            **kwargs,
        )

        self._scatter_on_violins(
            osis, ax, validation_subwrap, scatter_best, scatter_all
        )
        ax.set_ylim(-0.05, 1)
        return fig, ax, colors

    def osis_paper_figure(
        self,
        intensity: list,
        subwrap="oriented_bar",
        validation_subwrap="original_validation_v2",
        fig=None,
        ax=None,
        bold_output_type_labels=True,
        scatter_best=True,
        sorted_type_list=None,
        known_on_off_first=True,
        ylim=(0, 1),
        sort_descending=False,
        scatter_all=False,
        nonlinearity=True,
    ):
        if known_on_off_first:
            sorted_type_list = utils.nodes_list_sorting_on_off_unknown()

        ON = utils.color_utils.ON_OSI
        OFF = utils.color_utils.OFF_OSI
        BOTH = "#abcdef"

        if intensity == [0]:
            _color = OFF
        elif intensity == [1]:
            _color = ON
        else:
            _color = BOTH

        osis, node_types = self.osis(subwrap, intensity)

        fig, ax, _ = self.osi_violins(
            subwrap,
            intensity=intensity,
            osis=osis,
            node_types=node_types,
            figsize=[10, 1],
            showmedians=True,
            showmeans=False,
            color=_color,
            violin_alpha=1,
            fig=fig,
            ax=ax,
            sorted_type_list=sorted_type_list,
            sort_descending=sort_descending,
            scatter_best=scatter_best,
            validation_subwrap=validation_subwrap,
            scatter_all=scatter_all,
            nonlinearity=nonlinearity,
        )

        ax.grid(False)

        if bold_output_type_labels:
            plt_utils.boldify_labels(self.output_node_types, ax)

        ax.set_ylim(*ylim)
        plt_utils.trim_axis(ax)
        plt_utils.set_spine_tick_params(
            ax,
            tickwidth=0.5,
            ticklength=3,
            ticklabelpad=2,
            labelsize=6,
            spinewidth=0.5,
        )
        # plt_utils.color_labels(["T4a", "T4b", "T4c", "T4d"], ON, ax)
        # plt_utils.color_labels(["T5a", "T5b", "T5c", "T5d"], OFF, ax)
        plt_utils.patch_type_texts(ax)
        return fig, ax

    def osis_paper_si(
        self,
        intensity: list,
        subwrap="oriented_bar",
        best_ratio=0.5,
        worst_ratio=0.5,
    ):

        with self.model_ratio(best=best_ratio):
            osis_best_models, node_type = self.osis(subwrap, intensity)

        with self.model_ratio(worst=worst_ratio):
            osis_worst_models, node_type = self.osis(subwrap, intensity)

        if intensity == [0]:
            cmap = plt.cm.Oranges_r
        elif intensity == [1]:
            cmap = plt.cm.Greens_r
        else:
            raise ValueError

        osis = np.stack((osis_best_models, osis_worst_models), axis=1)
        fig, ax, C = plots.plots.violin_groups(
            osis.T,
            node_type[:],
            rotation=90,
            scatter=False,
            fontsize=6,
            figsize=[10, 1],
            width=0.7,
            scatter_edge_color="white",
            scatter_radius=5,
            scatter_edge_width=0.25,
            cdist=100,
            cmap=cmap,
            showmedians=True,
            showmeans=False,
            violin_marker_lw=0.25,
            legend=[f"best {best_ratio:.0%}", f"worst {worst_ratio:.0%}"],
            legend_kwargs=dict(
                fontsize=5,
                markerscale=10,
                loc="lower left",
                bbox_to_anchor=(0.9, 0.9),
            ),
        )
        ax.set_ylim(-0.05, 1)
        ax.set_ylim(-0.05, 1)
        # dvs.plots.rm_spines(ax, ('left',))
        ax.grid(False)

        for tick in ax.xaxis.get_major_ticks():
            if tick.label1.get_text().startswith("T"):
                tick.label1.set_weight("bold")

        plots.plt_utils.trim_axis(ax)
        ax.tick_params(axis="both", width=0.5, length=3, pad=2, labelsize=6)
        [i.set_linewidth(0.5) for i in ax.spines.values()]
        return fig, ax

    def fri_violins(
        self,
        radius,
        mode,
        scatter_best=True,
        scatter_all=True,
        validation_subwrap="original_validation_v2",
        subwrap="flashes",
        cmap=plt.cm.Oranges_r,
        colors=None,
        color="b",
        figsize=[10, 1],
        fontsize=6,
        fris=None,
        node_types=None,
        showmeans=False,
        showmedians=True,
        sorted_type_list=None,
        scatter_edge_width=0.5,
        scatter_best_edge_width=0.75,
        scatter_edge_color="k",
        scatter_face_color="none",
        scatter_alpha=0.35,
        scatter_best_alpha=1.0,
        scatter_all_marker="o",
        scatter_best_marker="o",
        scatter_best_color=None,
        known_first=True,
        subtract_baseline=False,
        nonlinearity=False,
        nonnegative=True,
        **kwargs,
    ):

        # if comparison_ensemble is not None:
        #     if (
        #         hasattr(self, "_censemble", None) is not None
        #         and self._censemble.name == comparison_ensemble
        #     ):
        #         self._censemble = EnsembleViews(comparison_ensemble)
        #         self._censemble.init_flashes(subwrap)
        #         _cfris, node_types = self.fris(radius, mode, subwrap=subwrap)

        if fris is None or node_types is None:
            fris, node_types = self.fris(
                radius,
                mode,
                subwrap=subwrap,
                subtract_baseline=subtract_baseline,
                nonlinearity=nonlinearity,
                nonnegative=nonnegative,
            )

        # always add empty group axis for violin plot unless fris is provided
        # with 3 axes
        if len(fris.shape) != 3:
            fris = fris[:, None]

        # transpose to #cell_types, #groups, #samples
        if fris.shape[0] != len(node_types):
            fris = np.transpose(fris, (2, 1, 0))

        if sorted_type_list is not None:
            fris = utils.nodes_edges_utils.sort_by_mapping_lists(
                node_types, sorted_type_list, fris, axis=0
            )
            node_types = np.array(sorted_type_list)

        if known_first:
            _node_types = utils.nodes_list_sorting_on_off_unknown(node_types)
            fris = utils.nodes_edges_utils.sort_by_mapping_lists(
                node_types, _node_types, fris, axis=0
            )
            node_types = np.array(_node_types)

        if colors is not None:
            pass
        elif cmap is not None:
            colors = None
        elif color is not None:
            cmap = None
            colors = (color,)

        fig, ax, colors = plots.violin_groups(
            fris,
            node_types[:],
            rotation=90,
            scatter=False,
            cmap=cmap,
            colors=colors,
            fontsize=fontsize,
            figsize=figsize,
            width=0.7,
            # scatter_edge_color=scatter_edge_color,
            # scatter_radius=5,
            # scatter_edge_width=scatter_edge_width,
            showmeans=showmeans,
            showmedians=showmedians,
            **kwargs,
        )

        if fris.shape[1] == 1:
            self._scatter_on_violins(
                fris.T.squeeze(),
                ax,
                validation_subwrap,
                scatter_best,
                scatter_all,
                linewidth=scatter_edge_width,
                best_linewidth=scatter_best_edge_width,
                edgecolor=scatter_edge_color,
                facecolor=scatter_face_color,
                all_scatter_alpha=scatter_alpha,
                best_scatter_alpha=scatter_best_alpha,
                all_marker=scatter_all_marker,
                best_marker=scatter_best_marker,
                best_color=scatter_best_color,
            )
        return fig, ax, colors, fris

    def fris_paper_figure(
        self,
        mode="convention?",
        validation_subwrap="original_validation_v2",
        bold_output_type_labels=True,
        scatter_best=True,
        scatter_all=True,
        sorted_type_list=None,
        known_first=True,
        figsize=[10, 1],
        cmap=plt.cm.Greys_r,
        ylim=(-1, 1),
        color_known_types=True,
        fontsize=6,
        **kwargs,
    ):

        fig, ax, colors, fris = self.fri_violins(
            6,
            mode=mode,
            cmap=cmap,
            fontsize=fontsize,
            sorted_type_list=sorted_type_list,
            figsize=figsize,
            scatter_best=scatter_best,
            scatter_all=scatter_all,
            validation_subwrap=validation_subwrap,
            known_first=known_first,
            **kwargs,
        )
        ax.grid(False)

        if bold_output_type_labels:
            plt_utils.boldify_labels(self.output_node_types, ax)

        ax.set_ylim(*ylim)
        plt_utils.trim_axis(ax)
        plt_utils.set_spine_tick_params(
            ax,
            tickwidth=0.5,
            ticklength=3,
            ticklabelpad=2,
            labelsize=6,
            spinewidth=0.5,
        )

        if color_known_types:
            ax = plt_utils.flash_response_color_labels(ax)

        ax.hlines(
            0,
            min(ax.get_xticks()),
            max(ax.get_xticks()),
            linewidth=0.25,
            # linestyles="dashed",
            color="k",
            zorder=0,
        )
        ax.set_yticks(np.arange(-1.0, 1.5, 0.5))

        plt_utils.patch_type_texts(ax)
        return fig, ax

    def fris_paper_si(
        self,
        radius=6,
        mode="transient",
        subwrap="flashes",
        best_ratio=0.5,
        worst_ratio=0.5,
        known_first=True,
    ):

        with self.model_ratio(best=best_ratio):
            fris_best_models, node_types = self.fris(radius, mode, subwrap=subwrap)

        with self.model_ratio(worst=worst_ratio):
            fris_worst_models, node_types = self.fris(radius, mode, subwrap=subwrap)

        if known_first:
            sorted_type_list = utils.nodes_list_sorting_on_off_unknown()
            fris_best_models = utils.nodes_edges_utils.sort_by_mapping_lists(
                node_types, sorted_type_list, fris_best_models, axis=1
            )
            fris_worst_models = utils.nodes_edges_utils.sort_by_mapping_lists(
                node_types, sorted_type_list, fris_worst_models, axis=1
            )
            node_types = sorted_type_list

        cmap = plt.cm.Greys_r

        fris = np.stack((fris_best_models, fris_worst_models), axis=1)
        fig, ax, C = plots.plots.violin_groups(
            fris.T,
            node_types[:],
            rotation=90,
            scatter=False,
            fontsize=6,
            figsize=[10, 1],
            width=0.7,
            scatter_edge_color="white",
            scatter_radius=5,
            scatter_edge_width=0.25,
            cdist=100,
            cmap=cmap,
            showmedians=True,
            showmeans=False,
            violin_marker_lw=0.25,
            legend=[
                f"best {best_ratio:.0%} models",
                f"worst {worst_ratio:.0%} models",
            ],
            legend_kwargs=dict(
                fontsize=5,
                markerscale=10,
                loc="lower left",
                bbox_to_anchor=(0.88, 0.9),
            ),
        )
        ax.set_ylim(-1, 1)
        # dvs.plots.rm_spines(ax, ('left',))
        ax.grid(False)

        plt_utils.trim_axis(ax)
        plt_utils.set_spine_tick_params(
            ax,
            tickwidth=0.5,
            ticklength=3,
            ticklabelpad=2,
            labelsize=6,
            spinewidth=0.5,
        )
        plt_utils.boldify_labels(self.output_node_types, ax)
        ax = plt_utils.flash_response_color_labels(ax)
        plt_utils.patch_type_texts(ax)
        ax.hlines(
            0,
            *ax.get_xlim(),
            linewidth=0.5,
            linestyles="dashed",
            color="0.5",
            zorder=0,
        )
        return fig, ax

    def biases(self, cmap=plt.cm.viridis, mode="best"):

        nodes, bias = self.get_biases(mode=mode)

        fig, ax, C = plots.violin_groups(
            bias.T[:, None],
            nodes,
            rotation=90,
            scatter=False,
            fontsize=6,
            figsize=[10, 1],
            width=0.7,
            scatter_edge_color="white",
            scatter_radius=5,
            scatter_edge_width=0.25,
            cstart=64,
            cdist=0,
            cmap=cmap,
            showmedians=True,
            showmeans=False,
            violin_marker_lw=0.25,
            violin_alpha=1,
            violin_marker_color="k",
        )

        # dvs.plots.rm_spines(ax, ('left',))\
        ax.grid(False)

        # for tick in ax.xaxis.get_major_ticks(): if
        #     tick.label1.get_text().startswith("T"):
        #     tick.label1.set_weight("bold")

        # dvs.plots.plt_utils.trim_axis(ax)
        ax.tick_params(axis="both", width=0.5, length=3, pad=2, labelsize=6)
        [i.set_linewidth(0.5) for i in ax.spines.values()]

        # ax.set_yscale("symlog")
        ax.set_ylabel(r"$V_{t_i}^\mathrm{rest}$ (a.u.)")
        return fig, ax, bias

    def biases_paper_si(
        self,
        model_scatter=False,
        mode="best",  # initial or parameters of best checkpoint
        validation_subwrap="original_validation_v2",
    ):
        fig, ax, biases = self.biases(plt.cm.viridis_r, mode=mode)

        if model_scatter:
            task_error = self.task_error(validation_subwrap=validation_subwrap)
            colors, cmap, norm = (
                task_error.colors,
                task_error.cmap,
                task_error.norm,
            )
            model_indices = np.arange(len(self))[::-1]

            xticks = ax.get_xticks()
            for index in model_indices:
                ax.scatter(xticks, biases[index], s=3, c=colors[index])
        ax.set_title("Resting potentials", y=0.91, fontsize=8)
        return fig, ax

    def get_cell_type_params(
        self,
        cell_type,
        param_name,
        validation_subwrap,
        loss_fn=None,
        mode="best",
    ):
        def not_implemented_fn(*args, **kwargs):
            raise NotImplementedError(
                f"no getter function implemented for {param_name}"
            )

        get_param_fns = {
            "nodes_time_const": self.get_time_constants,
            "nodes_bias": self.get_biases,
            "edges_syn_strength": self.get_syn_strength,
            "edges_syn_count": self.get_syn_count,
            "edges_sign": not_implemented_fn,
        }
        get_param_fn = get_param_fns[param_name]
        keys, params = get_param_fn(mode=mode, validation_subwrap=validation_subwrap)

        if "edges" in param_name:
            # then variables should describe edges in List[Tuple[str]]
            keys = np.array([f"{source}->{target}" for source, target in keys])

        # filter params and variables
        keys_pattern = re.compile(cell_type)
        if "edges" in param_name:
            keys_pattern = re.compile(f".*->{cell_type}")
        index = np.array([i for i, v in enumerate(keys) if keys_pattern.match(v)])
        params = params[:, index]
        keys = keys[index]
        return keys, params

    def plot_parameter_marginals(
        self,
        param_name,
        max_per_ax=20,
        colors=None,
        cmap=plt.cm.viridis_r,
        best_ratio=1,
        worst_ratio=None,
        initialization=True,
        optimized=True,
        bold_output_type_labels=False,
        pretty_title=True,
        ylabel_param=True,
        fontsize=5,
        scatter_all=True,
        scatter_best=True,
        violin_width=0.7,
        validation_subwrap="original_validation_v2",
        legend=None,
        uniform=[-0.35, 0.35],
        cell_type=None,
        figwidth=10,
        fig=None,
        axes=None,
        ylabel_offset=0.2,
        trim_axis=True,
        ylim=None,
        **kwargs,
    ):
        """Generic method to plot the network parameter marginals.

        param_name: can currently be nodes_time_const, nodes_bias,
            edges_syn_strength, edges_syn_count.
        """

        def not_implemented_fn(*args, **kwargs):
            raise NotImplementedError(
                f"no getter function implemented for {param_name}"
            )

        get_param_fns = {
            "nodes_time_const": self.get_time_constants,
            "nodes_bias": self.get_biases,
            "edges_syn_strength": self.get_syn_strength,
            "edges_syn_count": self.get_syn_count,
            "edges_sign": not_implemented_fn,
        }

        get_param_fn = get_param_fns[param_name]

        data = {}
        best_indices = {}
        best_colors = {}

        if best_ratio is not None:
            with self.model_ratio(best=best_ratio):
                if initialization:
                    variables, initial_param_best_models = get_param_fn(mode="initial")
                    data[
                        f"initial parameter of the best {best_ratio:.0%} models"
                    ] = initial_param_best_models
                if optimized:
                    variables, best_param_best_models = get_param_fn(mode="best")
                    data[
                        f"task-optimized parameter of the best {best_ratio:.0%} models"
                    ] = best_param_best_models
                best_indices["best"] = self.best_index(
                    validation_subwrap=validation_subwrap
                )
                best_colors["best"] = self.best_color(
                    validation_subwrap=validation_subwrap
                )

        if worst_ratio is not None:
            with self.model_ratio(worst=worst_ratio):
                if initialization:
                    variables, initial_param_worst_models = get_param_fn(mode="initial")
                    data[
                        f"initial parameter  of the worst {best_ratio:.0%} models"
                    ] = initial_param_worst_models
                if optimized:
                    variables, best_param_worst_models = get_param_fn(mode="best")
                    data[
                        f"task-optimized parameter of the worst {worst_ratio:.0%} models"
                    ] = best_param_worst_models
                best_indices["worst"] = self.best_index(
                    validation_subwrap=validation_subwrap
                )
                best_colors["worst"] = self.best_color(
                    validation_subwrap=validation_subwrap
                )

        if "edges" in param_name:
            # then variables should describe edges in List[Tuple[str]]
            variables = np.array(
                [f"{source}->{target}" for source, target in variables]
            )

        groups = list(data.keys())
        params = np.stack(list(data.values()), axis=1)

        if cell_type is not None:
            # filter params and variables
            variable_pattern = re.compile(
                f"^{cell_type}$"
            )  # ^ is begin of str, $ end of str
            if "edges" in param_name:
                variable_pattern = re.compile(
                    f".*->\\b{cell_type}\\b"
                )  # \b is word boundary (escaped with \\b)
            index = np.array(
                [i for i, v in enumerate(variables) if variable_pattern.match(v)]
            )
            params = params[:, :, index]
            variables = variables[index]

        n_variables = len(variables)
        if max_per_ax is None:
            max_per_ax = n_variables
        max_per_ax = min(max_per_ax, n_variables)
        n_axes = int(n_variables / max_per_ax)
        max_per_ax += int(np.ceil((n_variables % max_per_ax) / n_axes))

        # breakpoint()
        fig, axes, _ = plt_utils.get_axis_grid(
            gridheight=n_axes,
            gridwidth=1,
            figsize=[figwidth, n_axes * 1.2],
            hspace=1,
            alpha=0,
            fig=fig,
            axes=axes,
        )

        for i in range(n_axes):

            _params = params.T[i * max_per_ax : (i + 1) * max_per_ax]
            _variables = variables[i * max_per_ax : (i + 1) * max_per_ax]

            fig, ax, C = plots.plots.violin_groups(
                _params,
                _variables,
                rotation=90,
                scatter=False,
                fontsize=fontsize,
                width=violin_width,
                scatter_edge_color="white",
                scatter_radius=5,
                scatter_edge_width=0.25,
                cdist=100,
                colors=colors,
                cmap=cmap,
                showmedians=True,
                showmeans=False,
                violin_marker_lw=0.25,
                legend=(
                    (legend if legend is not None else list(data.keys()))
                    if i == 0
                    else None
                ),
                legend_kwargs=dict(
                    fontsize=5,
                    markerscale=10,
                    loc="lower left",
                    bbox_to_anchor=(0.75, 0.75),
                ),
                fig=fig,
                ax=axes[i],
                **kwargs,
            )

            n_variables, n_groups, _ = _params.shape
            violin_locations, _ = plots.get_violin_x_locations(
                n_groups, n_variables, violin_width
            )
            # breakpoint()
            for group in range(n_groups):

                group_label = "best" if "best" in groups[group] else "worst"
                _scatter_best = scatter_best if group_label == "best" else False
                self._scatter_on_violins(
                    data=_params[:, group].T,
                    ax=axes[i],
                    validation_subwrap=validation_subwrap,
                    scatter_best=_scatter_best,
                    scatter_all=scatter_all,
                    xticks=violin_locations[group],
                    best_index=best_indices[group_label],
                    best_color=best_colors[group_label],
                    uniform=uniform,
                )

            if ylim is not None:
                ax.set_ylim(*ylim)

            # ax.set_ylim(-0.05, 1) ax.set_ylim(-0.05, 1)
            # dvs.plots.rm_spines(ax, ('left',))
            ax.grid(False)

            if bold_output_type_labels:
                plt_utils.boldify_labels(self.output_node_types, ax)

            if trim_axis:
                plt_utils.trim_axis(ax, yaxis=False)
            plt_utils.set_spine_tick_params(
                ax,
                tickwidth=0.5,
                ticklength=3,
                ticklabelpad=2,
                labelsize=fontsize,
                spinewidth=0.5,
            )

            plt_utils.patch_type_texts(ax)

        if ylabel_param:
            lefts, bottoms, rights, tops = np.array(
                [ax.get_position().extents for ax in axes]
            ).T
            fig.text(
                lefts.min() - ylabel_offset * lefts.min(),
                (tops.max() - bottoms.min()) / 2,
                param_to_tex[param_name],
                rotation=90,
                fontsize=fontsize,
                ha="right",
                va="center",
            )

        if pretty_title:
            axes[0].set_title(pretty_param_label[param_name], y=0.91, fontsize=fontsize)

        return fig, axes, params

    def biases_paper_si_v2(
        self,
        max_per_ax=20,
        cmap=plt.cm.viridis_r,
        best_ratio=0.2,
        worst_ratio=0.2,
        initialization=True,
        optimized=True,
    ):

        data = {}

        with self.model_ratio(best=best_ratio):
            if initialization:
                nodes, initial_bias_best_models = self.get_biases(mode="initial")
                data[
                    f"initialization of the best {best_ratio:.0%} models"
                ] = initial_bias_best_models
            if optimized:
                nodes, best_bias_best_models = self.get_biases(mode="best")
                data[
                    f"optimal parameter of the best {best_ratio:.0%} models"
                ] = best_bias_best_models

        with self.model_ratio(worst=worst_ratio):
            if initialization:
                nodes, initial_bias_worst_models = self.get_biases(mode="initial")
                data[
                    f"initialization of the worst {best_ratio:.0%} models"
                ] = initial_bias_worst_models
            if optimized:
                nodes, best_bias_worst_models = self.get_biases(mode="best")
                data[
                    f"optimal parameter of the worst {worst_ratio:.0%} models"
                ] = best_bias_worst_models

        biases = np.stack(list(data.values()), axis=1)
        n_nodes = len(nodes)
        if max_per_ax is None:
            max_per_ax = n_nodes
        n_axes = int(n_nodes / max_per_ax)
        max_per_ax += int(np.ceil((n_nodes % max_per_ax) / n_axes))

        # breakpoint()
        fig, axes, _ = plt_utils.get_axis_grid(
            gridheight=n_axes, gridwidth=1, figsize=[10, n_axes * 1.2], hspace=1
        )

        for i in range(n_axes):
            fig, ax, C = plots.plots.violin_groups(
                biases.T[i * max_per_ax : (i + 1) * max_per_ax],
                nodes[i * max_per_ax : (i + 1) * max_per_ax],
                rotation=90,
                scatter=False,
                fontsize=6,
                figsize=[10, 1],
                width=0.7,
                scatter_edge_color="white",
                scatter_radius=5,
                scatter_edge_width=0.25,
                cdist=100,
                cmap=cmap,
                showmedians=True,
                showmeans=False,
                violin_marker_lw=0.25,
                legend=list(data.keys()) if i == 0 else None,
                legend_kwargs=dict(
                    fontsize=5,
                    markerscale=10,
                    loc="lower left",
                    bbox_to_anchor=(0.85, 0.75),
                ),
                fig=fig,
                ax=axes[i],
            )
            # ax.set_ylim(-0.05, 1) ax.set_ylim(-0.05, 1)
            # dvs.plots.rm_spines(ax, ('left',))
            ax.grid(False)
            plots.plt_utils.trim_axis(ax)
            ax.tick_params(axis="both", width=0.5, length=3, pad=2, labelsize=6)
            [i.set_linewidth(0.5) for i in ax.spines.values()]

            for tick in ax.xaxis.get_major_ticks():
                if tick.label1.get_text().startswith("T"):
                    tick.label1.set_weight("bold")

        axes[0].set_title("resting potentials", y=0.91, fontsize=8)
        axes[0].set_ylabel(param_to_tex["nodes_bias"], fontsize=5)

        return fig, axes, biases

    def time_constants(self, cmap=plt.cm.viridis_r, mode="best"):
        nodes, time_constants = self.get_time_constants(mode=mode)
        fig, ax, C = plots.violin_groups(
            time_constants.T[:, None],
            nodes,
            rotation=90,
            scatter=False,
            fontsize=6,
            figsize=[10, 1],
            width=0.7,
            scatter_edge_color="white",
            scatter_radius=5,
            scatter_edge_width=0.25,
            cstart=64,
            cdist=0,
            cmap=cmap,
            showmedians=True,
            showmeans=False,
            violin_marker_lw=0.25,
            violin_alpha=1,
            violin_marker_color="k",
        )

        # dvs.plots.rm_spines(ax, ('left',))\
        ax.grid(False)

        # for tick in ax.xaxis.get_major_ticks(): if
        #     tick.label1.get_text().startswith("T"):
        #     tick.label1.set_weight("bold")

        # dvs.plots.plt_utils.trim_axis(ax)
        ax.tick_params(axis="both", width=0.5, length=3, pad=2, labelsize=6)
        [i.set_linewidth(0.5) for i in ax.spines.values()]

        # ax.set_yscale("symlog")
        ax.set_ylabel(r"$\tau_{t_i}$ (s)")
        return fig, ax, time_constants

    def time_constants_paper_si(
        self,
        model_scatter=False,
        mode="best",
        validation_subwrap="original_validation_v2",
    ):
        fig, ax, time_constants = self.time_constants(mode=mode)

        if model_scatter:
            task_error = self.task_error(validation_subwrap=validation_subwrap)
            colors, cmap, norm = (
                task_error.colors,
                task_error.cmap,
                task_error.norm,
            )

            model_indices = np.arange(len(self))[::-1]

            xticks = ax.get_xticks()
            for index in model_indices:
                ax.scatter(xticks, time_constants[index], s=3, c=colors[index])

        ax.set_title("Time constants", y=0.91, fontsize=8)
        return fig, ax

    def time_constants_paper_si_v2(
        self,
        max_per_ax=20,
        cmap=plt.cm.viridis_r,
        best_ratio=0.5,
        worst_ratio=0.5,
    ):

        with self.model_ratio(best=best_ratio):
            nodes, initial_tc_best_models = self.get_time_constants(mode="initial")
            nodes, best_tc_best_models = self.get_time_constants(mode="best")

        with self.model_ratio(worst=worst_ratio):
            nodes, initial_tc_worst_models = self.get_time_constants(mode="initial")
            nodes, best_tc_worst_models = self.get_time_constants(mode="best")

        time_constants = np.stack(
            (
                initial_tc_best_models,
                initial_tc_worst_models,
                best_tc_best_models,
                best_tc_worst_models,
            ),
            axis=1,
        )
        n_nodes = len(nodes)
        n_axes = int(n_nodes / min(n_nodes, max_per_ax))
        max_per_ax += int(np.ceil((n_nodes % max_per_ax) / n_axes))

        # breakpoint()
        fig, axes, _ = plt_utils.get_axis_grid(
            gridheight=n_axes, gridwidth=1, figsize=[10, n_axes * 1.2], hspace=1
        )

        for i in range(n_axes):
            fig, ax, C = plots.plots.violin_groups(
                time_constants.T[i * max_per_ax : (i + 1) * max_per_ax],
                nodes[i * max_per_ax : (i + 1) * max_per_ax],
                rotation=90,
                scatter=False,
                fontsize=6,
                figsize=[10, 1],
                width=0.7,
                scatter_edge_color="white",
                scatter_radius=5,
                scatter_edge_width=0.25,
                cdist=100,
                cmap=cmap,
                showmedians=True,
                showmeans=False,
                violin_marker_lw=0.25,
                legend=[
                    f"initial - best {best_ratio:.2%} models",
                    f"initial - worst {worst_ratio:.0%} models",
                    f"best - best {best_ratio:.0%} models",
                    f"best - worst {worst_ratio:.0%} models",
                ]
                if i == 0
                else None,
                legend_kwargs=dict(
                    fontsize=5,
                    markerscale=10,
                    loc="lower left",
                    bbox_to_anchor=(0.85, 0.75),
                ),
                fig=fig,
                ax=axes[i],
            )
            # ax.set_ylim(-0.05, 1) ax.set_ylim(-0.05, 1)
            # dvs.plots.rm_spines(ax, ('left',))
            ax.grid(False)
            plots.plt_utils.trim_axis(ax)
            ax.tick_params(axis="both", width=0.5, length=3, pad=2, labelsize=6)
            [i.set_linewidth(0.5) for i in ax.spines.values()]

            for tick in ax.xaxis.get_major_ticks():
                if tick.label1.get_text().startswith("T"):
                    tick.label1.set_weight("bold")

        axes[0].set_title("time constants", y=0.91, fontsize=8)
        axes[0].set_ylabel(param_to_tex["nodes_time_const"], fontsize=5)

        return fig, ax

    def syn_strength(
        self, max_per_ax=100, cmap=plt.cm.plasma_r, figwidth=10, mode="best"
    ):
        edges, syn_strength = self.get_syn_strength(target_sorted=True, mode=mode)
        edges = [f"{source}->{target}" for source, target in edges]

        # n_models, n_edges = syn_strength.shape
        # n_axes = int(n_edges / max_per_ax)
        # max_per_ax += int(
        #     np.ceil((n_edges % max_per_ax) / min(n_edges, n_axes))
        # )

        n_edges = len(edges)
        n_axes = int(n_edges / min(n_edges, max_per_ax))
        max_per_ax += int(np.ceil((n_edges % max_per_ax) / n_axes))

        # breakpoint()
        fig, axes, _ = plt_utils.get_axis_grid(
            gridheight=n_axes, gridwidth=1, figsize=[10, n_axes * 1.2], hspace=1
        )

        for i in range(n_axes):
            fig, ax, C = plots.violin_groups(
                syn_strength.T[i * max_per_ax : (i + 1) * max_per_ax, None],
                edges[i * max_per_ax : (i + 1) * max_per_ax],
                rotation=90,
                scatter=False,
                fontsize=5,
                figsize=[10, 1],
                width=0.7,
                scatter_edge_color="white",
                scatter_radius=5,
                scatter_edge_width=0.25,
                cstart=64,
                cdist=0,
                cmap=cmap,
                showmedians=True,
                showmeans=False,
                violin_marker_lw=0.25,
                violin_alpha=1,
                fig=fig,
                ax=axes[i],
            )

            # dvs.plots.rm_spines(ax, ('left',))
            ax.grid(False)

            # dvs.plots.plt_utils.trim_axis(ax)
            ax.tick_params(axis="both", width=0.5, length=3, pad=2, labelsize=5)
            [i.set_linewidth(0.5) for i in ax.spines.values()]
            ax.set_facecolor("none")

        lefts, bottoms, rights, tops = np.array(
            [ax.get_position().extents for ax in axes]
        ).T
        fig.text(
            lefts.min() - 0.2 * lefts.min(),
            (tops.max() - bottoms.min()) / 2,
            r"$\alpha_{t_it_j}$ (a.u.)",
            rotation=90,
            fontsize=5,
            ha="right",
            va="center",
        )
        return fig, axes, (edges, syn_strength)

    def syn_strength_paper_si(
        self,
        model_scatter=False,
        mode="best",
        validation_subwrap="original_validation_v2",
    ):
        fig, axes, (edges, syn_strength) = self.syn_strength(
            120, cmap=plt.cm.viridis_r, mode=mode
        )

        if model_scatter:
            task_error = self.task_error(validation_subwrap=validation_subwrap)
            colors, cmap, norm = (
                task_error.colors,
                task_error.cmap,
                task_error.norm,
            )

            model_indices = np.arange(len(self))[::-1]
            ticks_per_ax = max([len(ax.get_xticks()) for ax in axes])
            for i, ax in enumerate(axes):
                xticks = ax.get_xticks()
                syn_strength_indices = (i * ticks_per_ax + xticks).astype(int)
                for index in model_indices:
                    ax.scatter(
                        xticks,
                        syn_strength[index, syn_strength_indices],
                        s=0.25,
                        c=colors[index],
                    )

        axes[0].set_title("Filter scaling factors", y=0.91, fontsize=8)
        return fig, axes

    def syn_strength_paper_si_v2(
        self,
        max_per_ax=40,
        cmap=plt.cm.viridis_r,
        best_ratio=0.5,
        worst_ratio=0.5,
    ):

        with self.model_ratio(best=best_ratio):
            (
                edges,
                initial_syn_strength_best_models,
            ) = self.get_syn_strength(target_sorted=True, mode="initial")
            edges, best_syn_strength_best_models = self.get_syn_strength(
                target_sorted=True, mode="best"
            )

        with self.model_ratio(worst=worst_ratio):
            (
                edges,
                best_syn_strength_worst_models,
            ) = self.get_syn_strength(target_sorted=True, mode="best")
            (
                edges,
                initial_syn_strength_worst_models,
            ) = self.get_syn_strength(target_sorted=True, mode="initial")

        syn_strength = np.stack(
            (
                initial_syn_strength_best_models,
                initial_syn_strength_worst_models,
                best_syn_strength_best_models,
                best_syn_strength_worst_models,
            ),
            axis=1,
        )

        edges = [f"{source}->{target}" for source, target in edges]

        n_edges = len(edges)
        n_axes = int(n_edges / min(n_edges, max_per_ax))
        max_per_ax += int(np.ceil((n_edges % max_per_ax) / n_axes))

        # breakpoint()
        fig, axes, _ = plt_utils.get_axis_grid(
            gridheight=n_axes, gridwidth=1, figsize=[10, n_axes * 2], hspace=1
        )

        for i in range(n_axes):
            fig, ax, C = plots.plots.violin_groups(
                syn_strength.T[i * max_per_ax : (i + 1) * max_per_ax],
                edges[i * max_per_ax : (i + 1) * max_per_ax],
                rotation=90,
                scatter=False,
                fontsize=6,
                figsize=[10, 1],
                width=0.7,
                scatter_edge_color="white",
                scatter_radius=5,
                scatter_edge_width=0.25,
                cdist=100,
                cmap=cmap,
                showmedians=True,
                showmeans=False,
                violin_marker_lw=0.25,
                legend=[
                    f"initial - best {best_ratio:.2%} models",
                    f"initial - worst {worst_ratio:.0%} models",
                    f"best - best {best_ratio:.0%} models",
                    f"best - worst {worst_ratio:.0%} models",
                ]
                if i == 0
                else None,
                legend_kwargs=dict(
                    fontsize=5,
                    markerscale=10,
                    loc="lower left",
                    bbox_to_anchor=(0.85, 0.75),
                ),
                fig=fig,
                ax=axes[i],
            )
            # ax.set_ylim(-0.05, 1) ax.set_ylim(-0.05, 1)
            # dvs.plots.rm_spines(ax, ('left',))
            ax.grid(False)
            plots.plt_utils.trim_axis(ax)
            ax.tick_params(axis="both", width=0.5, length=3, pad=2, labelsize=6)
            [i.set_linewidth(0.5) for i in ax.spines.values()]

            # for tick in ax.xaxis.get_major_ticks(): if
            #     tick.label1.get_text().startswith("T"):
            #     tick.label1.set_weight("bold")

        axes[0].set_title("filter scaling factors", y=0.91, fontsize=8)
        axes[0].set_ylabel(param_to_tex["edges_syn_strength"], fontsize=5)
        return fig, ax

    def syn_count(self, max_per_ax=100, cmap=plt.cm.plasma_r, figwidth=10, mode="best"):
        edges, syn_count = self.get_syn_count(target_sorted=True, mode=mode)
        edges = [f"{source}->{target}" for source, target in edges]

        n_models, n_edges = syn_count.shape
        n_axes = int(n_edges / max_per_ax)
        max_per_ax += int(np.ceil((n_edges % max_per_ax) / n_axes))

        # breakpoint()
        fig, axes, _ = plt_utils.get_axis_grid(
            gridheight=n_axes, gridwidth=1, figsize=[10, n_axes * 1.2], hspace=1
        )

        for i in range(n_axes):
            fig, ax, C = plots.violin_groups(
                syn_count.T[i * max_per_ax : (i + 1) * max_per_ax, None],
                edges[i * max_per_ax : (i + 1) * max_per_ax],
                rotation=90,
                scatter=False,
                fontsize=5,
                figsize=[10, 1],
                width=0.7,
                scatter_edge_color="white",
                scatter_radius=5,
                scatter_edge_width=0.25,
                cstart=64,
                cdist=0,
                cmap=cmap,
                showmedians=True,
                showmeans=False,
                violin_marker_lw=0.25,
                violin_alpha=1,
                fig=fig,
                ax=axes[i],
            )

            # dvs.plots.rm_spines(ax, ('left',))
            ax.grid(False)

            # dvs.plots.plt_utils.trim_axis(ax)
            ax.tick_params(axis="both", width=0.5, length=3, pad=2, labelsize=5)
            [i.set_linewidth(0.5) for i in ax.spines.values()]
            ax.set_facecolor("none")

        lefts, bottoms, rights, tops = np.array(
            [ax.get_position().extents for ax in axes]
        ).T
        fig.text(
            lefts.min() - 0.2 * lefts.min(),
            (tops.max() - bottoms.min()) / 2,
            r"$\mathrm{N}_{t_it_j\DeltaU\DeltaV}$ (a.u.)",
            rotation=90,
            fontsize=5,
            ha="right",
            va="center",
        )
        return fig, axes, (edges, syn_count)

    def flash_responses(
        self,
        node_type,
        z_score=True,
        timelim=(1, 2),
        figsize=[2, 0.8],
        fontsize=5,
        colors=None,
        cbar=True,
        fig=None,
        axes=None,
        validation_subwrap="original_validation_v2",
    ):

        if colors is None:
            task_error = self.task_error(validation_subwrap=validation_subwrap)
            colors, cmap, norm = (
                task_error.colors,
                task_error.cmap,
                task_error.norm,
            )
        else:
            cbar = False

        time, fr_off, fr_on, mask = self.flash_response_data(
            node_type, z_score=z_score, timelim=timelim, stack=False
        )

        if fig is None or axes is None:
            fig, axes, _ = plt_utils.get_axis_grid(
                gridwidth=2,
                gridheight=1,
                figsize=figsize,
                fontsize=fontsize,
                wspace=0.5,
            )

        for i in range(len(self)):
            axes[0].plot(time, fr_off[len(self) - i - 1], c=colors[len(self) - i - 1])
            axes[1].plot(time, fr_on[len(self) - i - 1], c=colors[len(self) - i - 1])
        ylims = plt_utils.get_lims((fr_off, fr_on), 0.1)
        _zscored = "\n(standardized)" if z_score else ""
        axes[0].set_ylabel(f"activity (a.u.){_zscored}", fontsize=fontsize)
        flash = self[0].flashes_dataset[1].cpu()
        central_flash = flash[:, 360]
        plt_utils.plot_stim_contour(time, central_flash, axes[0])

        flash = self[0].flashes_dataset[3].cpu()
        central_flash = flash[:, 360]
        plt_utils.plot_stim_contour(time, central_flash, axes[1])
        axes[0].set_xlabel("time (s)", fontsize=fontsize, labelpad=-0.5)
        if cbar:
            plt_utils.add_colorbar(
                fig,
                axes[1],
                cmap=cmap,
                norm=norm,
                label="task error",
                fontsize=fontsize,
            )
        fig.suptitle(f"{node_type} flash responses", fontsize=fontsize)
        axes[0].set_ylim(*ylims)
        axes[1].set_ylim(*ylims)
        return fig, axes, (fr_off, fr_on)

    # def umap_clustering_pre_recorded(node_type, subwrap, ): def
    # preferred_directions( self, node_types=["T4a", "T4b", "T4c", "T4d", "T5a",
    # "T5b", "T5c", "T5d"], round_directions=[0, 90, 180, 270, 360],
    # ):
    #     node_types = node_types or self.node_types_sorted
    #     self.init_movingbar()

    #     def pc_to_intensity(pc): if pc == 1: intensity = 1 elif pc == -1:
    #         intensity = 0 elif pc == 0: intensity = [0, 1] else: raise
    #         ValueError(f"preferred contrast {pc} invalid") return intensity

    #     intensities = { node_type: pc_to_intensity(pc) for node_type, pc in
    #         groundtruth_utils.preferred_contrasts.items()
    #     }

    #     PDs = [] for nnv in self.values(): _PDs = {} for node_type in
    #     node_types: pd = utils.round_angles(
    #     nnv.movingbar.preferred_direction( node_type,
    #     intensity=intensities[node_type]
    #                 ),
    #                 round_directions,
    #             )
    #             _PDs[node_type] = pd
    #         PDs.append(_PDs)
    #     return PDs

    def correct_pd_mask(self, t4_or_t5):
        def filter(adict, fkey):
            return {key: val for key, val in adict.items() if fkey not in key}

        pds = self.preferred_directions()

        if t4_or_t5 == "T4":
            mask = [
                True
                if filter(pd, fkey="T5")
                == filter(groundtruth_utils.preferred_directions, fkey="T5")
                else False
                for pd in pds
            ]
        elif t4_or_t5 == "T5":
            mask = [
                True
                if filter(pd, fkey="T4")
                == filter(groundtruth_utils.preferred_directions, fkey="T4")
                else False
                for pd in pds
            ]
        else:
            raise ValueError(f"{t4_or_t5} must be one on 'T4', 'T5'")
        return np.array(mask)

    def training_and_EMD_tuning_overview(
        self,
        comment="",
        validation_subwrap="original_validation_v2",
        fig=None,
        axes=None,
        fontsize=6,
    ):
        """Overview of training loss curves and EMD + TMY tuning."""

        self.init_movingbar(subwrap="movingedge_chkpt_best_v4")
        self.init_flashes(subwrap="flashes")

        if fig is None or axes is None:
            fig, axes, _ = plt_utils.get_axis_grid(
                gridwidth=4, gridheight=3, as_matrix=True, figsize=[16, 9]
            )

        comment = comment or self[0].tnn.meta.spec.comment

        fig.suptitle(
            "\n".join(
                textwrap.wrap(
                    f"ensemble {self.name}"
                    + f" ({len(self)} models) - "
                    + (comment or self[0].tnn.spec.comment),
                    60,
                )
            ),
            fontsize=fontsize * 3,
            va="bottom",
        )

        # [0, 0] - training loss
        self.training_loss(
            validation_subwrap=validation_subwrap,
            fig=fig,
            ax=axes[0, 0],
            grid=False,
            cbar=False,
        )
        axes[0, 0].set_title("training loss", fontsize=fontsize)

        # [1, 0] - validation loss
        self.validation_loss(
            validation_subwrap=validation_subwrap,
            fig=fig,
            ax=axes[1, 0],
            grid=False,
            cbar=False,
        )
        axes[1, 0].set_title(
            f"validation loss: {validation_subwrap}", fontsize=fontsize
        )

        # [2, 0] - validation hist
        self.loss_histogramm(
            validation_subwrap=validation_subwrap,
            fig=fig,
            ax=axes[2, 0],
            grid=False,
        )
        axes[2, 0].set_title(
            f"minimal validation loss histogram: {validation_subwrap}",
            fontsize=fontsize,
        )

        with self.sort("min", validation_subwrap=validation_subwrap):
            nnv = self[0]

            # [0, 1] - T4 best model tuning
            _axes = plt_utils.divide_axis_to_grid(
                axes[0, 1],
                matrix=[[0, 1, 2, 3]],
                wspace=0,
                hspace=0,
                projection="polar",
            )
            nnv.paper_direction_tuning(
                "T4", fig=fig, axes=[ax for ax in _axes.values()]
            )

            # [1, 1] - T5 best model tuning
            _axes = plt_utils.divide_axis_to_grid(
                axes[1, 1],
                matrix=[[0, 1, 2, 3]],
                wspace=0,
                hspace=0,
                projection="polar",
            )
            nnv.paper_direction_tuning(
                "T5", fig=fig, axes=[ax for ax in _axes.values()]
            )

            # [2, 1] - TmY best model tuning
            _axes = plt_utils.divide_axis_to_grid(
                axes[2, 1],
                matrix=[[0, 1, 2, 3]],
                wspace=0,
                hspace=0,
                projection="polar",
            )
            nnv.paper_direction_tuning(
                "TmY", fig=fig, axes=[ax for ax in _axes.values()]
            )

        axes[0, 1].set_title("best model tuning", fontsize=fontsize)

        with self.sort(mode="min", validation_subwrap=validation_subwrap):
            with self.model_items(list(range(5))):

                # [0, 2] - T4 5 best model tuning
                _axes = plt_utils.divide_axis_to_grid(
                    axes[0, 2],
                    matrix=[[0, 1, 2, 3]],
                    wspace=0,
                    hspace=0,
                    projection="polar",
                )
                self.paper_direction_tuning(
                    "T4", fig=fig, axes=[ax for ax in _axes.values()]
                )

                # [1, 2] - T5 5 best model tuning
                _axes = plt_utils.divide_axis_to_grid(
                    axes[1, 2],
                    matrix=[[0, 1, 2, 3]],
                    wspace=0,
                    hspace=0,
                    projection="polar",
                )
                self.paper_direction_tuning(
                    "T5", fig=fig, axes=[ax for ax in _axes.values()]
                )

                # [2, 2] - TmY 5 best model tuning
                _axes = plt_utils.divide_axis_to_grid(
                    axes[2, 2],
                    matrix=[[0, 1, 2, 3]],
                    wspace=0,
                    hspace=0,
                    projection="polar",
                )
                self.paper_direction_tuning(
                    "TmY", fig=fig, axes=[ax for ax in _axes.values()]
                )

        axes[0, 2].set_title("5 best model tuning", fontsize=fontsize)

        # [0, 3] - T4 all model tuning
        _axes = plt_utils.divide_axis_to_grid(
            axes[0, 3],
            matrix=[[0, 1, 2, 3]],
            wspace=0,
            hspace=0,
            projection="polar",
        )
        self.paper_direction_tuning("T4", fig=fig, axes=[ax for ax in _axes.values()])

        # [1, 3] - T5 all model tuning
        _axes = plt_utils.divide_axis_to_grid(
            axes[1, 3],
            matrix=[[0, 1, 2, 3]],
            wspace=0,
            hspace=0,
            projection="polar",
        )
        self.paper_direction_tuning("T5", fig=fig, axes=[ax for ax in _axes.values()])

        # [2, 3] - TmY all model tuning
        _axes = plt_utils.divide_axis_to_grid(
            axes[2, 3],
            matrix=[[0, 1, 2, 3]],
            wspace=0,
            hspace=0,
            projection="polar",
        )
        self.paper_direction_tuning("TmY", fig=fig, axes=[ax for ax in _axes.values()])

        axes[0, 3].set_title("all model tuning", fontsize=fontsize)
        return fig

    def training_and_tuning_overview(
        self,
        comment="",
        fontsize=6,
        validation_subwrap="original_validation_v2",
        best_ratio=1.0,
    ):
        """Overview of training loss curves, EMD + TMY, DSI and FRI."""

        self.init_movingbar(subwrap="movingedge_chkpt_best_v4")
        self.init_flashes(subwrap="flashes")

        # creates an axis grid of 5 rows and four columns
        fig, axes, _ = plt_utils.get_axis_grid(
            gridwidth=4, gridheight=5, as_matrix=True, figsize=[16, 11]
        )
        self.training_and_EMD_tuning_overview(
            comment=comment,
            validation_subwrap=validation_subwrap,
            fig=fig,
            axes=axes[0:3],
            fontsize=fontsize,
        )

        fig.suptitle(
            "\n".join(
                textwrap.wrap(
                    f"ensemble {self.name}"
                    + f" ({len(self)} models) - "
                    + (comment or self[0].tnn.spec.comment),
                    60,
                )
            ),
            fontsize=fontsize * 3,
            y=0.95,
            va="bottom",
        )
        # [3, -] - FRI
        fri_ax = plots.plt_utils.merge_axes(fig, axes[3])
        with self.model_ratio(best=best_ratio):
            self.fris_paper_figure(
                validation_subwrap=validation_subwrap, fig=fig, ax=fri_ax
            )
        fri_ax.set_title(
            f"{best_ratio:.0%} best model flash response index",
            fontsize=fontsize,
            y=0.9,
        )
        fri_ax.set_ylabel("flash response index", fontsize=fontsize)

        # [4, _] - DSI
        dsi_axes = plt_utils.divide_axis_to_grid(
            plots.plt_utils.merge_axes(fig, axes[4]),
            matrix=[[0], [1]],
            wspace=0,
            hspace=0,
        )
        for ax in dsi_axes.values():
            ax.patch.set_alpha(0)
        with self.model_ratio(best=best_ratio):
            self.dsis_stacked_on_off(
                scatter_best=True,
                scatter_all=True,
                nonlinearity=True,
                validation_subwrap=validation_subwrap,
                fig=fig,
                axes=dsi_axes,
            )
        dsi_axes[0].set_title(
            f"{best_ratio:.0%} best model direction selectivity index",
            fontsize=fontsize,
            y=0.9,
        )
        dsi_axes[0].set_ylabel("direction selectivity index", fontsize=fontsize, y=0)


@dataclass
class TaskError:
    values: np.ndarray
    colors: np.ndarray
    cmap: Colormap
    norm: Normalize
    scalarmappable: cm.ScalarMappable


# ----------- NODES


def node_statistics(nns, param="time_const_trained", title="", **kwargs):
    """
    Plots node parameter statistics across several trained models.

    Args: nns (list) mode

    """

    nnv = NetworkViews(f"experiments/{nns[0]}")
    nodes = nnv.nodes.to_df()
    node_index = nodes[(nodes.u == 0) & (nodes.v == 0)].index
    params = nodes[param][node_index]
    all_params = np.zeros([len(nns), len(params)])
    all_params[0] = params

    for i, nn in enumerate(nns[1:], start=1):
        nnv = NetworkViews(f"experiments/{nns[i]}")
        all_params[i] = nodes[param][node_index]

    # title = title if title else param
    return (
        plots.param_stats(nodes.type[node_index], all_params, title=title, **kwargs),
        all_params,
    )  # fig, ax, label_text


# ----------- EDGES


def edge_statistics(nns, lognormal=False, **kwargs):
    means, coef_of_var, rfs = get_rf_statistics(nns, lognormal=lognormal)
    return filter_statistics(
        means,
        coef_of_var,
        [nt for nt in rfs[next(iter(rfs.keys()))].keys()],
        lognormal=lognormal,
        **kwargs,
    )


def filter_statistics(
    means,
    coef_of_var,
    nodes,
    title_left="Average Trained Synaptic Input Across Models",
    title_right="Variation Across Trials (σ/|μ|)",
    figsize=[20, 20],
    scale=10,
    wspace=0.3,
    fontsize=8,
    lognormal=False,
    **kwargs,
):
    """Plot two matrices of the matrices returned by
    plots.datastructs.get_rf_statistics."""

    fig, ax, _ = plt_utils.get_axis_grid([0, 1], scale=scale, figsize=figsize)

    midpoint = 1 if lognormal else 0

    ax1 = plots.heatmap_uniform(
        means,
        nodes,
        cmap=cm.get_cmap("seismic"),
        symlog=1e-5,
        fig=fig,
        ax=ax[0],
        fontsize=fontsize,
        title=title_left,
        **kwargs,
    )

    # coef_of_var = coef_of_var / coef_of_var.max()
    ax2 = plots.heatmap_uniform(
        coef_of_var,
        nodes,
        cmap=cm.get_cmap("Reds"),
        vmin=0.1,
        vmax=coef_of_var.max(),
        log=True,
        midpoint=None,
        fig=fig,
        ax=ax[1],
        fontsize=fontsize,
        title=title_right,
        **kwargs,
    )
    plt.subplots_adjust(wspace=wspace)
    return fig, [ax1, ax2]


def scaling_change(
    nns,
    title="Relative Change Averaged Across Models",
    fontsize=10,
    lognormal=True,
    **kwargs,
):
    means, _, rfs = get_rf_statistics(nns, lognormal=lognormal)
    means = np.ma.masked_not_equal(means, 0)
    midpoint = 0 if lognormal else 1
    return plots.heatmap_uniform(
        means,
        rfs[next(iter(rfs.keys()))].keys(),
        cmap=cm.get_cmap("seismic"),
        vmin=means.min(),
        vmax=means.max(),
        midpoint=midpoint,
        fontsize=fontsize,
        title=title,
        **kwargs,
    )


# -----  supporting definitions


class ReceptiveFields(Namespace):
    """Object storing the receptive fields in attribute style access.

    Args: edges (pd.DataFrame)

    Note: maybe become obsolete in favor of connectome wrap with RF indices. Or
    leverage it.
    """

    class Target(Namespace):
        pass

    class Source(Namespace):
        pass

    def __init__(self, edges):

        connections = edges.groupby(
            ["target_type", "source_type"], as_index=False
        ).first()

        targets = order_nodes_list(edges.target_type.unique())[0]

        u_cond = edges.target_u == 0
        v_cond = edges.target_v == 0
        edges = edges[u_cond & v_cond]

        for target in tqdm(targets, desc="Receptive Fields", leave=False):

            sources = connections[connections.target_type == target].source_type
            self[target] = self.Target()

            tt_cond = edges.target_type == target

            for source in sources:
                st_cond = edges.source_type == source
                _receptive_field = edges[st_cond & tt_cond]
                _rf_syn_count = _receptive_field["weight_trained"].values / (
                    _receptive_field["syn_strength_trained"].values + 1e-16
                )
                self[target].update(
                    {
                        f"{source}": self.Source(
                            param=_rf_syn_count,
                            sign=_receptive_field.sign.values,
                            u=_receptive_field.source_u.values,
                            v=_receptive_field.source_v.values,
                            index=_receptive_field.index.values,
                        )
                    }
                )


def get_rf_statistics(nns, lognormal=False):
    """Computes statistics over receptive fields of multiple trained networks.

    Args: nns (list of str): list of NetworkViews wraps in "root_dir /
        experiments / []". param (str): column of the edge dataframe normalize
        (str): another column of the edge dataframe. This divides the 'param'
        column. lognormal (bool): whether to take the logarithm of the receptive
        field values before computing the mean.

    Returns: array of shape (# node_types, # node_types): avg of filter means.
        array of shape (# node_types, # node_types): coefficient of variation
        over trained networks of filter means. dict: holding a ReceptiveFields
        object per nn in nns.
    """

    def signed_weight_to_log(weight):
        return np.log(np.abs(weight) + np.exp(-100))

        # weight = np.log(np.abs(weight)) return np.ma.masked_invalid(weight)

    # def log_coef_of_var(mean, std): return np.sqrt(np.exp(std**2) - 1)

    log_or_not = signed_weight_to_log if lognormal is True else (lambda weight: weight)
    # _coef_of_var = log_coef_of_var if lognormal else (lambda mean, std: (std /
    # (np.abs(mean) + 1e-30)))

    # Load all receptive fields into memory.
    rf_all = Namespace()
    for i, nn in enumerate(nns):
        nnv = NetworkViews(f"experiments/{nns[i]}")
        rf_all.update({f"{nn}": ReceptiveFields(nnv.edges.to_df())})

    node_types = order_nodes_list(rf_all[next(iter(nns))].keys())[0]

    # Compute means of the filters. Optionally of the log(|w|) of the filter
    # values. (presynaptic, postsynaptic, # trained networks) - (features,
    # features, observations)
    filter_means = np.zeros([len(node_types), len(node_types), len(nns)])

    for i, tgttyp in enumerate(node_types):
        for j, srctyp in enumerate(node_types):
            for k, nn in enumerate(nns):
                rf = rf_all[nn]
                target = rf[tgttyp]
                if srctyp in target:
                    filter_means[j, i, k] = log_or_not(
                        rf[tgttyp][srctyp].param
                    ).sum()  # Mean of log(w)
    # filter_means = np.ma.masked_invalid(filter_means) Assuming filter_means is
    # normal and not lognormal over trained networks, also after log transform,
    # i.e. cv = std/mu in either case.
    avg_of_filter_means = filter_means.mean(axis=2)
    std_of_filter_means = filter_means.std(axis=2)
    coef_of_var = std_of_filter_means / (np.abs(avg_of_filter_means) + 1e-30)

    if lognormal is True:
        # Transform back for easier interpretation.
        avg_of_filter_means = np.exp(avg_of_filter_means)
    return avg_of_filter_means, coef_of_var, rf_all
