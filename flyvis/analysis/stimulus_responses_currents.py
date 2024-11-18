"""Stimulus response functions with caching using xarray."""

from __future__ import annotations

import pprint
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

import flyvis
from flyvis.connectome import ReceptiveFields
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.utils.activity_utils import LayerActivity, SourceCurrentView

__all__ = ["compute_currents", "generic_currents", "moving_edge_currents"]


@dataclass
class TargetData:
    activity_central: List[Any] = field(default_factory=list)
    source_data: Dict[str, List[Any]] = field(default_factory=dict)

    def __repr__(self):
        sd = {ct: np.shape(d) for ct, d in self.source_data.items()}
        formatted_sd = pprint.pformat(sd, indent=4)
        return (
            f"TargetData(activity_central={np.shape(self.activity_central)},\n"
            f"            source_data={formatted_sd})"
        )


@dataclass
class ExperimentData:
    config: Any
    checkpoint: str = ""
    target_data: Dict[str, TargetData] = field(default_factory=dict)


def compute_currents(
    network: "flyvis.network.CheckpointedNetwork",
    dataset_class: type,
    dataset_config: Dict,
    target_cell_types: Optional[List[str]] = None,
    t_pre: float = 2.0,
    t_fade_in: float = 0.0,
    dt: float = 1 / 200,
) -> ExperimentData:
    # Reconstruct the network
    network.recover()
    checkpoint = network.checkpoint
    network = network.network  # type: flyvis.network.Network

    # Initialize dataset
    dataset = dataset_class(**dataset_config)

    # Initialize the experiment data
    experiment_data = ExperimentData(config=dataset.config, checkpoint=checkpoint)

    edges = network.connectome.edges.to_df()
    target_types = target_cell_types or edges.target_type.unique()

    # to store the responses and currents in a structured way
    activity_indexer = LayerActivity(None, network.connectome, keepref=True)
    source_current_indexer = {
        target_type: SourceCurrentView(ReceptiveFields(target_type, edges), None)
        for target_type in target_types
    }

    # Initialize target_data in experiment_data
    for target_type in target_types:
        experiment_data.target_data[target_type] = TargetData()

    for _, activity, current in network.current_response(
        dataset,
        dt,
        indices=None,
        t_pre=t_pre,
        t_fade_in=t_fade_in,
    ):
        # implementing computation and writing here together to save some runtime memory

        # Update activity indexer
        activity_indexer.update(activity)

        for target_type in target_types:
            target_data = experiment_data.target_data[target_type]

            # Append central activity data
            target_data.activity_central.append(activity_indexer.central[target_type])

            # Update source current indexer
            source_current_indexer[target_type].update(current)
            for source_type in source_current_indexer[target_type].source_types:
                if source_type not in target_data.source_data:
                    target_data.source_data[source_type] = []
                # Append source current data
                target_data.source_data[source_type].append(
                    source_current_indexer[target_type][source_type]
                )

    return experiment_data


def generic_currents(
    network_view_or_ensemble: Union["flyvis.NetworkView", "flyvis.ensemble.Ensemble"],
    dataset,
    dataset_config: Dict,
    default_dataset_cls: type,
    target_cell_types: Optional[List[str]],
    t_pre: float,
    t_fade_in: float,
    dt: float,
) -> List[ExperimentData]:
    """Return responses for a given dataset as an xarray Dataset."""
    # Handle both single and multiple NetworkViews
    if isinstance(network_view_or_ensemble, flyvis.NetworkView):
        network_views = [network_view_or_ensemble]
    else:
        network_views = list(network_view_or_ensemble.values())

    # Prepare dataset class
    dataset_class = default_dataset_cls if dataset is None else type(dataset)
    dataset_config = dataset_config if dataset is None else dataset.config.to_dict()

    # quick bugfix from datasets that have type in their config but don't expect it
    dataset_config.pop("type", None)

    # Prepare list to collect datasets
    results = []
    # checkpoints = []

    def handle_network(idx, network_view, network=None):
        # Pass initialized network over to next network view to avoid
        # reinitializing the network
        checkpointed_network = network_view.network(
            checkpoint="best", network=network, lazy=True
        )

        # use the cache from this network_view
        cached_compute_responses_fn = network_view.memory.cache(
            compute_currents,
        )
        call_in_cache = cached_compute_responses_fn.check_call_in_cache(
            checkpointed_network,
            dataset_class,
            dataset_config,
            target_cell_types,
            t_pre,
            t_fade_in,
            dt,
        )

        if call_in_cache:
            # don't initialize the network when the call is in cache
            # print('call in cache')
            pass
        elif network is None and checkpointed_network.network is None:
            # initialize the network when the call is not in cache
            # and the network is not passed from the previous network view
            # print('call not in cache, init network')
            checkpointed_network.init()

        # Call the cached compute_responses function
        results.append(
            cached_compute_responses_fn(
                checkpointed_network,
                dataset_class,
                dataset_config,
                target_cell_types,
                t_pre,
                t_fade_in,
            )  # type: ExperimentData
        )
        # checkpoints.append(checkpointed_network.checkpoint)
        return checkpointed_network.network

    network = handle_network(0, network_views[0], None)

    for idx, network_view in enumerate(network_views[1:], 1):
        network = handle_network(idx, network_view, network)

    return results


def moving_edge_currents(
    network_view_or_ensemble: Union["flyvis.NetworkView", "flyvis.ensemble.Ensemble"],
    target_cell_types: Optional[List[str]] = [
        "T4a",
        "T4b",
        "T4c",
        "T4d",
        "T5a",
        "T5b",
        "T5c",
        "T5d",
        "TmY3",
    ],
    dataset: Optional[MovingEdge] = None,
    speeds=(19,),
    offsets=(-10, 11),
    angles=(0, 45, 90, 180, 225, 270),
    dt=1 / 200,
) -> List[ExperimentData]:
    default_dataset_config = dict(
        widths=[80],
        offsets=offsets,
        intensities=[0, 1],
        speeds=speeds,
        height=80,
        bar_loc_horizontal=0.0,
        shuffle_offsets=False,
        post_pad_mode="continue",
        t_pre=1.0,
        t_post=1.0,
        dt=dt,
        angles=angles,
    )
    return generic_currents(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        MovingEdge,
        target_cell_types,
        t_pre=1.0,
        t_fade_in=0.0,
        dt=dt,
    )


if __name__ == "__main__":
    from flyvis import NetworkView

    nv = NetworkView("flow/0000/000")
    moving_edge_currents(nv)
