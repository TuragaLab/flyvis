import argparse

from flyvision import NetworkView
from flyvision.connectome import ReceptiveFields
from flyvision.datasets.moving_bar import MovingEdge
from flyvision.utils.activity_utils import LayerActivity, SourceCurrentView


def run_experiment(network_view, network, dataset, subdir, target_cell_types, dt):
    print("Running experiment...")

    # Remove existing data in subdir if it exists
    del network_view.dir[subdir]

    network_view.dir[subdir].path.mkdir(parents=True, exist_ok=True)
    network_view.dir[subdir].config = dataset.config
    edges = network.connectome.edges.to_df()

    # Initialize data structures for storing responses and currents
    activity_indexer = LayerActivity(None, network_view.connectome, keepref=True)
    source_current_indexer = {
        target_type: SourceCurrentView(ReceptiveFields(target_type, edges), None)
        for target_type in edges.target_type.unique()
    }

    for _, activity, current in network_view.network.current_response(
        dataset,
        dt,
        indices=None,
        t_pre=2,
        t_fade_in=0,
    ):
        # Update activity indexer and store data
        activity_indexer.update(activity)
        for target_type in target_cell_types:
            network_view.dir[subdir][target_type].extend(
                "activity_central", [activity_indexer.central[target_type]]
            )
            for source_type in source_current_indexer[target_type].source_types:
                source_current_indexer[target_type].update(current)
                network_view.dir[subdir][target_type].extend(
                    source_type, [source_current_indexer[target_type][source_type]]
                )


def main():
    parser = argparse.ArgumentParser(description='Run flyvision experiment.')
    parser.add_argument(
        '--ensemble_and_network_id',
        type=str,
        default='0000/000',
        help='Ensemble and network ID',
    )
    parser.add_argument('--task_name', type=str, default='flow', help='Task name')
    parser.add_argument(
        '--validation_subdir',
        type=str,
        default='validation',
        help='Validation subdirectory',
    )
    parser.add_argument(
        '--loss_file_name', type=str, default='epe', help='Loss file name'
    )
    parser.add_argument('--dt', type=float, default=1 / 200, help='Time step')
    parser.add_argument(
        '--target_cell_types',
        nargs='+',
        default=["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "TmY3"],
        help='Target cell types',
    )

    args = parser.parse_args()

    network_view = NetworkView(
        f"{args.task_name}/{args.ensemble_and_network_id}",
        best_checkpoint_fn_kwargs={
            "validation_subdir": args.validation_subdir,
            "loss_file_name": args.loss_file_name,
        },
    )

    subdir = f"movingedge_responses/{network_view.checkpoints.current_chkpt_key}/currents"
    network = network_view.init_network()

    dataset = MovingEdge(
        widths=[80],
        offsets=[-10, 11],
        intensities=[0, 1],
        speeds=[19],
        height=80,
        bar_loc_horizontal=0.0,
        shuffle_offsets=False,
        post_pad_mode="continue",
        t_pre=1.0,
        t_post=1.0,
        dt=args.dt,
        angles=[0, 45, 90, 180, 225, 270],
    )

    run_experiment(
        network_view, network, dataset, subdir, args.target_cell_types, args.dt
    )


if __name__ == "__main__":
    main()
