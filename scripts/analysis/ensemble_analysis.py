"""Script to store recordings of an ensemble.

Example usage:
python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=0000/000
--functions spatial_impulses_responses_main central_impulses_responses_main
"""

from flyvision import Ensemble
from flyvision.analysis.clustering import umap_and_clustering_main
from flyvision.utils.config_utils import HybridArgumentParser

if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args=["task_name", "ensemble_id"],
        description="Recordings for ensemble.",
    )
    parser.add_argument(
        "--chkpt", type=str, default="best", help="checkpoint to evaluate."
    )
    parser.add_argument(
        "--validation_subdir",
        type=str,
        default="validation",
    )
    parser.add_argument(
        "--loss_file_name",
        type=str,
        default="EPE",
    )
    default_functions = [
        "umap_and_clustering_main",
    ]

    parser.add_argument(
        "--functions",
        nargs="+",
        help="List of functions to run.",
        default=default_functions,
        choices=default_functions,
    )
    args = parser.parse_args()

    ensemble_name = f"{args.task_name}/{args.ensemble_id}"
    ensemble = Ensemble(
        ensemble_name,
        checkpoint=args.chkpt,
        validation_subdir=args.validation_subdir,
        loss_file_name=args.loss_file_name,
    )

    if "umap_and_clustering_main" in args.functions:
        umap_and_clustering_main(
            ensemble,
            dt=1 / 200,
            batch_size=4,
            embedding_kwargs={
                "min_dist": 0.105,
                "spread": 9.0,
                "n_neighbors": 5,
                "random_state": 42,
                "n_epochs": 1500,
            },
            gm_kwargs={
                "range_n_clusters": [2, 3, 3, 4, 5],
                "n_init": 100,
                "max_iter": 1000,
                "random_state": 42,
                "tol": 0.001,
            },
            subdir="umap_and_clustering",
        )
