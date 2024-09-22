"""Script to store recordings of an ensemble.

Example usage:
python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=0000/000
--functions spatial_impulses_responses_main central_impulses_responses_main
"""

import logging
import pickle

from flyvision import Ensemble
from flyvision.analysis.clustering import umap_and_clustering_generator
from flyvision.utils.config_utils import HybridArgumentParser

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logging = logging.getLogger(__name__)

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
        default="epe",
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
    parser.add_argument(
        "--delete_umap_and_clustering",
        action="store_true",
    )
    args = parser.parse_with_hybrid_args()

    ensemble_name = f"{args.task_name}/{args.ensemble_id}"
    ensemble = Ensemble(ensemble_name)

    if "umap_and_clustering_main" in args.functions:
        destination = ensemble.path / "umap_and_clustering"
        for cell_type, embedding_and_clustering in umap_and_clustering_generator(
            ensemble
        ):
            # stores if the file does not exist or if the flag is set
            if not (destination / cell_type).with_suffix(".pickle").exists() or (
                (destination / cell_type).with_suffix(".pickle").exists()
                and args.delete_umap_and_clustering
            ):
                destination.mkdir(parents=True, exist_ok=True)
                # Save the renamed pickle
                with open((destination / cell_type).with_suffix(".pickle"), "wb") as f:
                    pickle.dump(embedding_and_clustering, f)
                logging.info(
                    "Saved %s embedding and clustering to %s.", cell_type, destination
                )
            # else nothing to do
