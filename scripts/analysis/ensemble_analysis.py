"""Store analysis results of an ensemble.

Example:
    Compute UMAP and clustering for the ensemble 0000:
    ```bash
    python ensemble_analysis.py \
        task_name=flow \
        ensemble_and_network_id=0000/000 \
        --functions umap_and_clustering_main
    ```
"""

import logging
import pickle

from flyvis import Ensemble
from flyvis.analysis.clustering import umap_and_clustering_generator
from flyvis.utils.config_utils import HybridArgumentParser

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logging = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args={
            "task_name": {"required": True},
            "ensemble_id": {"required": True},
        },
        description="Analysis for ensemble.",
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
    parser.epilog = __doc__
    args = parser.parse_with_hybrid_args()

    ensemble_name = f"{args.task_name}/{args.ensemble_id}"
    ensemble = Ensemble(
        ensemble_name,
        best_checkpoint_fn_kwargs={
            "validation_subdir": args.validation_subdir,
            "loss_file_name": args.loss_file_name,
        },
    )

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
