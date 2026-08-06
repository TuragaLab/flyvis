"""Precompute and store the response normalization constants of an ensemble."""

import argparse
import logging

from flyvis import Ensemble
from flyvis.analysis import response_norms
from flyvis.utils.config_utils import HybridArgumentParser

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logging = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args={
            "task_name": {"required": True, "help": "Name of the task, e.g. 'flow'."},
            "ensemble_id": {"required": True, "help": "Id of the ensemble, e.g. '0000'."},
        },
        description=(
            "Compute the normalization constants of an ensemble from its responses to "
            "naturalistic stimuli and store them in the ensemble directory, so that "
            "they never have to be recomputed. Responses that are not cached yet are "
            "simulated, which is the expensive part of this script."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        usage=(
            "\nflyvis responses-norm [-h] task_name=TASK ensemble_id=XXXX [options]\n"
            "       or\n"
            "%(prog)s [-h] task_name=TASK ensemble_id=XXXX [options]\n"
            "\n"
            "Examples:\n"
            "    Store the constants of ensemble flow/0000 next to the ensemble:\n"
            "    flyvis responses-norm task_name=flow ensemble_id=0000\n"
            "\n"
            "    Additionally write them into the constants shipped with the package:\n"
            "    flyvis responses-norm task_name=flow ensemble_id=0000 --export\n"
        ),
    )
    parser.add_argument("--validation_subdir", type=str, default="validation")
    parser.add_argument("--loss_file_name", type=str, default="epe")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if constants are already stored.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help=(
            "Also write the constants into the file shipped with the package "
            f"({response_norms.PRECOMPUTED_FILE}). For maintainers preparing a "
            "release of an ensemble."
        ),
    )
    args = parser.parse_with_hybrid_args()

    ensemble = Ensemble(
        f"{args.task_name}/{args.ensemble_id}",
        best_checkpoint_fn_kwargs={
            "validation_subdir": args.validation_subdir,
            "loss_file_name": args.loss_file_name,
        },
    )

    norms = None if args.force else response_norms.load_response_norms(ensemble)
    if norms is None:
        norms = response_norms.compute_response_norms(ensemble)
    else:
        logging.info("Constants for %s are already stored.", ensemble.name)

    path = response_norms.store_response_norms(ensemble, norms)
    logging.info("Stored %s at %s.", norms, path)

    if args.export:
        path = response_norms.store_response_norms(
            ensemble, norms, path=response_norms.PRECOMPUTED_FILE
        )
        logging.info("Exported %s at %s.", norms, path)
