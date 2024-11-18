"""Script to validate a single network across all its checkpoints."""

import argparse

from flyvis import NetworkView
from flyvis.analysis.validation import validate_all_checkpoints
from flyvis.utils.config_utils import HybridArgumentParser


def main():
    parser = HybridArgumentParser(
        hybrid_args={
            'ensemble_and_network_id': {
                'required': True,
                'help': 'ID of the ensemble and network to use, e.g. 0045/000',
            },
            'task_name': {
                'required': True,
                'help': (
                    'Name of the task. Resulting network name will be '
                    'task_name/ensemble_and_network_id.'
                ),
            },
        },
        description=(
            "Validate a single network across all its checkpoints. "
            "Computes and stores validation metrics in the network's validation "
            "directory."
        ),
        usage=(
            "\nflyvis val-single [-h] task_name=TASK ensemble_and_network_id=XXXX/YYY\n"
            "       or\n"
            "%(prog)s [-h] task_name=TASK ensemble_and_network_id=XXXX/YYY\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
--------
1. Validate a specific network:
    flyvis val-single task_name=flow ensemble_and_network_id=0045/000

2. Validate a network from a different task:
    flyvis val-single task_name=depth ensemble_and_network_id=0023/012
""",
    )

    args = parser.parse_with_hybrid_args()

    network_name = f"{args.task_name}/{args.ensemble_and_network_id}"
    network_view = NetworkView(network_name)

    # TODO: add logic to pass different validation methods
    validate_all_checkpoints(network_view)


if __name__ == "__main__":
    main()
