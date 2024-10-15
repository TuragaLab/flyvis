from flyvision import NetworkView
from flyvision.analysis.validation import validate_all_checkpoints
from flyvision.utils.config_utils import HybridArgumentParser


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
        description="Validate a single network.",
    )
    args = parser.parse_with_hybrid_args()

    network_name = f"{args.task_name}/{args.ensemble_and_network_id}"
    network_view = NetworkView(network_name)

    validate_all_checkpoints(network_view)


if __name__ == "__main__":
    main()
