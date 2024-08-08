import argparse

from flyvision import NetworkView
from flyvision.analysis.validation import validate_all_checkpoints
from flyvision.utils.config_utils import parse_kwargs_to_dict


def main():
    parser = argparse.ArgumentParser(description="Validate network.")
    parser.add_argument(
        "keywords", nargs="*", help="keyword arguments in the form key=value"
    )
    args = parser.parse_args()
    kwargs = parse_kwargs_to_dict(args.keywords)

    network_name = f"{kwargs.task_name}/{kwargs.ensemble_and_network_id}"
    network_view = NetworkView(network_name)

    validate_all_checkpoints(network_view)


if __name__ == "__main__":
    main()
