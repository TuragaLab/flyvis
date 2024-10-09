"""
Train the visual system model using the specified configuration.

This script initializes and runs the training process for the model.
It uses the configuration provided through Hydra to set up the solver, manage the
training process, and handle checkpoints.

The function supports the following key operations:

1. Initializing the solver with the provided configuration
2. Saving the environment details if requested for reproducibility
3. Resuming training from a checkpoint if specified
4. Running the training process
5. Creating a checkpoint without training if requested

Key configuration options:

- train (bool): Whether to run the training process
- resume (bool): Whether to resume training from the last checkpoint
- overfit (bool): Whether to use overfitting mode in training
- checkpoint_only (bool): Whether to create a checkpoint without training
- save_environment (bool): Whether to save the source code and environment details

Example:
    Train a network for 1000 iterations and describe it as 'test':
    ```bash
    $ python train.py \
        ensemble_and_network_id=0045/000 \
        task_name=flow \
        train=true \
        resume=false \
        task.n_iters=1000
        description='test'
    ```
"""

import logging
import shutil
from pathlib import Path

import hydra
from datamate import set_root_context
from omegaconf import OmegaConf

from flyvision import results_dir, source_dir
from flyvision.solver import MultiTaskSolver
from flyvision.utils.logging_utils import save_conda_environment

logging = logger = logging.getLogger(__name__)


def save_env(target_dir: Path = None):
    """
    Save source code and the environment details to a file in the target directory.
    """
    try:
        shutil.copytree(source_dir, target_dir / "source")
        save_conda_environment(target_dir / "conda_environment.json")
    except FileExistsError:
        pass


@hydra.main(
    config_path=str(Path(__file__).parent.parent.parent / "config"),
    config_name="solver.yaml",
    version_base="1.1",
)
def main(args):
    """
    Train the visual system model using the specified configuration.

    This script initializes and runs the training process for the model.
    It uses the configuration provided through Hydra to set up the solver, manage the
    training process, and handle checkpoints.

    The function supports the following key operations:

    1. Initializing the solver with the provided configuration
    2. Saving the environment details if requested for reproducibility
    3. Resuming training from a checkpoint if specified
    4. Running the training process
    5. Creating a checkpoint without training if requested

    Key configuration options:

    - train (bool): Whether to run the training process
    - resume (bool): Whether to resume training from the last checkpoint
    - overfit (bool): Whether to use overfitting mode in training
    - checkpoint_only (bool): Whether to create a checkpoint without training
    - save_environment (bool): Whether to save the source code and environment details
    """

    logging.info("Initializing solver.")
    with set_root_context(results_dir):
        solver = MultiTaskSolver(
            config=prepare_config(args),
            delete_if_exists=(
                False if args.resume else args.get("delete_if_exists", False)
            ),
        )

    if args.get("save_environment", False):
        save_env(solver.dir.path)

    if args.resume:
        solver.recover(
            network=True,
            decoder=True,
            optimizer=True,
            penalty=True,
            checkpoint=-1,
            strict=True,
            force=False,
        )
        logging.info("Resuming from last checkpoint.")

    if args.train:
        logging.info("Starting training.")
        solver.train(args.overfit)
        logging.info("Finished training.")

    if args.get("checkpoint_only", False):
        logging.info("Making a checkpoint.")
        # to save the initial state without training
        # when the model is initialized randomly and not trained but needs a
        # checkpoint for the evaluation pipeline
        solver.checkpoint()


def prepare_config(args):
    config = OmegaConf.to_container(args, resolve=True)
    config = {
        key: value
        for key, value in config.items()
        if key
        in [
            "network_name",
            "network",
            "task",
            "optim",
            "penalizer",
            "scheduler",
            "description",
        ]
    }
    return config


if __name__ == "__main__":
    main()
