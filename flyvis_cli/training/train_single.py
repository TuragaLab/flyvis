"""
Train the visual system model using the specified configuration.

This script initializes and runs the training process for the model.
It uses the configuration provided through Hydra to set up the solver, manage the
training process, and handle checkpoints.

Key option defaults:

- `train=true`: Whether to run the training process
- `resume=false`: Whether to resume training from the last checkpoint
- `overfit=false`: Whether to use overfitting mode in training
- `checkpoint_only=false`: Whether to create a checkpoint without training
- `save_environment=false`: Whether to save the source code and environment details

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
from importlib import resources
from pathlib import Path

import hydra
from datamate import set_root_context
from omegaconf import OmegaConf

from flyvis import results_dir, source_dir
from flyvis.solver import MultiTaskSolver
from flyvis.utils.logging_utils import save_conda_environment

logging = logger = logging.getLogger(__name__)


def save_env(target_dir: Path = None):
    try:
        shutil.copytree(source_dir, target_dir / "source")
        save_conda_environment(target_dir / "conda_environment.json")
    except FileExistsError:
        pass


CONFIG_PATH = str(resources.files("flyvis") / "config")


@hydra.main(
    config_path="../../flyvis/config",
    config_name="solver.yaml",
    version_base="1.1",
)
def main(args):
    config = prepare_config(args)
    logging.info("Initializing solver with config: %s", config)
    with set_root_context(results_dir):
        solver = MultiTaskSolver(
            config=config,
            delete_if_exists=(
                False if args.resume else args.get("delete_if_exists", False)
            ),
        )
    logging.info("Initialized solver with NetworkDir at %s.", solver.dir.path)

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
