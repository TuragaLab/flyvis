"""Script to train the visual system model.


The training is configuration based. The default configuration is stored in
`flyvision/config/solver.yaml`. The configuration can be changed by passing
arguments to the script. The arguments are parsed by Hydra. The configuration
is then used to initialize the solver.

Example:
    $ python train.py \
        id=0 \
        task_name=flow \
        train=true \
        resume=false \
        description='test' \
        task.n_iters=1000
"""

import logging
import shutil
import traceback
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
    logging.info("Initializing solver.")

    # ---- SOLVER INITIALIZATION

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
    delete_if_exists = args.get("delete_if_exists", False)

    with set_root_context(results_dir):
        solver = MultiTaskSolver(
            config=config,
            delete_if_exists=(False if args.resume else delete_if_exists),
        )

    if args.get("save_environment", False):
        save_env(solver.dir.path)

    # ---- NETWORK TRAINING

    def train():
        try:
            solver.train(args.overfit)
        except (OverflowError, MemoryError, RuntimeError) as e:
            if not args.get("failsafe", False):
                raise e
            logging.info(f"{e}")
            traceback.print_exc()
            logging.info("Finished training.")

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
        train()

    # ---- CHECKPOINT

    if args.get("checkpoint_only", False):
        logging.info("Making a checkpoint.")
        # This can be useful to save the initial state without training, e.g.
        # when the model is initialized randomly and not trained but needs a
        # checkpoint for the evaluation pipeline.
        solver.checkpoint()


if __name__ == "__main__":
    main()
