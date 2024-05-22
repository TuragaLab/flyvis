#!/groups/turaga/home/lappalainenj/miniconda3/envs/flyvision/bin/python
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

from pathlib import Path
import shutil
import traceback
import logging
import hydra
from omegaconf import OmegaConf

from datamate import set_root_context, namespacify

from flyvision import results_dir, source_dir
from flyvision.solver import MultiTaskSolver

logging = logger = logging.getLogger(__name__)


def save_conda_environment(path):
    import subprocess
    import json

    # Use the conda list command to get a list of installed packages and their versions
    result = subprocess.run(
        ["conda", "list", "--json"], stdout=subprocess.PIPE, text=True
    )

    # Parse the JSON output
    installed_packages = json.loads(result.stdout)

    # Write the parsed JSON data to the specified file path
    with open(path.with_suffix(".json"), "w") as json_file:
        json.dump(installed_packages, json_file, indent=4)


@hydra.main(
    config_path=str(Path(__file__).parent.parent.parent / "config"),
    config_name="solver.yaml",
)
def main(args):

    logging.info("Initializing solver.")

    # ---- SOLVER INITIALIZATION

    config = OmegaConf.to_container(args, resolve=True)
    config = {key: value for key, value in config.items() if key in ["network_name", "network", "task", "optim", "penalizer", "scheduler", "description"]}

    delete_if_exists = args.get("delete_if_exists", False)

    with set_root_context(results_dir):
        solver = MultiTaskSolver(
            config=config,
            delete_if_exists=(False if args.resume else delete_if_exists),
        )

    if args.get("save_environment", False):
        try:
            # save environment details to model directory
            save_conda_environment(solver.dir.path / "conda_environment.json")
            shutil.copytree(source_dir, solver.dir.path / "code/dvs")
        except FileExistsError:
            pass

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
            recover_network=True,
            recover_decoder=True,
            recover_optimizer=True,
            recover_penalty=True,
            strict_recover=True,
            checkpoint=-1,
            other=None,
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
