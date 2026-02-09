"""Train VanillaHexCNN baselines in a flyvis-compatible results layout."""

import logging
from importlib import resources

import hydra
from omegaconf import OmegaConf

from flyvis.baselines.vanilla_hex_cnn.trainer import train_baseline

logger = logging.getLogger(__name__)

CONFIG_PATH = str(resources.files("flyvis.baselines.vanilla_hex_cnn").joinpath("config"))


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="config.yaml",
    version_base="1.1",
)
def main(args):
    config = OmegaConf.to_container(args, resolve=True)
    result = train_baseline(config)
    logger.info(
        "Finished baseline run %s (steps=%d, checkpoints=%d) at %s",
        result.network_name,
        result.n_train_steps,
        result.n_checkpoints,
        result.model_path,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

