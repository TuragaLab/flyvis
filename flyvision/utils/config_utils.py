from typing import List

import hydra
from datamate import namespacify
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def get_default_config(
    overrides: List[str],
    config_path: str = "../../config",
    config_name: str = "solver",
):
    """
    Expected overridess are:
        - task_name
        - network_id
    """

    GlobalHydra.instance().clear()
    hydra.initialize(config_path=config_path, version_base=None)
    config = hydra.compose(config_name=config_name, overrides=overrides)
    config = namespacify(OmegaConf.to_container(config, resolve=True))
    GlobalHydra.instance().clear()
    return config
