from datamate import namespacify


def get_default_config(overrides=[], config_path="../../config", config_name="config"):
    """
    Expected overridess are:
        - id
        - task
        - train
        - resume
        - solver.comment
    """
    import hydra
    from omegaconf import OmegaConf
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    hydra.initialize(config_path=config_path)
    config = hydra.compose(config_name=config_name, overrides=overrides)
    config = namespacify(OmegaConf.to_container(config, resolve=True))
    GlobalHydra.instance().clear()
    return config
