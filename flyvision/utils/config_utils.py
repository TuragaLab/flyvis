import argparse
import sys
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


class HybridArgumentParser(argparse.ArgumentParser):
    """Hybrid argument parser that can parse unknown arguments in basic key=value style.

    Args:
        hybrid_args (list): List of required hybrid arguments passed as key=value.
    """

    def __init__(self, *args, hybrid_args=None, allow_unrecognized=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hybrid_args = hybrid_args
        self.allow_unrecognized = allow_unrecognized

    def parse_with_hybrid_args(self, args=None, namespace=None):
        """Parse arguments and set hybrid arguments as attributes in the namespace."""
        if args is None:
            args = sys.argv[1:]

        args_for_parser = []
        key_value_args = []

        # Separate key=value pairs from other arguments
        for arg in args:
            if '=' in arg and not arg.startswith('-'):
                key_value_args.append(arg)
            else:
                args_for_parser.append(arg)

        # Parse the known arguments
        args, unknown_args = self.parse_known_args(args_for_parser, namespace)

        # Combine key_value_args with unknown_args for processing
        all_unknown_args = key_value_args + unknown_args

        argv = []
        for arg in all_unknown_args:
            if ":" in arg and "=" in arg:
                keytype, value = arg.split("=")
                key, astype = keytype.split(":")
                if value in ["True", "true", "1"] and astype == "bool":
                    setattr(args, key, True)
                elif value in ["False", "false", "0"] and astype == "bool":
                    setattr(args, key, False)
                else:
                    setattr(args, key, eval(astype)(value))
            elif "=" in arg:
                key, value = arg.split("=", 1)
                setattr(args, key, value)
            else:
                argv.append(arg)

        if self.hybrid_args:
            for arg in self.hybrid_args:
                if not hasattr(args, arg):
                    self.error(f"argument {arg} is required as {arg}=value")

        if argv and not self.allow_unrecognized:
            msg = "unrecognized arguments: %s"
            self.error(msg % " ".join(argv))

        return args


def parse_kwargs_to_dict(values):
    """Parse a list of key-value pairs into a dictionary."""
    kwargs = argparse.Namespace()
    for value in values:
        key, value = value.split("=")
        setattr(kwargs, key, value)
    return kwargs
