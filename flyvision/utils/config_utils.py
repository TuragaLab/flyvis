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
        hybrid_args (dict): Dictionary of hybrid arguments with their requirements
            and help texts.
    """

    def __init__(self, *args, hybrid_args=None, allow_unrecognized=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hybrid_args = hybrid_args or {}
        self.allow_unrecognized = allow_unrecognized
        self._add_hybrid_args_to_help()

    def _add_hybrid_args_to_help(self):
        if self.hybrid_args:
            hybrid_group = self.add_argument_group('Hybrid Arguments')
            for arg, config in self.hybrid_args.items():
                help_text = config.get('help', '')
                required = config.get('required', False)
                arg_type = config.get('type', None)
                arg_help = f"{arg}=value: {help_text}"
                if arg_type:
                    arg_help += f" (type: {arg_type.__name__})"
                if required:
                    arg_help += " (Required)"
                hybrid_group.add_argument(f"--{arg}", help=arg_help, required=False)

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
                try:
                    if value.lower() in ["true", "1", 'yes'] and astype == "bool":
                        setattr(args, key, True)
                    elif value.lower() in ["false", "0", 'no'] and astype == "bool":
                        setattr(args, key, False)
                    else:
                        setattr(args, key, safe_cast(value, astype))
                except (ValueError, TypeError):
                    self.error(
                        f"Invalid type '{astype}' or value '{value}' for argument {key}"
                    )
            elif "=" in arg:
                key, value = arg.split("=", 1)
                if key in self.hybrid_args and 'type' in self.hybrid_args[key]:
                    arg_type = self.hybrid_args[key]['type']
                    try:
                        typed_value = arg_type(value)
                        setattr(args, key, typed_value)
                    except ValueError:
                        self.error(
                            f"Invalid {arg_type.__name__} value '{value}' "
                            f"for argument {key}"
                        )
                else:
                    setattr(args, key, value)
            else:
                argv.append(arg)

        # Apply type conversion for arguments parsed by argparse
        for arg, config in self.hybrid_args.items():
            if (
                hasattr(args, arg)
                and config.get('type')
                and getattr(args, arg) is not None
            ):
                setattr(args, arg, config['type'](getattr(args, arg)))

        # Check for required arguments
        missing_required = []
        for arg, config in self.hybrid_args.items():
            if config.get('required', False) and getattr(args, arg) is None:
                missing_required.append(arg)

        if missing_required:
            self.error(
                f"The following required arguments are missing: "
                f"{', '.join(missing_required)}"
            )

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


def safe_cast(value, type_name):
    if type_name == 'int':
        return int(value)
    elif type_name == 'float':
        return float(value)
    elif type_name == 'bool':
        return value.lower() in ('true', 'yes', '1', 'on')
    else:
        return value  # Default to string
