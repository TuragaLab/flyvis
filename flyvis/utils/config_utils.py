import argparse
import inspect
import os
import sys
import warnings
from importlib import resources
from typing import Any, Dict, List, Optional, Union

import hydra
from colorama import Fore, Style, init
from datamate import Namespace, namespacify
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictKeyType, OmegaConf, errors

CONFIG_PATH = str(resources.files("flyvis") / "config")

init(autoreset=True)  # Initialize colorama


def get_default_config(
    overrides: List[str],
    path: str = "../../config/solver.yaml",
    as_namespace: bool = True,
) -> Union[Dict[DictKeyType, Any], List[Any], None, str, Any, Namespace]:
    """
    Get the default configuration using Hydra.

    Args:
        overrides: List of configuration overrides.
        path: Path to the configuration file.
        as_namespace: Whether to return a namespaced configuration or the
            OmegaConf object.

    Returns:
        The configuration object.

    Note:
        Expected overrides are:
        - task_name
        - network_id
    """

    config = get_config_from_file(path, overrides, resolve=True)
    if as_namespace:
        return namespacify(config)
    return config


def get_config_from_file(
    path: str,
    overrides: List[str] = [],
    resolve: bool = False,
    throw_on_missing: bool = False,
) -> Union[Dict[DictKeyType, Any], List[Any], None, str, Any]:
    """
    Get the configuration from a file.

    Args:
        path: Path to the configuration file (absolute or relative).
        overrides: List of configuration overrides.
        resolve: Whether to resolve the configuration.
        throw_on_missing: Whether to throw an error if a key is missing.

    Returns:
        The configuration object.
    """
    # Get absolute paths
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    caller_dir = os.path.dirname(os.path.abspath(caller_file))

    abs_config_path = os.path.join(caller_dir, path) if not os.path.isabs(path) else path

    config_name = os.path.basename(abs_config_path).replace(".yaml", "")
    config_dir = os.path.dirname(abs_config_path)

    # Calculate relative path from caller to config directory
    rel_config_dir = os.path.relpath(config_dir, caller_dir)
    GlobalHydra.instance().clear()
    hydra.initialize(config_path=rel_config_dir, version_base=None)
    config = hydra.compose(config_name=config_name, overrides=overrides)
    config = OmegaConf.to_container(
        config, resolve=resolve, throw_on_missing=throw_on_missing
    )
    GlobalHydra.instance().clear()
    return config


class HybridArgumentParser(argparse.ArgumentParser):
    """
    Hybrid argument parser that can parse unknown arguments in basic key=value style.

    Attributes:
        hybrid_args: Dictionary of hybrid arguments with their requirements and
            help texts.
        allow_unrecognized: Whether to allow unrecognized arguments.
        drop_disjoint_from: Path to a configuration file that can be used to filter
            out arguments that are present in the command line arguments but not in
            the configuration file. This is to pass through arguments through multiple
            scripts as hydra does not support this.

    Args:
        hybrid_args: Dictionary of hybrid arguments with their requirements and
            help texts.
        allow_unrecognized: Whether to allow unrecognized arguments.
    """

    def __init__(
        self,
        *args: Any,
        hybrid_args: Optional[Dict[str, Dict[str, Any]]] = None,
        allow_unrecognized: bool = True,
        drop_disjoint_from: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hybrid_args = hybrid_args or {}
        self.allow_unrecognized = allow_unrecognized
        self.drop_disjoint_from = drop_disjoint_from
        self._add_hybrid_args_to_help()

    def _add_hybrid_args_to_help(self) -> None:
        """Add hybrid arguments to the help message."""
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

    def parse_with_hybrid_args(
        self,
        args: Optional[List[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> argparse.Namespace:
        """
        Parse arguments and set hybrid arguments as attributes in the namespace.

        Args:
            args: List of arguments to parse.
            namespace: Namespace to populate with parsed arguments.

        Returns:
            Namespace with parsed arguments.

        Raises:
            argparse.ArgumentError: If required arguments are missing or invalid
                values are provided.
        """
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

        if self.drop_disjoint_from:
            args = self._filter_args_based_on_config(args)

        return args

    def hydra_argv(self) -> List[str]:
        hybrid_args = self.parse_with_hybrid_args()
        return [
            f"{key}={value}" for key, value in vars(hybrid_args).items() if ":" not in key
        ]

    def get_registered_args(self) -> List[str]:
        """
        Get a list of all argument names that were registered using add_argument.

        Returns:
            List of argument names (without the -- prefix)
        """
        return [
            action.dest
            for action in self._actions
            if action.dest != "help"  # Exclude the default help action
        ]

    def _filter_args_based_on_config(
        self, args: argparse.Namespace
    ) -> argparse.Namespace:
        """
        Filter arguments based on the Hydra config file specified in drop_disjoint_from.

        Args:
            args: Namespace containing all parsed arguments.

        Returns:
            Filtered Namespace with only arguments present in the config or with
                Hydra syntax.
        """
        if not self.drop_disjoint_from:
            return args

        config = OmegaConf.create(
            get_config_from_file(self.drop_disjoint_from, resolve=False)
        )

        filtered_args = argparse.Namespace()
        registered_args = self.get_registered_args()

        for arg, value in vars(args).items():
            if (
                self._is_in_config(arg, config)
                or arg.startswith('+')
                or arg.startswith('++')
                or arg.startswith('~')
            ):
                setattr(filtered_args, arg, value)
            elif arg not in registered_args:
                warnings.warn(
                    f"{Fore.YELLOW}Argument {Style.BRIGHT}{arg}={value}"
                    f"{Style.RESET_ALL}{Fore.YELLOW} "
                    f"does not affect the hydra config because it is not present in "
                    f"the config file {Style.BRIGHT}{self.drop_disjoint_from}"
                    f"{Style.RESET_ALL}{Fore.YELLOW}. "
                    f"This may be unintended, like a typo, or intended, like a "
                    f"hydra-style argument passed through to another script. "
                    f"Check script docs and config file "
                    f"for clarification.{Style.RESET_ALL}",
                    stacklevel=2,
                )

        return filtered_args

    def _is_in_config(self, arg: str, config: Any) -> bool:
        """
        Check if an argument exists in the config, including nested structures.

        Args:
            arg: The argument to check.
            config: The configuration object or sub-object.

        Returns:
            True if the argument is found in the config, False otherwise.
        """
        try:
            return OmegaConf.select(config, arg, throw_on_missing=True) is not None
        except errors.MissingMandatoryValue:
            return True


def parse_kwargs_to_dict(values: List[str]) -> argparse.Namespace:
    """
    Parse a list of key-value pairs into a dictionary.

    Args:
        values: List of key-value pairs in the format "key=value".

    Returns:
        Namespace object with parsed key-value pairs as attributes.
    """
    kwargs = argparse.Namespace()
    for value in values:
        key, value = value.split("=")
        setattr(kwargs, key, value)
    return kwargs


def safe_cast(value: str, type_name: str) -> Union[int, float, bool, str]:
    """
    Safely cast a string value to a specified type.

    Args:
        value: The string value to cast.
        type_name: The name of the type to cast to.

    Returns:
        The casted value.

    Note:
        Supports casting to int, float, bool, and str.
    """
    if type_name == 'int':
        return int(value)
    elif type_name == 'float':
        return float(value)
    elif type_name == 'bool':
        return value.lower() in ('true', 'yes', '1', 'on')
    else:
        return value  # Default to string


if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args={"abc": {"type": str, "required": True}},
        drop_disjoint_from="../../config/solver.yaml",
        allow_unrecognized=True,
    )
    args, kwargs = parser.parse_known_args()
    print(args)
    print(kwargs)
    args = parser.parse_with_hybrid_args()
    print(args)
    kwargs = ["=".join(arg) for arg in vars(args).items()]
    print(kwargs)
