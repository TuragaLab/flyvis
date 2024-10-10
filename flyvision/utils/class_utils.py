"""Utilities for working with classes."""

from copy import deepcopy
from typing import Any, Dict, Optional, Type
from warnings import warn


def find_subclass(cls: Type, target_subclass_name: str) -> Optional[Type]:
    """
    Recursively search for the target subclass.

    Args:
        cls: The base class to start the search from.
        target_subclass_name: The name of the subclass to find.

    Returns:
        The found subclass, or None if not found.
    """
    for subclass in cls.__subclasses__():
        if subclass.__qualname__ == target_subclass_name:
            return subclass
        # Recursively check the subclasses of the current subclass
        found_subclass = find_subclass(subclass, target_subclass_name)
        if found_subclass is not None:
            return found_subclass
    return None


def forward_subclass(
    cls: Type,
    config: Dict[str, Any] = {},
    subclass_key: str = "type",
    unpack_kwargs: bool = True,
) -> Any:
    """
    Forward to a subclass based on the `<subclass_key>` key in `config`.

    Forwards to the parent class if `<subclass_key>` is not in `config`.

    Args:
        cls: The base class to forward from.
        config: Configuration dictionary containing subclass information.
        subclass_key: Key in the config dictionary specifying the subclass.
        unpack_kwargs: Whether to unpack kwargs when initializing the instance.

    Returns:
        An instance of the specified subclass or the base class.

    Note:
        If the specified subclass is not found, a warning is issued and the base
        class is used instead.
    """
    config = deepcopy(config)
    target_subclass = config.pop(subclass_key, None)

    # Prepare kwargs by removing the subclass_key if it exists
    kwargs = {k: v for k, v in config.items() if k != subclass_key}

    def init_with_kwargs(instance: Any) -> None:
        if unpack_kwargs:
            instance.__init__(**kwargs)
        else:
            instance.__init__(kwargs)

    if target_subclass is not None:
        # Find the target subclass recursively
        subclass = find_subclass(cls, target_subclass)
        if subclass is not None:
            instance = object.__new__(subclass)
            init_with_kwargs(instance)
            return instance
        else:
            warn(
                f"Unrecognized {subclass_key} {target_subclass}. "
                f"Using {cls.__qualname__}.",
                stacklevel=2,
            )
    else:
        warn(f"Missing {subclass_key} in config. Using {cls.__qualname__}.", stacklevel=2)

    # Default case: create an instance of the base class
    instance = object.__new__(cls)
    init_with_kwargs(instance)
    return instance
