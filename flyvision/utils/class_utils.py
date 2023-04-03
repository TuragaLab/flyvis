"""Utilities for working with classes."""
from copy import deepcopy
from warnings import warn


def forward_subclass(cls: type, config: object = {}, subclass_key="type") -> object:
    """Forward to a subclass based on the `<subclass_key>` key in `config`.

    Forwards to the parent class if `<subclass_key>` is not in `config`.
    """
    config = deepcopy(config)
    target_subclass = config.pop(subclass_key, None)

    if target_subclass is not None:
        for subclass in cls.__subclasses__():
            if target_subclass == subclass.__qualname__:
                return object.__new__(subclass)
        if target_subclass == cls.__qualname__:
            pass
        else:
            warn(
                f"Unrecognized {subclass_key} {target_subclass}. Using {cls.__qualname__}."
            )
    else:
        warn(f"Missing {subclass_key} in config. Using {cls.__qualname__}.")

    return object.__new__(cls)
