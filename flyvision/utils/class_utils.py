"""Utilities for working with classes."""
from copy import deepcopy
from warnings import warn


# def forward_subclass(cls: type, config: object = {}, subclass_key="type") -> object:
#     """Forward to a subclass based on the `<subclass_key>` key in `config`.

#     Forwards to the parent class if `<subclass_key>` is not in `config`.
#     """
#     config = deepcopy(config)
#     target_subclass = config.pop(subclass_key, None)

#     if target_subclass is not None:
#         for subclass in cls.__subclasses__():
#             if target_subclass == subclass.__qualname__:
#                 return object.__new__(subclass)
#         if target_subclass == cls.__qualname__:
#             pass
#         else:
#             warn(
#                 f"Unrecognized {subclass_key} {target_subclass}. Using {cls.__qualname__}."
#             )
#     else:
#         warn(f"Missing {subclass_key} in config. Using {cls.__qualname__}.")

#     return object.__new__(cls)


# def forward_subclass(
#     cls: type, config: object = {}, subclass_key="type", exclude_keys=[]
# ) -> object:
#     """Forward to a subclass based on the `<subclass_key>` key in `config`.

#     Forwards to the parent class if `<subclass_key>` is not in `config`.
#     """
#     config = deepcopy(config)
#     target_subclass = config.pop(subclass_key, None)

#     # Prepare kwargs by removing the subclass_key if it exists
#     kwargs = {
#         k: v for k, v in config.items() if k != subclass_key and k not in exclude_keys
#     }

#     if target_subclass is not None:
#         for subclass in cls.__subclasses__():
#             if target_subclass == subclass.__qualname__:
#                 instance = object.__new__(subclass)
#                 instance.__init__(**kwargs)  # Initialize with kwargs
#                 return instance
#         if target_subclass == cls.__qualname__:
#             pass
#         else:
#             warn(
#                 f"Unrecognized {subclass_key} {target_subclass}. Using {cls.__qualname__}."
#             )
#     else:
#         warn(f"Missing {subclass_key} in config. Using {cls.__qualname__}.")

#     # Default case: create an instance of the base class
#     instance = object.__new__(cls)
#     instance.__init__(**kwargs)  # Initialize with kwargs
#     return instance


def find_subclass(cls, target_subclass_name):
    """Recursively search for the target subclass."""
    for subclass in cls.__subclasses__():
        if subclass.__qualname__ == target_subclass_name:
            return subclass
        # Recursively check the subclasses of the current subclass
        found_subclass = find_subclass(subclass, target_subclass_name)
        if found_subclass is not None:
            return found_subclass
    return None


def forward_subclass(
    cls: type, config: object = {}, subclass_key="type", unpack_kwargs=True
) -> object:
    """Forward to a subclass based on the `<subclass_key>` key in `config`.

    Forwards to the parent class if `<subclass_key>` is not in `config`.
    """
    config = deepcopy(config)
    target_subclass = config.pop(subclass_key, None)

    # Prepare kwargs by removing the subclass_key if it exists
    kwargs = {k: v for k, v in config.items() if k != subclass_key}

    def init_with_kwargs(instance):
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
                f"Unrecognized {subclass_key} {target_subclass}. Using {cls.__qualname__}."
            )
    else:
        warn(f"Missing {subclass_key} in config. Using {cls.__qualname__}.")

    # Default case: create an instance of the base class
    instance = object.__new__(cls)
    init_with_kwargs(instance)
    return instance
