"""
Package alias for flyvision that maintains identical functionality while allowing
imports through the shorter 'flyvis' name.
"""

import sys
from importlib import import_module
import flyvision


class _PackageAlias:
    def __init__(self):
        # Mirror all regular attributes to maintain identical behavior
        for attr in dir(flyvision):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(flyvision, attr))

        # Required for Python's import machinery to treat us as a proper package
        self.__file__ = getattr(flyvision, '__file__', None)
        self.__path__ = getattr(flyvision, '__path__', None)
        self.__name__ = 'flyvis'
        self.__package__ = 'flyvis'

        # Fall back to public attributes if __all__ isn't defined to support star imports
        self.__all__ = getattr(
            flyvision,
            '__all__',
            [attr for attr in dir(flyvision) if not attr.startswith('_')],
        )

    def __dir__(self):
        # Support IDE autocompletion and dir() calls by exposing all valid attributes
        return sorted(
            set(
                list(self.__dict__.keys())
                + [attr for attr in dir(flyvision) if not attr.startswith('_')]
                + (self.__all__ if hasattr(self, '__all__') else [])
            )
        )

    def __getattr__(self, name):
        try:
            # Enable nested imports by dynamically importing from the original package
            module = import_module(f"flyvision.{name}")
            # Cache to avoid repeated imports of the same module
            setattr(self, name, module)
            return module
        except ImportError as e:
            # Fall back to attribute access for non-module attributes
            try:
                return getattr(flyvision, name)
            except AttributeError:
                raise ImportError(f"Cannot import name '{name}' from 'flyvision'") from e


# Hook into Python's import system
sys.modules[__name__] = _PackageAlias()
