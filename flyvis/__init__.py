"""
Alias package for flyvision
"""

from flyvision import *

from contextlib import suppress

with suppress(ImportError):
    from flyvision import __all__

with suppress(ImportError):
    from flyvision import __version__
