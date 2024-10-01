"""Directory classes.

Note: autoreload in jupyter notebooks can break the reference to the directory classes,
leading to errors after saving of a file that contains a reference to a directory class
that is then reinstantiated in the jupyter notebook.
This is why the classes are defined in a separate file. The error can be fixed by
restarting the kernel.
"""

from datamate import Directory, root

import flyvision

__all__ = ["NetworkDir", "EnsembleDir"]


@root(flyvision.results_dir)
class NetworkDir(Directory):
    """Directory for a network."""


@root(flyvision.results_dir)
class EnsembleDir(Directory):
    """A directory that contains a collection of trained networks."""

    pass
