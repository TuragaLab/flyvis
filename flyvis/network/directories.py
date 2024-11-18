"""Directory classes.

Note:
    Autoreload in jupyter notebooks can break the reference to the directory classes,
    leading to errors after saving of a file that contains a reference to a directory
    class that is then reinstantiated in the jupyter notebook.
    This is why the classes are defined in a separate file. The error can be fixed by
    restarting the kernel.
"""

from datamate import Directory, root

import flyvis

__all__ = ["NetworkDir", "EnsembleDir"]


@root(flyvis.results_dir)
class NetworkDir(Directory):
    """Directory for a network.

    Attributes: Written to by the solver.
        loss (ArrayFile): Loss values over iterations.
        activity (ArrayFile): Mean activity values over iterations.
        activity_min (ArrayFile): Minimum activity values over iterations.
        activity_max (ArrayFile): Maximum activity values over iterations.
        loss_<task> (ArrayFile): Loss values for each specific task over iterations.
        chkpt_index (ArrayFile): Numerical identifiers of checkpoints.
        chkpt_iter (ArrayFile): Iterations at which checkpoints were recorded.
        best_chkpt_index (ArrayFile): Checkpoint index with minimal validation loss.
        dt (ArrayFile): Current time constant of the dataset.
        time_trained (ArrayFile): Total time spent training.

    Attributes: Written by NetworkView.
        __cache__ (Directory): joblib cache.
    """


@root(flyvis.results_dir)
class EnsembleDir(Directory):
    """Contains many NetworkDirs."""
