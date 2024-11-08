"""Utility functions for the flyvis package.

This module organizes various utility submodules and sets up custom
accessors and backends.
"""

from importlib import import_module


def __getattr__(name):
    # This lazy loading mechanism is implemented to improve import performance
    # by deferring the import of submodules until they are actually needed.
    # It reduces the initial import time of the utils package, especially
    # beneficial for large codebases or when only specific utilities are required.
    if name in (
        'activity_utils',
        'cache_utils',
        'chkpt_utils',
        'class_utils',
        'color_utils',
        'compute_cloud_utils',
        'config_utils',
        'dataset_utils',
        'df_utils',
        'groundtruth_utils',
        'hex_utils',
        'logging_utils',
        'log_utils',
        'nn_utils',
        'nodes_edges_utils',
        'tensor_utils',
        'type_utils',
        'xarray_joblib_backend',
        'xarray_utils',
    ):
        return import_module(f'flyvis.utils.{name}')
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def setup_xarray_accessors():
    import xarray as xr
    from . import xarray_utils

    if not hasattr(xr.DataArray, "custom"):
        xr.register_dataarray_accessor("custom")(xarray_utils.CustomAccessor)
    if not hasattr(xr.Dataset, "custom"):
        xr.register_dataset_accessor("custom")(xarray_utils.CustomAccessor)


def setup_joblib_backend():
    from joblib import register_store_backend
    from . import xarray_joblib_backend

    register_store_backend(
        'xarray_dataset_h5', xarray_joblib_backend.H5XArrayDatasetStoreBackend
    )


# Setup functions are called here, but they can be moved to be called only when needed
setup_xarray_accessors()
setup_joblib_backend()
