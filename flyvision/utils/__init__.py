"""Utility functions."""

from flyvision.utils import (
    activity_utils,
    dataset_utils,
    hex_utils,
    nn_utils,
    nodes_edges_utils,
    tensor_utils,
    xarray_utils,
    xarray_joblib_backend,
)

import xarray as xr

# Register the custom accessors
xr.register_dataarray_accessor("custom")(xarray_utils.CustomAccessor)
xr.register_dataset_accessor("custom")(xarray_utils.CustomAccessor)

del xr


from joblib import register_store_backend

# Register xarray store backend for joblib
register_store_backend(
    'xarray_dataset_zarr', xarray_joblib_backend.XArrayDatasetZarrStoreBackend
)
register_store_backend(
    'xarray_dataarray_zarr', xarray_joblib_backend.XArrayDataArrayZarrStoreBackend
)
register_store_backend(
    'xarray_dataarray_netcdf', xarray_joblib_backend.XArrayDataArrayNetCDFStoreBackend
)
register_store_backend(
    'xarray_dataset_netcdf', xarray_joblib_backend.XArrayDatasetNetCDFStoreBackend
)
register_store_backend(
    'xarray_dataset_h5', xarray_joblib_backend.H5XArrayDatasetStoreBackend
)

del register_store_backend, xarray_joblib_backend
