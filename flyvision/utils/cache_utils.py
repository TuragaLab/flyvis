import os
from functools import wraps

import xarray as xr
from joblib import Memory


def context_aware_cache(func=None, context=lambda self: None):
    """Decorator to cache the result of a method based on its arguments and context."""
    if func is None:
        # The decorator is called with argument context
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                context_key = make_hashable(context(self))
                cache_key = hash(
                    make_hashable((func.__name__, args, kwargs, context_key))
                )
                if cache_key in self.cache:
                    return self.cache[cache_key]
                result = func(self, *args, **kwargs)
                self.cache[cache_key] = result
                return result

            return wrapper

        return decorator
    else:
        # The decorator is called without argument context
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            context_key = make_hashable(context(self))
            cache_key = hash(make_hashable((func.__name__, args, kwargs, context_key)))
            if cache_key in self.cache:
                return self.cache[cache_key]
            result = func(self, *args, **kwargs)
            self.cache[cache_key] = result
            return result

        return wrapper


def make_hashable(obj):
    """Recursively converts an object into a hashable type."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        # Immutable and hashable types
        return obj
    elif isinstance(obj, (list, set)):
        # Convert lists and sets to tuples
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        # Convert dictionaries to tuples of sorted key-value pairs
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (tuple, frozenset)):
        # Tuples and frozensets are already immutable
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, slice):
        # Convert slices to tuples
        return (obj.start, obj.stop, obj.step)
    else:
        # Fallback: convert to string (not recommended for complex objects)
        return str(obj)


# Define writers and readers for both Dataset and DataArray with NetCDF and Zarr backends
writers = {
    'dataset': {'netcdf': xr.Dataset.to_netcdf, 'zarr': xr.Dataset.to_zarr},
    'dataarray': {'netcdf': xr.DataArray.to_netcdf, 'zarr': xr.DataArray.to_zarr},
}

readers = {
    'dataset': {'netcdf': xr.open_dataset, 'zarr': xr.open_zarr},
    'dataarray': {'netcdf': xr.open_dataarray, 'zarr': xr.open_zarr},
}


class XarrayMemory(Memory):
    def __init__(self, location, backend="netcdf", **kwargs):
        """
        Initialize the XarrayMemory with a specified backend.

        Parameters:
        - location (str): Directory path for the cache.
        - backend (str): 'netcdf' or 'zarr' to specify the storage format.
        - **kwargs: Additional keyword arguments for joblib.Memory.
        """
        super().__init__(location, **kwargs)
        if backend not in ('netcdf', 'zarr'):
            raise ValueError("backend must be 'netcdf' or 'zarr'")
        self.backend = backend

    def dump(self, value, compress=0):
        """
        Serialize and store the xarray object using the specified backend.

        Parameters:
        - value (xr.Dataset or xr.DataArray): The xarray object to cache.
        - compress (int): Compression level (unused for xarray backends).

        Returns:
        - str: Path to the cached file or directory.
        """
        if isinstance(value, (xr.Dataset, xr.DataArray)):
            # Determine the type key
            type_key = 'dataset' if isinstance(value, xr.Dataset) else 'dataarray'
            writer = writers[type_key][self.backend]

            # Create a unique cache path based on the value's hash
            cache_path = self._cache_path_for_value(value)

            # Prepend type information to the cache path to distinguish between
            # Dataset and DataArray
            cache_path = f"{type_key}_{cache_path}"

            # Set the appropriate file extension based on the backend
            extension = '.nc' if self.backend == 'netcdf' else '.zarr'

            # Full path to store the cached object
            full_cache_path = os.path.join(self.location, cache_path + extension)

            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(full_cache_path), exist_ok=True)

            # Serialize and store the xarray object
            writer(value, full_cache_path)

            return full_cache_path
        else:
            # For non-xarray objects, fallback to joblib's default serialization
            return super().dump(value, compress=compress)

    def load(self, location, mmap_mode=None):
        """
        Deserialize and retrieve the xarray object from the cache.

        Parameters:
        - location (str): Path to the cached file or directory.
        - mmap_mode: Memory mapping mode (unused for xarray backends).

        Returns:
        - xr.Dataset or xr.DataArray or object: The retrieved xarray object or the
            default deserialized object.
        """
        if isinstance(location, str):
            base_name = os.path.basename(location)

            # Determine the type based on the filename prefix
            if base_name.startswith('dataset_'):
                type_key = 'dataset'
            elif base_name.startswith('dataarray_'):
                type_key = 'dataarray'
            else:
                # If type cannot be determined, fallback to joblib's default
                # deserialization
                return super().load(location, mmap_mode=mmap_mode)

            # Determine the backend based on the file extension
            if location.endswith('.zarr') and os.path.isdir(location):
                backend = 'zarr'
            elif location.endswith('.nc') and os.path.isfile(location):
                backend = 'netcdf'
            else:
                # Mismatch between backend and file type, fallback to joblib's default
                return super().load(location, mmap_mode=mmap_mode)

            # Retrieve the appropriate reader function
            reader = readers[type_key][backend]

            # Deserialize and return the xarray object
            return reader(location, decode_times=False)
        else:
            # If location is not a string, fallback to joblib's default deserialization
            return super().load(location, mmap_mode=mmap_mode)
