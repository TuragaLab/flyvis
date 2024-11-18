"""Joblib store backends for xarray objects using h5 via netcdf.

Info: Rationale
    Joblib is powerful for caching. It pickles objects into a directory. Pickling
    xarray objects however, which have their own storage solutions, seems inefficient.
    This module provides custom store backends for joblib that use h5 via netcdf for
    storage of xarray objects.
"""

import logging
import os
import warnings
from typing import Any, List

import xarray as xr
from joblib._store_backends import CacheWarning, FileSystemStoreBackend

logger = logging.getLogger(__name__)


class H5XArrayDatasetStoreBackend(FileSystemStoreBackend):
    """FileSystemStoreBackend subclass for handling xarray.Dataset objects.

    This class uses xarray's to_netcdf and open_dataset methods for Dataset objects and
    .h5 files.

    Attributes:
        location (str): The base directory for storing items.
    """

    def dump_item(self, path: List[str], item: Any, *args, **kwargs) -> None:
        """Dump an item to the store.

        If the item is an xarray.Dataset or the path ends with '.h5', use
        xarray.Dataset.to_netcdf. Otherwise, use the superclass method.

        Args:
            path: The identifier for the item in the store.
            item: The item to be stored.
            *args: Variable positional arguments passed to parent class or to_netcdf
            **kwargs: Variable keyword arguments passed to parent class or to_netcdf
        """
        is_dataset = isinstance(item, xr.Dataset)
        is_h5_file = path[-1].endswith('.h5') if path else False

        if is_dataset or is_h5_file:
            item_path = os.path.join(self.location, *path)
            nc_path = item_path if is_h5_file else os.path.join(item_path, 'output.h5')

            verbose = kwargs.get('verbose', 1)
            if verbose > 10:
                logger.info('Persisting Dataset to h5 at %s', nc_path)

            try:
                self.create_location(os.path.dirname(nc_path))
                logger.info("Store item %s", nc_path)
                # Ensure mode='w' by default but allow override through kwargs
                kwargs.setdefault('mode', 'w')
                item.to_netcdf(nc_path)
            except Exception as e:
                warnings.warn(
                    f"Unable to cache Dataset to h5. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
        else:
            super().dump_item(path, item, *args, **kwargs)

    def load_item(self, path: List[str], *args, **kwargs) -> Any:
        """Load an item from the store.

        If the path ends with '.h5' or the store contains a h5 file, use
        xarray.open_dataset. Otherwise, use the superclass method.

        Args:
            path: The identifier for the item in the store.
            *args: Variable positional arguments passed to parent class or xr.open_dataset
            **kwargs: Variable keyword arguments passed to parent class or xr.open_dataset

        Returns:
            The loaded item, either an xarray.Dataset or the original object.
        """
        item_path = os.path.join(self.location, *path)
        nc_path = (
            item_path
            if path[-1].endswith('.h5')
            else os.path.join(item_path, 'output.h5')
        )
        print(nc_path)
        if self._item_exists(nc_path):
            verbose = kwargs.get('verbose', 1)
            if verbose > 1:
                logger.info('Loading Dataset from h5 at %s', nc_path)
            try:
                return xr.open_dataset(nc_path)
            except Exception as e:
                warnings.warn(
                    f"Unable to load Dataset from h5. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
        return super().load_item(path, *args, **kwargs)

    def contains_item(self, path: List[str]) -> bool:
        """Check if there is an item at the given path.

        This method checks for both h5 and pickle files.

        Args:
            path: The identifier for the item in the store.

        Returns:
            True if the item exists in either h5 or pickle format, False otherwise.
        """
        item_path = os.path.join(self.location, *path)
        nc_filename = (
            item_path
            if path[-1].endswith('.h5')
            else os.path.join(item_path, 'output.h5')
        )
        super_filename = os.path.join(item_path, 'output.pkl')

        return self._item_exists(nc_filename) or super()._item_exists(super_filename)
