"""Joblib store backends for xarray objects using Zarr and NetCDF formats.

Rationale: Joblib is a popular library for caching. It stores pickled objects in a
directory. However, pickling xarray objects can be slow and inefficient. This module
provides custom store backends for xarray.Dataset and xarray.DataArray objects that
use Zarr and NetCDF formats, which are more efficient for xarray returned from cached
functions.
"""

import os
import warnings

import xarray as xr
from joblib._store_backends import CacheWarning, FileSystemStoreBackend


class XArrayDatasetZarrStoreBackend(FileSystemStoreBackend):
    """
    A FileSystemStoreBackend subclass that handles xarray.Dataset objects and .zarr files
    using xarray's to_zarr and open_zarr methods. For all other cases, it delegates to the
    superclass's dump_item and load_item methods.
    """

    def dump_item(self, path, item, verbose=1):
        """
        Dump an item to the store. If the item is an xarray.Dataset or the path
        ends with '.zarr', use xarray.Dataset.to_zarr. Otherwise, use the superclass
        method.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.
        item : object
            The item to be stored.
        verbose : int
            Verbosity level.
        """
        # Determine if the item should be handled as a Zarr store
        is_dataset = isinstance(item, xr.Dataset)
        is_zarr_file = path[-1].endswith('.zarr') if path else False

        if is_dataset or is_zarr_file:
            item_path = os.path.join(self.location, *path)
            zarr_path = (
                item_path if is_zarr_file else os.path.join(item_path, 'output.zarr')
            )

            if verbose > 10:
                print(f'Persisting Dataset to Zarr at {zarr_path}')

            try:
                # Ensure the directory exists
                self.create_location(os.path.dirname(zarr_path))
                print("Store item", zarr_path)

                # Save the Dataset to Zarr
                item.to_zarr(zarr_path, mode='w')
            except Exception as e:
                warnings.warn(
                    f"Unable to cache Dataset to Zarr. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
        else:
            # Delegate to the superclass for other item types
            super().dump_item(path, item, verbose)

    def load_item(self, path, verbose=1, msg=None):
        """
        Load an item from the store. If the path ends with '.zarr' or the store contains
        a Zarr directory, use xarray.open_zarr. Otherwise, use the superclass method.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.
        verbose : int
            Verbosity level.
        msg : str, optional
            Additional message for logging (not used here).

        Returns
        -------
        The loaded item, either an xarray.Dataset or the original object.
        """
        item_path = os.path.join(self.location, *path)
        zarr_path = (
            item_path
            if path[-1].endswith('.zarr')
            else os.path.join(item_path, 'output.zarr')
        )

        # Check if the Zarr directory exists
        if self._item_exists(zarr_path):
            if verbose > 1:
                print(f'Loading Dataset from Zarr at {zarr_path}')
            try:
                # Load the Dataset from Zarr
                return xr.open_zarr(zarr_path)
            except Exception as e:
                warnings.warn(
                    f"Unable to load Dataset from Zarr. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
                # Fall back to the superclass method if loading fails
        # Delegate to the superclass for other item types
        return super().load_item(path, verbose, msg)

    def contains_item(self, path):
        """
        Check if there is an item at the path, given as a list of strings.
        It checks for both Zarr and pickle files.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.

        Returns
        -------
        bool
            True if the item exists in either Zarr or pickle format, False otherwise.
        """
        item_path = os.path.join(self.location, *path)
        zarr_filename = os.path.join(item_path, 'output.zarr')
        # delegates to default implementation
        super_filename = os.path.join(item_path, 'output.pkl')

        return self._item_exists(zarr_filename) or super()._item_exists(super_filename)


class XArrayDataArrayZarrStoreBackend(FileSystemStoreBackend):
    """
    A FileSystemStoreBackend subclass that handles xarray.DataArray objects and .zarr
    files
    using xarray's to_zarr and open_zarr methods. For all other cases, it delegates to the
    superclass's dump_item and load_item methods.
    """

    def dump_item(self, path, item, verbose=1):
        """
        Dump an item to the store. If the item is an xarray.DataArray or the path
        ends with '.zarr', use xarray.DataArray.to_zarr. Otherwise, use the superclass
        method.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.
        item : object
            The item to be stored.
        verbose : int
            Verbosity level.
        """
        # Determine if the item should be handled as a Zarr store
        is_dataarray = isinstance(item, xr.DataArray)
        is_zarr_file = path[-1].endswith('.zarr') if path else False

        if is_dataarray or is_zarr_file:
            item_path = os.path.join(self.location, *path)
            zarr_path = (
                item_path if is_zarr_file else os.path.join(item_path, 'output.zarr')
            )

            if verbose > 10:
                print(f'Persisting DataArray to Zarr at {zarr_path}')

            try:
                # Ensure the directory exists
                self.create_location(os.path.dirname(zarr_path))
                print("Store item", zarr_path)

                # Save the DataArray to Zarr
                item.to_zarr(zarr_path, mode='w')
            except Exception as e:
                warnings.warn(
                    f"Unable to cache DataArray to Zarr. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
        else:
            # Delegate to the superclass for other item types
            super().dump_item(path, item, verbose)

    def load_item(self, path, verbose=1, msg=None):
        """
        Load an item from the store. If the path ends with '.zarr' or the store contains
        a Zarr directory, use xarray.open_zarr. Otherwise, use the superclass method.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.
        verbose : int
            Verbosity level.
        msg : str, optional
            Additional message for logging (not used here).

        Returns
        -------
        The loaded item, either an xarray.DataArray or the original object.
        """
        item_path = os.path.join(self.location, *path)
        zarr_path = (
            item_path
            if path[-1].endswith('.zarr')
            else os.path.join(item_path, 'output.zarr')
        )

        # Check if the Zarr directory exists
        if self._item_exists(zarr_path):
            if verbose > 1:
                print(f'Loading DataArray from Zarr at {zarr_path}')
            try:
                # Load the DataArray from Zarr
                return xr.open_zarr(zarr_path)
            except Exception as e:
                warnings.warn(
                    f"Unable to load DataArray from Zarr. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
                # Fall back to the superclass method if loading fails
        # Delegate to the superclass for other item types
        return super().load_item(path, verbose, msg)

    def contains_item(self, path):
        """
        Check if there is an item at the path, given as a list of strings.
        It checks for both Zarr and pickle files.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.

        Returns
        -------
        bool
            True if the item exists in either Zarr or pickle format, False otherwise.
        """
        item_path = os.path.join(self.location, *path)
        zarr_filename = os.path.join(item_path, 'output.zarr')
        super_filename = os.path.join(item_path, 'output.pkl')

        return self._item_exists(zarr_filename) or super()._item_exists(super_filename)


class XArrayDataArrayNetCDFStoreBackend(FileSystemStoreBackend):
    """
    A FileSystemStoreBackend subclass that handles xarray.DataArray objects and .nc files
    using xarray's to_netcdf and open_dataset methods. For all other cases, it delegates
    to the superclass's dump_item and load_item methods.
    """

    def dump_item(self, path, item, verbose=1):
        """
        Dump an item to the store. If the item is an xarray.DataArray or the path
        ends with '.nc', use xarray.DataArray.to_netcdf. Otherwise, use the superclass
        method.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.
        item : object
            The item to be stored.
        verbose : int
            Verbosity level.
        """
        # Determine if the item should be handled as a NetCDF file
        is_dataarray = isinstance(item, xr.DataArray)
        is_netcdf_file = path[-1].endswith('.nc') if path else False

        if is_dataarray or is_netcdf_file:
            item_path = os.path.join(self.location, *path)
            nc_path = (
                item_path if is_netcdf_file else os.path.join(item_path, 'output.nc')
            )

            if verbose > 10:
                print(f'Persisting DataArray to NetCDF at {nc_path}')

            try:
                # Ensure the directory exists
                self.create_location(os.path.dirname(nc_path))
                print("Store item", nc_path)

                # Save the DataArray to NetCDF
                item.to_netcdf(nc_path, mode='w')
            except Exception as e:
                warnings.warn(
                    f"Unable to cache DataArray to NetCDF. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )

        else:
            # Delegate to the superclass for other item types
            super().dump_item(path, item, verbose)

    def load_item(self, path, verbose=1, msg=None):
        """
        Load an item from the store. If the path ends with '.nc' or the store contains
        a NetCDF file, use xarray.open_dataset. Otherwise, use the superclass method.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.
        verbose : int
            Verbosity level.
        msg : str, optional
            Additional message for logging (not used here).

        Returns
        -------
        The loaded item, either an xarray.DataArray or the original object.
        """
        item_path = os.path.join(self.location, *path)
        nc_path = (
            item_path
            if path[-1].endswith('.nc')
            else os.path.join(item_path, 'output.nc')
        )

        # Check if the NetCDF file exists
        if self._item_exists(nc_path):
            if verbose > 1:
                print(f'Loading DataArray from NetCDF at {nc_path}')
            try:
                # Load the Dataset from NetCDF
                dataset = xr.open_dataset(nc_path)
                # Convert Dataset to DataArray if possible
                if 'variable_name' in dataset:
                    return dataset['variable_name']
                else:
                    # If the variable name is unknown, return the first variable
                    first_var = list(dataset.data_vars)[0]
                    return dataset[first_var]
            except Exception as e:
                warnings.warn(
                    f"Unable to load DataArray from NetCDF. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
                # Fall back to the superclass method if loading fails
        # Delegate to the superclass for other item types
        return super().load_item(path, verbose, msg)

    def contains_item(self, path):
        """
        Check if there is an item at the path, given as a list of strings.
        It checks for both NetCDF and pickle files.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.

        Returns
        -------
        bool
            True if the item exists in either NetCDF or pickle format, False otherwise.
        """
        item_path = os.path.join(self.location, *path)
        nc_filename = (
            item_path
            if path[-1].endswith('.nc')
            else os.path.join(item_path, 'output.nc')
        )
        super_filename = os.path.join(item_path, 'output.pkl')

        return self._item_exists(nc_filename) or super()._item_exists(super_filename)


class XArrayDatasetNetCDFStoreBackend(FileSystemStoreBackend):
    """
    A FileSystemStoreBackend subclass that handles xarray.Dataset objects and .nc files
    using xarray's to_netcdf and open_dataset methods. For all other cases, it delegates
    to the superclass's dump_item and load_item methods.
    """

    def dump_item(self, path, item, verbose=1):
        """
        Dump an item to the store. If the item is an xarray.Dataset or the path
        ends with '.nc', use xarray.Dataset.to_netcdf. Otherwise, use the superclass
        method.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.
        item : object
            The item to be stored.
        verbose : int
            Verbosity level.
        """
        # Determine if the item should be handled as a NetCDF file
        is_dataset = isinstance(item, xr.Dataset)
        is_netcdf_file = path[-1].endswith('.nc') if path else False

        if is_dataset or is_netcdf_file:
            item_path = os.path.join(self.location, *path)
            nc_path = (
                item_path if is_netcdf_file else os.path.join(item_path, 'output.nc')
            )

            if verbose > 10:
                print(f'Persisting Dataset to NetCDF at {nc_path}')

            try:
                # Ensure the directory exists
                self.create_location(os.path.dirname(nc_path))
                print("Store item", nc_path)

                # Save the Dataset to NetCDF
                item.to_netcdf(nc_path, mode='w')
            except Exception as e:
                warnings.warn(
                    f"Unable to cache Dataset to NetCDF. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
        else:
            # Delegate to the superclass for other item types
            super().dump_item(path, item, verbose)

    def load_item(self, path, verbose=1, msg=None):
        """
        Load an item from the store. If the path ends with '.nc' or the store contains
        a NetCDF file, use xarray.open_dataset. Otherwise, use the superclass method.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.
        verbose : int
            Verbosity level.
        msg : str, optional
            Additional message for logging (not used here).

        Returns
        -------
        The loaded item, either an xarray.Dataset or the original object.
        """
        item_path = os.path.join(self.location, *path)
        nc_path = (
            item_path
            if path[-1].endswith('.nc')
            else os.path.join(item_path, 'output.nc')
        )

        # Check if the NetCDF file exists
        if self._item_exists(nc_path):
            if verbose > 1:
                print(f'Loading Dataset from NetCDF at {nc_path}')
            try:
                # Load the Dataset from NetCDF
                return xr.open_dataset(nc_path)
            except Exception as e:
                warnings.warn(
                    f"Unable to load Dataset from NetCDF. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
                # Fall back to the superclass method if loading fails
        # Delegate to the superclass for other item types
        return super().load_item(path, verbose, msg)

    def contains_item(self, path):
        """
        Check if there is an item at the path, given as a list of strings.
        It checks for both NetCDF and pickle files.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.

        Returns
        -------
        bool
            True if the item exists in either NetCDF or pickle format, False otherwise.
        """
        item_path = os.path.join(self.location, *path)
        nc_filename = (
            item_path
            if path[-1].endswith('.nc')
            else os.path.join(item_path, 'output.nc')
        )
        super_filename = os.path.join(item_path, 'output.pkl')

        return self._item_exists(nc_filename) or super()._item_exists(super_filename)


class H5XArrayDatasetStoreBackend(FileSystemStoreBackend):
    """
    A FileSystemStoreBackend subclass that handles xarray.Dataset objects and .nc files
    using xarray's to_netcdf and open_dataset methods. For all other cases, it delegates
    to the superclass's dump_item and load_item methods.
    """

    def dump_item(self, path, item, verbose=1):
        """
        Dump an item to the store. If the item is an xarray.Dataset or the path
        ends with '.h5', use xarray.Dataset.to_netcdf. Otherwise, use the superclass
        method.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.
        item : object
            The item to be stored.
        verbose : int
            Verbosity level.
        """
        # Determine if the item should be handled as a NetCDF file
        is_dataset = isinstance(item, xr.Dataset)
        is_netcdf_file = path[-1].endswith('.h5') if path else False

        if is_dataset or is_netcdf_file:
            item_path = os.path.join(self.location, *path)
            nc_path = (
                item_path if is_netcdf_file else os.path.join(item_path, 'output.h5')
            )

            if verbose > 10:
                print(f'Persisting Dataset to NetCDF at {nc_path}')

            try:
                # Ensure the directory exists
                self.create_location(os.path.dirname(nc_path))
                print("Store item", nc_path)

                # Save the Dataset to NetCDF
                item.to_netcdf(nc_path, mode='w')
            except Exception as e:
                warnings.warn(
                    f"Unable to cache Dataset to NetCDF. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
        else:
            # Delegate to the superclass for other item types
            super().dump_item(path, item, verbose)

    def load_item(self, path, verbose=1, msg=None):
        """
        Load an item from the store. If the path ends with '.h5' or the store contains
        a NetCDF file, use xarray.open_dataset. Otherwise, use the superclass method.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.
        verbose : int
            Verbosity level.
        msg : str, optional
            Additional message for logging (not used here).

        Returns
        -------
        The loaded item, either an xarray.Dataset or the original object.
        """
        item_path = os.path.join(self.location, *path)
        nc_path = (
            item_path
            if path[-1].endswith('.h5')
            else os.path.join(item_path, 'output.h5')
        )

        # Check if the NetCDF file exists
        if self._item_exists(nc_path):
            if verbose > 1:
                print(f'Loading Dataset from NetCDF at {nc_path}')
            try:
                # Load the Dataset from NetCDF
                return xr.open_dataset(nc_path)
            except Exception as e:
                warnings.warn(
                    f"Unable to load Dataset from NetCDF. Exception: {e}.",
                    CacheWarning,
                    stacklevel=2,
                )
                # Fall back to the superclass method if loading fails
        # Delegate to the superclass for other item types
        return super().load_item(path, verbose, msg)

    def contains_item(self, path):
        """
        Check if there is an item at the path, given as a list of strings.
        It checks for both NetCDF and pickle files.

        Parameters
        ----------
        path : list of str
            The identifier for the item in the store.

        Returns
        -------
        bool
            True if the item exists in either NetCDF or pickle format, False otherwise.
        """
        item_path = os.path.join(self.location, *path)
        nc_filename = (
            item_path
            if path[-1].endswith('.h5')
            else os.path.join(item_path, 'output.h5')
        )
        super_filename = os.path.join(item_path, 'output.pkl')

        return self._item_exists(nc_filename) or super()._item_exists(super_filename)
