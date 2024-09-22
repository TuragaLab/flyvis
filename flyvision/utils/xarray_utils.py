from __future__ import annotations

import contextlib
import operator
import warnings
from functools import wraps
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def where_xarray(dataset, rtol=1.0e-5, atol=1.0e-8, **kwargs):
    """Return a subset of the xarray Dataset or DataArray where coordinates meet
    specified query-like conditions.

    Parameters:
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        The dataset or data array to filter.
    atol : float
        Tolerance for floating point comparisons (used for continuous coordinates).
    rtol : float
        Relative tolerance for floating point comparisons (used for continuous
        coordinates).
    **kwargs : dict
        Query-like conditions on coordinates. Conditions can be specified as:
            - Strings with comma-separated conditions (interpreted as AND).
              Example: time='<1.0,>0'
            - Iterables (lists, tuples) representing multiple conditions (interpreted as
                OR).
              Example: cell_type=["T4a", "T4b"]
            - Single values for equality conditions.
              Example: intensity=1.0

    Returns:
    -------
    filtered_dataset : xarray.Dataset or xarray.DataArray
        The filtered dataset or data array.

    Example:
    -------
    filtered_ds = where_xarray(
        ds,
        cell_type=["T4a", "T4b"],
        time="<1.0,>0",
        intensity=1.0,
        radius=6,
        width=2.4
    )
    """

    # Define a mapping of operators from string to functions
    operators = {
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '<': operator.lt,
    }

    # Sort operators by length in descending order to match multi-character operators
    # first
    sorted_operators = sorted(operators.keys(), key=len, reverse=True)

    def parse_condition(cond_str):
        """Parse a single condition string into (operator_function, target_value)."""
        cond_str = cond_str.strip()
        for op_str in sorted_operators:
            if cond_str.startswith(op_str):
                target = cond_str[len(op_str) :].strip()
                with contextlib.suppress(ValueError):
                    target = float(target)
                return (operators[op_str], target)
        # If no operator is found, assume equality
        try:
            target = float(cond_str)
        except ValueError:
            target = cond_str
        return (operator.eq, target)

    filtered_dataset = dataset

    for coord_name, condition in kwargs.items():
        # Check if coord_name is a coordinate in the dataset
        if coord_name not in dataset.coords:
            raise ValueError(f"Coordinate '{coord_name}' not found in the dataset.")

        coord_values = dataset.coords[coord_name]
        coord_mask = xr.ones_like(coord_values, dtype=bool)  # Initialize mask as all True

        if isinstance(condition, str):
            # String conditions: multiple conditions separated by commas (AND logic)
            condition_strings = [c.strip() for c in condition.split(',') if c.strip()]
            for cond_str in condition_strings:
                op_func, target_value = parse_condition(cond_str)

                if np.issubdtype(coord_values.dtype, np.floating):
                    if op_func == operator.eq:
                        mask = np.isclose(
                            coord_values, target_value, atol=atol, rtol=rtol
                        )
                    else:
                        mask = op_func(coord_values, target_value)
                else:
                    mask = op_func(coord_values, target_value)

                # Combine masks using logical AND
                coord_mask &= xr.DataArray(
                    mask, dims=coord_values.dims, coords=coord_values.coords
                )

        elif isinstance(condition, Iterable) and not isinstance(condition, (str, bytes)):
            # Iterable conditions: each element is a separate condition (OR logic)
            temp_mask = xr.zeros_like(
                coord_values, dtype=bool
            )  # Initialize mask as all False
            for item in condition:
                if isinstance(item, str):
                    # Parse condition string
                    op_func, target_value = parse_condition(item)
                else:
                    # Assume equality if not a string condition
                    op_func, target_value = operator.eq, item

                if np.issubdtype(coord_values.dtype, np.floating):
                    if op_func == operator.eq:
                        mask = np.isclose(
                            coord_values, target_value, atol=atol, rtol=rtol
                        )
                    else:
                        mask = op_func(coord_values, target_value)
                else:
                    mask = op_func(coord_values, target_value)

                # Combine masks using logical OR
                temp_mask |= xr.DataArray(
                    mask, dims=coord_values.dims, coords=coord_values.coords
                )
            coord_mask &= temp_mask  # Apply OR mask with existing mask
        else:
            # Single non-string, non-iterable value: assume equality
            op_func, target_value = operator.eq, condition
            if np.issubdtype(coord_values.dtype, np.floating):
                if op_func == operator.eq:
                    mask = np.isclose(coord_values, target_value, atol=atol, rtol=rtol)
                else:
                    mask = op_func(coord_values, target_value)
            else:
                mask = op_func(coord_values, target_value)
            coord_mask &= xr.DataArray(
                mask, dims=coord_values.dims, coords=coord_values.coords
            )

        # Apply the combined mask
        filtered_dataset = filtered_dataset.where(coord_mask, drop=True)

    return filtered_dataset


def plot_traces(
    dataset: xr.DataArray | xr.Dataset,
    key: str,
    x: str,  # x-axis dimension
    extra_legend_coords: List[str] = [],
    plot_kwargs={},
    **kwargs,
):
    """
    Plot the flash response traces from the dataset, optionally filtered by various
    parameters.

    Parameters
    ----------
    dataset : xr.DataArray
        The dataset containing the responses to plot.
    key : str
        The key of the data to plot if the dataset is a Dataset.
    x : str
        The dimension to use as the x-axis.
    extra_legend_coords : List[str]
        Additional coordinates to include in the legend.
    plot_kwargs : dict
        Additional keyword arguments to pass to the plot function.
    **kwargs : dict
        Query-like conditions on coordinates, e.g., time='<0.5,>0.1', lat='>=45.0' or
        radius=6. Ranges and multiple conditions can be given as string and
        separated by commas. Lists can be provided for equality conditions.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes object containing the plot.
    """
    traces = dataset.custom.where(**kwargs)

    if key:
        traces = traces[key]

    arg_df = traces.sample.to_dataframe()

    # Stack all dims besides x
    stack_dims = [dim for dim in traces.dims if dim not in list(traces[x].coords.keys())]
    # logging.info("Stacking dimensions: %s", stack_dims)
    traces = traces.stack(traces=stack_dims)

    num_stacks = traces.sizes.get('traces', 0)
    if num_stacks > 250:
        warnings.warn(
            f"The traces stack has {num_stacks} elements.",
            UserWarning,
            stacklevel=2,
        )

    original_legend_labels = [col for col in arg_df.columns if col != 'sample']

    stacked_legend_labels = list(stack_dims)

    legend_labels = stacked_legend_labels + extra_legend_coords + original_legend_labels

    legend_coords = np.array([np.atleast_1d(traces[col].data) for col in legend_labels]).T

    # return legend_coords, traces, legend_labels

    legend_info = np.array([
        ", ".join([f"{col}: {value}" for col, value in zip(legend_labels, row)])
        for row in legend_coords
    ])

    traces = traces.assign_coords(legend_info=("traces", legend_info))

    traces.plot.line(x=x, hue="legend_info", **plot_kwargs)

    ax = plt.gca()

    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(None)

    return ax


class CustomAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj  # Store the xarray object

    @wraps(where_xarray)
    def where(self, **kwargs):
        __doc__ = where_xarray.__doc__
        return where_xarray(self._obj, **kwargs)

    @wraps(plot_traces)
    def plot_traces(self, x, key="", extra_legend_coords=[], plot_kwargs={}, **kwargs):
        __doc__ = (
            plot_traces.__doc__
            + """
        Example:
        -------
        fig, ax = plt.subplots()
        r.custom.plot_traces(key='stimulus',
                            x='time',
                            speed=[19, 25],
                            intensity=1,
                            angle=90,
                            u_in=0,
                            v_in=0,
                            plot_kwargs=dict(ax=ax),
                            time='>0,<1.0'
                            )
        r.custom.plot_traces(key='responses',
                            x='time',
                            speed=[19, 25],
                            cell_type='T4c',
                            intensity=1,
                            angle=90,
                            network_id=0,
                            plot_kwargs=dict(ax=ax),
                            time='>0,<1.0'
        )
        """
        )
        return plot_traces(
            self._obj,
            key,
            x,
            extra_legend_coords,
            plot_kwargs,
            **kwargs,
        )
