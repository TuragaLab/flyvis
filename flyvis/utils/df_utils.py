"""Utility functions for operations on pandas DataFrames."""

from typing import Iterable

from pandas import DataFrame


def filter_by_column_values(
    dataframe: DataFrame, column: str, values: Iterable
) -> DataFrame:
    """
    Return subset of dataframe based on list of values to appear in a column.

    Args:
        dataframe: DataFrame with key as column.
        column: Column of the dataframe, e.g. `type`.
        values: Types of neurons e.g. R1, T4a, etc.

    Returns:
        DataFrame: Subset of the input dataframe.

    Example:
        ```python
        filtered_df = filter_by_column_values(df, 'neuron_type', ['R1', 'T4a'])
        ```
    """
    cond = ""
    for t in values:
        cond += f"(dataframe.{column}=='{t}')"
        if t != values[-1]:
            cond += "|"
    return dataframe[eval(cond)]


def where_dataframe(arg_df: DataFrame, **kwargs) -> DataFrame:
    """
    Return indices of rows in a DataFrame where conditions are met.

    Conditions are passed as keyword arguments.

    Args:
        arg_df: Input DataFrame.
        **kwargs: Keyword arguments representing conditions.

    Returns:
        DataFrame: Indices of rows where conditions are met.

    Example:
        ```python
        indices = where_dataframe(df, type='T4a', u=2, v=0)
        ```

    Note:
        The dataframe is expected to have columns matching the keyword arguments.
    """

    def _query_from_kwargs(kwargs):
        _query_start = "{}=={}"
        _query_append = "& {}=={}"

        _query_elements = []
        for i, (key, value) in enumerate(kwargs.items()):
            if isinstance(value, str) and (
                not value.startswith("'") or value.startswith('"')
            ):
                value = f"'{value}'"
            if i == 0:
                _query_elements.append(_query_start.format(key, value))
            else:
                _query_elements.append(_query_append.format(key, value))
        return "".join(_query_elements)

    query = _query_from_kwargs(kwargs)

    return arg_df.query(query).index
