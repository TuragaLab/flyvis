"""Utility functions for operations on pandas DataFrames."""
from typing import Iterable
from pandas import DataFrame


def filter_by_column_values(
    dataframe: DataFrame, column: str, values: Iterable
) -> DataFrame:
    """Return subset of dataframe based on list of values to appear in a column.

    Args:
        dataframe (DataFrame): dataframe with key as column.
        column (str): column of the dataframe, e.g. `type`.
        values (list): e.g. types of neurons e.g. R1, T4a, etc.
    """
    cond = ""
    for t in values:
        cond += f"(dataframe.{column}=='{t}')"
        if t != values[-1]:
            cond += "|"
    return dataframe[eval(cond)]
