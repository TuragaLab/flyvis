"""Utility functions for operations on pd.DataFrames."""


def filter_df_by_list(alist, adf, column="type"):
    """Get all rows in the df with keys specified in list.

    Args:
        alist (list): e.g. types of neurons e.g. R1, T4a, etc.
        adf (DataFrame): dataframe with key as column.
        column (str): column of the dataframe, e.g. type.
    """
    cond = ""
    for t in alist:
        cond += f"(adf.{column}=='{t}')"
        if t != alist[-1]:
            cond += "|"
    return adf[eval(cond)]