"""Utility functions for tensors and arrays."""

from typing import Any, Dict, Iterable, List, Literal, Mapping, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


class RefTensor:
    """A tensor with reference indices along the last dimension.

    Attributes:
        values (torch.Tensor): The tensor values.
        indices (torch.Tensor): The reference indices.
    """

    def __init__(self, values: torch.Tensor, indices: torch.Tensor) -> None:
        self.values = values
        self.indices = indices

    def deref(self) -> torch.Tensor:
        """Index the values with the given indices in the last dimension."""
        return self.values.index_select(-1, self.indices)

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"RefTensor(values={self.values.data}, indices={self.indices})"

    def clone(self) -> "RefTensor":
        """Return a copy of the RefTensor cloning values."""
        return RefTensor(self.values.clone(), self.indices)

    def detach(self) -> "RefTensor":
        """Return a copy of the RefTensor detaching values."""
        return RefTensor(self.values.detach(), self.indices)


class AutoDeref(dict):
    """An auto-dereferencing namespace.

    Dereferencing means that if attributes are RefTensors,
    __getitem__ will call RefTensor.deref() to obtain the values at the
    given indices.

    Note:
        Constructed at each forward call in Network. A cache speeds up
        processing, e.g., for when a parameter is referenced multiple times in the
        dynamics.

    Attributes:
        _cache (Dict[str, object]): Cache for dereferenced values.
    """

    _cache: Dict[str, object]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_cache", {})

    def __setitem__(self, key: str, value: object) -> None:
        self._cache.pop(key, None)
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        try:
            val = super().__getitem__(key)
        except AttributeError as e:
            raise e
        if isinstance(val, RefTensor):
            if key not in self._cache:
                self._cache[key] = val.deref()
            val = self._cache[key]
        return val

    def __setattr__(self, key: str, value: object) -> None:
        self.__setitem__(key, value)

    def __getattr__(self, key: str) -> Any:
        return self.__getitem__(key)

    def __repr__(self) -> str:
        def single_line_repr(elem: object) -> str:
            if isinstance(elem, list):
                return "[" + ", ".join(map(single_line_repr, elem)) + "]"
            elif isinstance(elem, AutoDeref):
                return (
                    f"{elem.__class__.__name__}("
                    + ", ".join(f"{k}={single_line_repr(v)}" for k, v in elem.items())
                    + ")"
                )
            else:
                return repr(elem).replace("\n", " ")

        def repr_in_context(elem: object, curr_col: int, indent: int) -> str:
            sl_repr = single_line_repr(elem)
            if len(sl_repr) <= 80 - curr_col:
                return sl_repr
            elif isinstance(elem, list):
                return (
                    "[\n"
                    + " " * (indent + 2)
                    + (",\n" + " " * (indent + 2)).join(
                        repr_in_context(e, indent + 2, indent + 2) for e in elem
                    )
                    + "\n"
                    + " " * indent
                    + "]"
                )
            elif isinstance(elem, AutoDeref):
                return (
                    f"{elem.__class__.__name__}(\n"
                    + " " * (indent + 2)
                    + (",\n" + " " * (indent + 2)).join(
                        f"{k} = " + repr_in_context(v, indent + 5 + len(k), indent + 2)
                        for k, v in elem.items()
                    )
                    + "\n"
                    + " " * indent
                    + ")"
                )
            else:
                return repr(elem)

        return repr_in_context(self, 0, 0)

    def get_as_reftensor(self, key: str) -> RefTensor:
        """Get the original RefTensor without dereferencing."""
        return dict.__getitem__(self, key)

    def clear_cache(self) -> "AutoDeref":
        """Clear the cache and return a cloned instance."""
        object.__setattr__(self, "_cache", {})
        return clone(self)

    def detach(self) -> "AutoDeref":
        """Return a detached copy of the AutoDeref instance."""
        return detach(self)


def detach(obj: AutoDeref) -> AutoDeref:
    """Recursively detach AutoDeref mappings.

    Args:
        obj: The object to detach.

    Returns:
        A detached copy of the input object.

    Raises:
        TypeError: If the object type is not supported.
    """
    if isinstance(obj, (type(None), bool, int, float, str, type)):
        return obj
    elif isinstance(obj, (RefTensor, torch.Tensor)):
        return obj.detach()
    elif isinstance(obj, (list, tuple)):
        return [detach(v) for v in obj]
    elif isinstance(obj, Mapping):
        return AutoDeref({k: detach(dict.__getitem__(obj, k)) for k in obj})
    else:
        try:
            return detach(vars(obj))
        except TypeError as e:
            raise TypeError(f"{obj} of type {type(obj)} as {e}.") from None


def clone(obj: AutoDeref) -> AutoDeref:
    """Recursively clone AutoDeref mappings.

    Args:
        obj: The object to clone.

    Returns:
        A cloned copy of the input object.

    Raises:
        TypeError: If the object type is not supported.
    """
    if isinstance(obj, (type(None), bool, int, float, str, type)):
        return obj
    elif isinstance(obj, (RefTensor, torch.Tensor)):
        return obj.clone()
    elif isinstance(obj, (list, tuple)):
        return [clone(v) for v in obj]
    elif isinstance(obj, Mapping):
        return AutoDeref({k: clone(dict.__getitem__(obj, k)) for k in obj})
    else:
        try:
            print("reached")
            return clone(vars(obj))
        except TypeError as e:
            raise TypeError(f"{obj} of type {type(obj)} as {e}.") from None


def to_numpy(array: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
    """Convert array-like to numpy array.

    Args:
        array: The input array-like object.

    Returns:
        A numpy array.

    Raises:
        ValueError: If the input type cannot be cast to a numpy array.
    """
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    elif isinstance(array, list):
        return np.array(array)
    else:
        raise ValueError(f"type {type(array)} cannot be cast to numpy array")


def atleast_column_vector(array: np.ndarray) -> np.ndarray:
    """Convert 1d-array-like to column vector n x 1 or return the original.

    Args:
        array: The input array.

    Returns:
        A column vector or the original array if it's already 2D or higher.
    """
    array = np.array(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def matrix_mask_by_sub(sub_matrix: np.ndarray, matrix: np.ndarray) -> NDArray[bool]:
    """Create a mask of rows in matrix that are contained in sub_matrix.

    Args:
        sub_matrix: Shape (n_rows1, n_columns)
        matrix: Shape (n_rows2, n_columns)

    Returns:
        1D boolean array of length n_rows2

    Note:
        n_rows1 !<= n_rows2

    Example:
        ```python
        sub_matrix = np.array([[1, 2, 3],
                               [4, 3, 1]])
        matrix = np.array([[3, 4, 1],
                           [4, 3, 1],
                           [1, 2, 3]])
        matrix_mask_by_sub(sub_matrix, matrix)
        # array([False, True, True])
        ```

    Typically, indexing a tensor with indices instead of booleans is
    faster. Therefore, see also where_equal_rows.
    """
    from functools import reduce

    n_rows, n_columns = sub_matrix.shape
    n_rows2 = matrix.shape[0]
    if not n_rows <= n_rows2:
        raise ValueError("sub_matrix must have fewer or equal rows as matrix")
    row_mask = []
    for i in range(n_rows):
        column_mask = []
        for j in range(n_columns):
            column_mask.append(sub_matrix[i, j] == matrix[:, j])
        row_mask.append(reduce(np.logical_and, column_mask))
    return reduce(np.logical_or, row_mask)


def where_equal_rows(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    as_mask: bool = False,
    astype: str = "|S64",
) -> NDArray[int]:
    """Find indices where matrix1 rows are in matrix2.

    Args:
        matrix1: First input matrix.
        matrix2: Second input matrix.
        as_mask: If True, return a boolean mask instead of indices.
        astype: Data type to use for comparison.

    Returns:
        Array of indices or boolean mask.

    Example:
        ```python
        matrix1 = np.array([[1, 2, 3],
                            [4, 3, 1]])
        matrix2 = np.array([[3, 4, 1],
                            [4, 3, 1],
                            [1, 2, 3],
                            [0, 0, 0]])
        where_equal_rows(matrix1, matrix2)
        # array([2, 1])
        matrix2[where_equal_rows(matrix1, matrix2)]
        # array([[1, 2, 3],
        #        [4, 3, 1]])
        ```

    See also:
        matrix_mask_by_sub
    """
    matrix1 = atleast_column_vector(matrix1)
    matrix2 = atleast_column_vector(matrix2)
    matrix1 = matrix1.astype(astype)
    matrix2 = matrix2.astype(astype)

    if as_mask:
        return matrix_mask_by_sub(matrix1, matrix2)

    n_rows1, n_cols1 = matrix1.shape
    n_rows2, n_cols2 = matrix2.shape

    if not n_rows1 <= n_rows2:
        raise ValueError("matrix1 must have less or equal as many rows as matrix2")
    if not n_cols1 == n_cols2:
        raise ValueError("cannot compare matrices with different number of columns")

    where = []
    rows = np.arange(matrix2.shape[0])
    for row in matrix1:
        equal_rows = (row == matrix2).all(axis=1)
        for index in rows[equal_rows]:
            where.append(index)
    return np.array(where)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Broadcast `src` to the shape of `other` along dimension `dim`.

    Args:
        src: Source tensor to broadcast.
        other: Target tensor to broadcast to.
        dim: Dimension along which to broadcast.

    Returns:
        Broadcasted tensor.

    Note:
        From https://github.com/rusty1s/pytorch_scatter/.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_reduce(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    mode: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """Reduce along dimension `dim` using values in the `index` tensor.

    Args:
        src: Source tensor.
        index: Index tensor.
        dim: Dimension along which to reduce.
        mode: Reduction mode, either "mean" or "sum".

    Returns:
        Reduced tensor.

    Note:
        Convenience function for `torch.scatter_reduce` that broadcasts `index` to
        the shape of `src` along dimension `dim` to cohere to pytorch_scatter API.
    """
    index = broadcast(index.long(), src, dim)
    return torch.scatter_reduce(src, dim, index, reduce=mode)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Average along dimension `dim` using values in the `index` tensor.

    Args:
        src: Source tensor.
        index: Index tensor.
        dim: Dimension along which to average.

    Returns:
        Averaged tensor.
    """
    return scatter_reduce(src, index, dim, "mean")


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sum along dimension `dim` using values in the `index` tensor.

    Args:
        src: Source tensor.
        index: Index tensor.
        dim: Dimension along which to sum.

    Returns:
        Summed tensor.
    """
    return scatter_reduce(src, index, dim, "sum")


def select_along_axes(
    array: np.ndarray,
    indices: Union[int, Iterable[int]],
    dims: Union[int, Iterable[int]],
) -> np.ndarray:
    """Select indices from array along dims.

    Args:
        array: Array to take indices from.
        indices: Indices to take.
        dims: Dimensions to take indices from.

    Returns:
        Array with selected indices along specified dimensions.
    """
    if not isinstance(indices, Iterable):
        indices = [indices]
    if not isinstance(dims, Iterable):
        dims = [dims]

    for index, dim in zip(indices, dims):
        if not isinstance(index, Iterable):
            index = [index]
        array = array.take(index, axis=dim)
    return array


def asymmetric_weighting(
    tensor: Union[NDArray, torch.Tensor], gamma: float = 1.0, delta: float = 0.1
) -> Union[NDArray, torch.Tensor]:
    """
    Apply asymmetric weighting to the positive and negative elements of a tensor.

    Args:
        tensor: Input tensor.
        gamma: Positive weighting factor.
        delta: Negative weighting factor.

    Returns:
        Weighted tensor.

    Note:
        The function is defined as:
        f(x) = gamma * x if x > 0 else delta * x
    """
    return gamma * nn.functional.relu(tensor) - delta * nn.functional.relu(-tensor)
