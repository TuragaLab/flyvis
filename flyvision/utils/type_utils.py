from numbers import Number
from typing import Any, Mapping

import numpy as np


def byte_to_str(obj: Any) -> Any:
    """Cast byte elements to string types recursively.

    This function recursively converts byte elements to string types in nested
    data structures.

    Args:
        obj: The object to be processed. Can be of various types including
            Mapping, numpy.ndarray, list, tuple, bytes, str, or Number.

    Returns:
        The input object with all byte elements converted to strings.

    Raises:
        TypeError: If the input object cannot be cast to a string type.

    Note:
        This function will cast all byte elements in nested lists or tuples.

    Examples:
        ```python
        >>> byte_to_str(b"hello")
        'hello'
        >>> byte_to_str([b"world", 42, {b"key": b"value"}])
        ['world', 42, {'key': 'value'}]
        ```
    """
    if isinstance(obj, Mapping):
        return type(obj)({k: byte_to_str(v) for k, v in obj.items()})
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.dtype("S")):
            return obj.astype("U")
        return obj
    elif isinstance(obj, list):
        return [byte_to_str(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(byte_to_str(item) for item in obj)
    elif isinstance(obj, bytes):
        return obj.decode()
    elif isinstance(obj, (str, Number)):
        return obj
    else:
        raise TypeError(f"can't cast {obj} of type {type(obj)} to str")
