from typing import Mapping
from numbers import Number
import numpy as np


def byte_to_str(obj):
    """Cast byte elements to string types.

    Note, this function is recursive and will cast all byte elements in a nested
    list or tuple.
    """
    if isinstance(obj, Mapping):
        return type(obj)({k: byte_to_str(v) for k, v in obj.items()})
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.dtype("S")):
            return obj.astype("U")
        return obj
    elif isinstance(obj, list):
        obj = [byte_to_str(item) for item in obj]
        return obj
    elif isinstance(obj, tuple):
        obj = tuple([byte_to_str(item) for item in obj])
        return obj
    elif isinstance(obj, bytes):
        return obj.decode()
    elif isinstance(obj, (str, Number)):
        return obj
    else:
        raise TypeError(f"can't cast {obj} of type {type(obj)} to str")
