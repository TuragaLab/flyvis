from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def context_aware_cache(
    func: Callable[..., T] = None, context: Callable[[Any], Any] = lambda self: None
) -> Callable[..., T]:
    """
    Decorator to cache the result of a method based on its arguments and context.

    Args:
        func: The function to be decorated.
        context: A function that returns the context for caching.

    Returns:
        A wrapped function that implements caching based on arguments and context.

    Example:
        ```python
        class MyClass:
            def __init__(self):
                self.cache = {}

            @context_aware_cache(context=lambda self: self.some_attribute)
            def my_method(self, arg1, arg2):
                # Method implementation
                pass
        ```
    """
    if func is None:

        def decorator(f: Callable[..., T]) -> Callable[..., T]:
            @wraps(f)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
                context_key = make_hashable(context(self))
                cache_key = hash(make_hashable((f.__name__, args, kwargs, context_key)))
                if cache_key in self.cache:
                    return self.cache[cache_key]
                result = f(self, *args, **kwargs)
                self.cache[cache_key] = result
                return result

            return wrapper

        return decorator
    else:

        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            context_key = make_hashable(context(self))
            cache_key = hash(make_hashable((func.__name__, args, kwargs, context_key)))
            if cache_key in self.cache:
                return self.cache[cache_key]
            result = func(self, *args, **kwargs)
            self.cache[cache_key] = result
            return result

        return wrapper


def make_hashable(obj: Any) -> Any:
    """
    Recursively converts an object into a hashable type.

    Args:
        obj: The object to be converted.

    Returns:
        A hashable representation of the input object.

    Note:
        This function handles various types including immutable types, lists, sets,
        dictionaries, tuples, frozensets, and slices. For complex objects, it falls
        back to string conversion, which may not be ideal for all use cases.
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, set)):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (tuple, frozenset)):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, slice):
        return (obj.start, obj.stop, obj.step)
    else:
        return str(obj)
