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
    """Recursively converts an object into a hashable type."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, set)):
        try:
            # Try direct sorting first
            return tuple(make_hashable(e) for e in sorted(obj))
        except TypeError:
            # Fall back to sorting by hash
            return tuple(
                make_hashable(e)
                for e in sorted(obj, key=lambda x: hash(make_hashable(x)))
            )
    elif isinstance(obj, dict):
        try:
            # Try direct sorting of keys first
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        except TypeError:
            # Fall back to sorting by hash of keys
            return tuple(
                sorted(
                    ((k, make_hashable(v)) for k, v in obj.items()),
                    key=lambda x: hash(make_hashable(x[0])),
                )
            )
    elif isinstance(obj, (tuple, frozenset)):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, slice):
        return (obj.start, obj.stop, obj.step)
    else:
        # For other types, try to get a consistent string representation
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}:{str(obj)}"
