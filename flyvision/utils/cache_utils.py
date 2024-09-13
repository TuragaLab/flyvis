from functools import wraps


def context_aware_cache(func=None, context=lambda self: None):
    """Decorator to cache the result of a method based on its arguments and context."""
    if func is None:
        # The decorator is called with arguments
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                context_key = make_hashable(context(self))
                cache_key = make_hashable((func.__name__, args, kwargs, context_key))
                if cache_key in self.cache:
                    return self.cache[cache_key]
                result = func(self, *args, **kwargs)
                self.cache[cache_key] = result
                return result

            return wrapper

        return decorator
    else:
        # The decorator is called without arguments
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            context_key = make_hashable(context(self))
            cache_key = make_hashable((func.__name__, args, kwargs, context_key))
            if cache_key in self.cache:
                return self.cache[cache_key]
            result = func(self, *args, **kwargs)
            self.cache[cache_key] = result
            return result

        return wrapper


def make_hashable(obj):
    """Recursively converts an object into a hashable type."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        # Immutable and hashable types
        return obj
    elif isinstance(obj, (list, set)):
        # Convert lists and sets to tuples
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        # Convert dictionaries to tuples of sorted key-value pairs
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (tuple, frozenset)):
        # Tuples and frozensets are already immutable
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, slice):
        # Convert slices to tuples
        return (obj.start, obj.stop, obj.step)
    else:
        # Fallback: convert to string (not recommended for complex objects)
        return str(obj)
