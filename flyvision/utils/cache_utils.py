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
        return tuple(sorted((make_hashable(k), make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (tuple, frozenset)):
        # Tuples and frozensets are already immutable
        return tuple(make_hashable(e) for e in obj)
    else:
        # Fallback: convert to string (not recommended for complex objects)
        return str(obj)
