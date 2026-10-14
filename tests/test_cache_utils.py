from flyvis.utils.cache_utils import context_aware_cache, make_hashable


class Responses:
    def __init__(self, names):
        self.names = names
        self.cache = {}

    @context_aware_cache(context=lambda self: self.names)
    def responses(self, *args):
        return list(self.names)


def test_context_aware_cache_respects_order():
    responses = Responses(["a", "b", "c"])
    assert responses.responses() == ["a", "b", "c"]
    responses.names = ["c", "b", "a"]
    assert responses.responses() == ["c", "b", "a"]
    assert responses.responses([2, 1]) == ["c", "b", "a"]
    responses.names = ["a", "b", "c"]
    assert responses.responses([1, 2]) == ["a", "b", "c"]


def test_make_hashable():
    assert make_hashable([1, 2]) != make_hashable([2, 1])
    assert make_hashable((1, 2)) != make_hashable((2, 1))
    assert make_hashable({1, 2}) == make_hashable({2, 1})
    assert make_hashable({"a": 1, "b": 2}) == make_hashable({"b": 2, "a": 1})
