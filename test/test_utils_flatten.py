from specula.lib.utils import flatten

def test_flatten_simple():
    data = [1, [2, 3], 4]
    result = list(flatten(data))
    assert result == [1, 2, 3, 4]

def test_flatten_nested():
    data = [[1, [2, [3, 4]], 5], 6]
    result = list(flatten(data))
    assert result == [1, 2, 3, 4, 5, 6]

def test_flatten_nonlist():
    data = 42
    result = list(flatten([data]))
    assert result == [42]

def test_flatten_flat_list():
    data = [1, 2, 3, 4]
    result = list(flatten([data]))
    assert result == [1, 2, 3, 4]