import sys
import types

import numpy as np
import pytest

from iteras.lib import utils


class FakeCupy12Array:
    '''ndarray.get() with the CuPy 12 signature (no blocking argument)'''
    def __init__(self, data):
        self.data = data
        self.calls = []

    def get(self, stream=None, order='C', out=None):
        self.calls.append(dict(stream=stream, out=out))
        out[...] = self.data


class FakeCupy13Array(FakeCupy12Array):
    '''ndarray.get() with the CuPy >= 13 signature'''
    def get(self, stream=None, order='C', out=None, blocking=True):
        self.calls.append(dict(stream=stream, out=out, blocking=blocking))
        out[...] = self.data


@pytest.fixture
def fake_cupy(monkeypatch):
    def install(version):
        monkeypatch.setitem(sys.modules, 'cupy', types.SimpleNamespace(__version__=version))
        utils._cupy_get_needs_blocking_false.cache_clear()
    yield install
    utils._cupy_get_needs_blocking_false.cache_clear()


@pytest.mark.parametrize('version, array_class, expected_kwargs', [
    ('12.3.0', FakeCupy12Array, {}),
    ('13.0.0', FakeCupy13Array, {'blocking': False}),
    ('14.1.2', FakeCupy13Array, {'blocking': False}),
])
def test_get_async_uses_the_signature_of_the_cupy_version(fake_cupy, version, array_class, expected_kwargs):
    fake_cupy(version)
    value = array_class(np.arange(3.0))
    out = np.zeros(3)
    stream = object()

    utils.get_async(value, out, stream)

    assert value.calls == [dict(stream=stream, out=out, **expected_kwargs)]
    np.testing.assert_array_equal(out, [0.0, 1.0, 2.0])
