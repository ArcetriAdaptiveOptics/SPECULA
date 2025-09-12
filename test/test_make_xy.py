import specula
specula.init(0)

from specula.lib.make_xy import make_xy
from test.specula_testlib import cpu_and_gpu_noself


@cpu_and_gpu_noself
def test_make_xy_sampling_too_small(xp):
    """Sampling <= 1 should raise ValueError."""
    try:
        make_xy(1, 1.0, xp=xp)
        raise Exception('ValueError not raised')
    except ValueError:
        pass


@cpu_and_gpu_noself
def test_make_xy_vector_output(xp):
    """Vector output should return 1D array."""
    vec = make_xy(5, 1.0, xp=xp, vector=True)
    assert vec.ndim == 1
    assert vec.shape[0] == 5


@cpu_and_gpu_noself
def test_make_xy_cartesian_output_shape(xp):
    """Cartesian output should return square 2D arrays."""
    x, y = make_xy(4, 1.0, xp=xp)
    assert x.shape == (4, 4)
    assert y.shape == (4, 4)
    # Ensure symmetry of axes
    assert xp.allclose(x, y.T, atol=1e-6)


@cpu_and_gpu_noself
def test_make_xy_polar_conversion(xp):
    """Polar output should produce r and theta arrays."""
    x, y = make_xy(5, 1.0, xp=xp, polar=True)
    xx, yy = make_xy(5, 1.0, xp=xp)
    r = xp.sqrt(xx**2 + yy**2)
    theta = xp.arctan2(yy, xx)
    xp.testing.assert_allclose(r, x, rtol=1e-6) 
    xp.testing.assert_allclose(theta, y, rtol=1e-6) 

@cpu_and_gpu_noself
def test_make_xy_quarter_domain(xp):
    """Quarter flag should reduce output size."""
    x, y = make_xy(5, 1.0, xp=xp, quarter=True)
    size = (5 + 1) // 2
    assert x.shape == (size, size)
    assert y.shape == (size, size)
    # Check all values are >= 0
    assert xp.all(x >= 0)
    assert xp.all(y >= 0)


@cpu_and_gpu_noself
def test_make_xy_fft_ordering(xp):
    """FFT ordering should roll array elements appropriately."""
    vec_even = make_xy(4, 1.0, xp=xp, vector=True, fft=True)
    vec_odd = make_xy(5, 1.0, xp=xp, vector=True, fft=True)
    assert vec_even.shape[0] == 4
    assert vec_odd.shape[0] == 5
    # FFT ordering should change the first element for even/odd
    assert not xp.allclose(vec_even, make_xy(4, 1.0, xp=xp, vector=True))
    assert not xp.allclose(vec_odd, make_xy(5, 1.0, xp=xp, vector=True))


@cpu_and_gpu_noself
def test_make_xy_dtype(xp):
    """make_xy should respect the dtype parameter."""
    dtype = xp.float32
    x, y = make_xy(5, 1.0, xp=xp, dtype=dtype)
    assert x.dtype == dtype
    assert y.dtype == dtype
