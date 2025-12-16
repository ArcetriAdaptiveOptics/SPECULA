import specula
specula.init(0)

from specula import cpuArray
from specula.lib.make_mask import make_mask
from test.specula_testlib import cpu_and_gpu_noself


@cpu_and_gpu_noself
def test_circular_mask_shape(xp):
    """Test that the mask has the correct shape and values for circular mask."""
    np_size = 10
    mask = make_mask(np_size, obsratio=0.0, diaratio=1.0, xp=xp)
    assert mask.shape == (np_size, np_size)
    assert set(cpuArray(xp.unique(mask))).issubset({0, 1})


@cpu_and_gpu_noself
def test_square_mask(xp):
    """Test that square mask differs from circular mask."""
    np_size = 10
    mask_circular = make_mask(np_size, obsratio=0.0, diaratio=1.0, square=False, xp=xp)
    mask_square = make_mask(np_size, obsratio=0.0, diaratio=1.0, square=True, xp=xp)
    assert mask_square.shape == (np_size, np_size)
    assert not xp.array_equal(mask_circular, mask_square)


@cpu_and_gpu_noself
def test_inverse_mask(xp):
    """Test that inverse flag correctly inverts the mask."""
    np_size = 10
    mask = make_mask(np_size, obsratio=0.0, diaratio=1.0, xp=xp)
    mask_inv = make_mask(np_size, obsratio=0.0, diaratio=1.0, inverse=True, xp=xp)
    assert xp.array_equal(mask + mask_inv, xp.ones_like(mask))


@cpu_and_gpu_noself
def test_center_on_pixel(xp):
    """Test that centeronpixel moves the center to nearest pixel without errors."""
    np_size = 10
    mask1 = make_mask(np_size, xc=0.3, yc=-0.3, centeronpixel=False, xp=xp)
    mask2 = make_mask(np_size, xc=0.3, yc=-0.3, centeronpixel=True, xp=xp)
    assert not xp.array_equal(mask1, mask2)


@cpu_and_gpu_noself
def test_get_idx_returns_tuple(xp):
    """Test that get_idx returns a tuple (mask, idx)."""
    np_size = 10
    mask, idx = make_mask(np_size, get_idx=True, xp=xp)
    assert isinstance(mask, xp.ndarray)
    assert isinstance(idx, tuple)
    assert all(isinstance(a, xp.ndarray) for a in idx)


@cpu_and_gpu_noself
def test_spider_arms_presence(xp):
    """Test that adding spider arms modifies the mask."""
    np_size = 20
    mask_no_spider = make_mask(np_size, obsratio=0.0, diaratio=1.0, xp=xp)
    mask_spider = make_mask(np_size, obsratio=0.0, diaratio=1.0, spider=True,
                            spider_width=2, n_petals=4, xp=xp)
    assert not xp.array_equal(mask_no_spider, mask_spider)
    assert set(cpuArray(xp.unique(mask_spider))).issubset({0, 1})


@cpu_and_gpu_noself
def test_obsratio_applies_central_obstruction(xp):
    """Test that obsratio creates a central obstruction in the mask."""
    np_size = 20
    mask_no_obs = make_mask(np_size, obsratio=0.0, xp=xp)
    mask_obs = make_mask(np_size, obsratio=0.5, xp=xp)
    assert xp.sum(mask_obs) < xp.sum(mask_no_obs)
