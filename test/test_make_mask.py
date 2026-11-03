import numpy as np

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


# ---------------------------------------------------------------------------
# Spider arms: width, symmetry, off-centre pupil, mean width
# ---------------------------------------------------------------------------

def _spider_pixels(xp, np_size, **kwargs):
    """Boolean host array, True where the spiders remove a pupil pixel."""
    no_spider = cpuArray(make_mask(np_size, xp=xp, **kwargs))
    kwargs = dict(kwargs, spider=True)
    with_spider = cpuArray(make_mask(np_size, xp=xp, **kwargs))
    return (no_spider - with_spider) > 0


def _expected_indices(np_size, width, centre=None):
    """Indices j whose centre is closer than width / 2 to the axis (border excluded)."""
    if centre is None:
        centre = (np_size - 1) / 2
    return [j for j in range(np_size) if abs(j - centre) < width / 2 - 1e-6]


@cpu_and_gpu_noself
def test_spider_width_even_frame_two_pixels(xp):
    """Horizontal arm of spider_width=2 on N=64 covers exactly rows 31 and 32.

    The old code traced arms with a 1.2x width buffer and int() truncation, so
    they were about twice spider_width (4 px here).
    """
    spid = _spider_pixels(xp, 64, obsratio=0.0, diaratio=1.0,
                          spider_width=2, n_petals=4, angle_offset=0)
    assert list(np.nonzero(spid[:, 64 - 6])[0]) == [31, 32]


@cpu_and_gpu_noself
def test_spider_width_exact_rows(xp):
    """The arm occupies exactly the rows whose centre is within w/2 of the axis.

    Pixels exactly on the arm border are excluded on both sides (N=64, w=3 gives
    rows 31, 32 only). The old code gave arms ~1.2x to 2x wider and asymmetric
    for odd widths.
    """
    for np_size in (64, 65):
        for width in (1, 2, 3, 4, 5):
            spid = _spider_pixels(xp, np_size, obsratio=0.0, diaratio=1.0,
                                  spider_width=width, n_petals=4, angle_offset=0)
            rows = list(np.nonzero(spid[:, np_size - 6])[0])
            assert rows == _expected_indices(np_size, width), (np_size, width)
    # Explicit cases
    spid = _spider_pixels(xp, 64, obsratio=0.0, diaratio=1.0,
                          spider_width=3, n_petals=4, angle_offset=0)
    assert list(np.nonzero(spid[:, 58])[0]) == [31, 32]
    spid = _spider_pixels(xp, 64, obsratio=0.0, diaratio=1.0,
                          spider_width=4, n_petals=4, angle_offset=0)
    assert list(np.nonzero(spid[:, 58])[0]) == [30, 31, 32, 33]
    spid = _spider_pixels(xp, 65, obsratio=0.0, diaratio=1.0,
                          spider_width=3, n_petals=4, angle_offset=0)
    assert list(np.nonzero(spid[:, 59])[0]) == [31, 32, 33]


@cpu_and_gpu_noself
def test_spider_symmetry(xp):
    """Spider masks have the symmetries of the arm layout, for any width/parity.

    4 petals at 0 and 45 deg: x flip, y flip, transposition. 6 petals at 0 deg
    and 3 petals at 90 deg: flips (x and y / x only). 8 petals at 22.5 deg: x, y
    flips and transposition. The old code failed e.g. for w=3 with 4, 6 and 3
    petals (truncation, arms always at the array centre).
    """
    # (n_petals, angle_offset, symmetries)
    layouts = [
        (4, 0, ('x', 'y', 't')),
        (4, 45, ('x', 'y', 't')),
        (6, 0, ('x', 'y')),
        (3, 90, ('x',)),
        (8, 22.5, ('x', 'y', 't')),
    ]
    for np_size in (32, 33, 64, 65):
        for width in (1, 2, 3, 4, 5, 2.5):
            for n_petals, angle, syms in layouts:
                mask = cpuArray(make_mask(np_size, obsratio=0.1, diaratio=1.0, xp=xp,
                                          spider=True, spider_width=width,
                                          n_petals=n_petals, angle_offset=angle))
                msg = (np_size, width, n_petals, angle)
                if 'x' in syms:
                    np.testing.assert_array_equal(mask, mask[:, ::-1], err_msg=str(msg))
                if 'y' in syms:
                    np.testing.assert_array_equal(mask, mask[::-1, :], err_msg=str(msg))
                if 't' in syms:
                    np.testing.assert_array_equal(mask, mask.T, err_msg=str(msg))


@cpu_and_gpu_noself
def test_spider_follows_pupil_centre(xp):
    """Arms are centred on the pupil centre (xc, yc), not on the array centre.

    N=64, diaratio=0.8: the pupil centre is at (N-1)/2 + xc*N/2 (column) and
    (N-1)/2 + yc*N/2 (row). With xc=0.2 this is 37.9, so the vertical arm of
    width 2 covers columns 37, 38. The old code ignored xc, yc and put the arms
    at columns/rows 31, 32.
    """
    np_size = 64
    for xc, yc in ((0.2, 0.0), (0.2, -0.15), (0.0, -0.15)):
        spid = _spider_pixels(xp, np_size, obsratio=0.0, diaratio=0.8, xc=xc, yc=yc,
                              spider_width=2, n_petals=4, angle_offset=0)
        cx = (np_size - 1) / 2 + xc * np_size / 2
        cy = (np_size - 1) / 2 + yc * np_size / 2
        exp_cols = _expected_indices(np_size, 2, cx)
        exp_rows = _expected_indices(np_size, 2, cy)
        if xc == 0.2:
            assert exp_cols == [37, 38]
        # Vertical arms, well away from the centre (pupil radius is 25.6 px)
        for row in (int(round(cy)) + 18, int(round(cy)) - 18):
            assert list(np.nonzero(spid[row, :])[0]) == exp_cols, (xc, yc, row)
        # Horizontal arms
        for col in (int(round(cx)) + 18, int(round(cx)) - 18):
            assert list(np.nonzero(spid[:, col])[0]) == exp_rows, (xc, yc, col)


@cpu_and_gpu_noself
def test_spider_mean_width(xp):
    """Mean arm width (removed pixels / arm length) is close to spider_width.

    400 px pupil, obsratio 0.28, 6 petals: total arm length in the annulus is
    6 * (0.5 - 0.14) * 400 px. Measured 4.83 for w=5 (axis-aligned arms through a
    half-integer centre only have an even pixel count). The old code gave about
    5.1 for w=3, i.e. arms too wide.
    """
    np_size = 400
    obsratio = 0.28
    n_petals = 6
    length = n_petals * (0.5 - obsratio / 2) * np_size
    for width, lo, hi in ((5, 4.5, 5.2), (3, 2.5, 3.2)):
        spid = _spider_pixels(xp, np_size, obsratio=obsratio, diaratio=1.0,
                              spider_width=width, n_petals=n_petals, angle_offset=0)
        mean_width = spid.sum() / length
        assert lo < mean_width < hi, (width, mean_width)
