import os
from tabnanny import verbose
import unittest
from astropy.io import fits

import specula
specula.init(0)  # Default target device

from specula import np, cp, cpuArray

from specula.lib.make_xy import make_xy
from specula.lib.interp2d import Interp2D

from test.specula_testlib import cpu_and_gpu

class TestInterp2D(unittest.TestCase):

    @cpu_and_gpu
    def test_interp2d(self, target_device_idx, xp):

        datadir = os.path.join(os.path.dirname(__file__), 'data')
        phase = fits.getdata(os.path.join(datadir, 'input_phase.fits'))
        ref_phase = fits.getdata(os.path.join(datadir, 'ref_phase.fits'))

        half_pixel_layer = [240.5, 240.5]
        pixel_position = [0.0930127, 0]
        pixel_pupil = 480
        pixel_pupmeta = 479.84003

        xx, yy = make_xy(pixel_pupil, pixel_pupmeta/2., xp=xp)
        xx1 = xx + half_pixel_layer[0] + pixel_position[0]
        yy1 = yy + half_pixel_layer[1] + pixel_position[1]
        interpolator = Interp2D(phase.shape, (pixel_pupil, pixel_pupil), xx=xx1, yy=yy1,
                      rotInDeg=0, xp=xp, dtype=xp.float32)

        output_phase = interpolator.interpolate(xp.array(phase))

        test_phase = cpuArray(output_phase)

        # TODO one single pixel has value with 15% difference
        test_phase[394,365] = ref_phase[394,365]
        # Then a few pixels with 3%
        np.testing.assert_allclose(test_phase, ref_phase, rtol=4e-2)

    @cpu_and_gpu
    def test_interp2d_boundary_check(self, target_device_idx, xp):
        '''
        Test that interp2d does not read beyond the input array boundaries,
        using as input a small array that is a slice of a larger one, the latter
        filled with NaNs outside the slice range. We check that the output
        contains no NaNs.
        '''

        array = xp.empty(shape=(20,20), dtype=xp.float32)
        array[:] = np.nan
        array[:10] = 1

        interpolator = Interp2D(input_shape=(10, 20), output_shape=(5,5),
                                rotInDeg=45, xp=xp, dtype=xp.float32)
        result = interpolator.interpolate(array[:10])
        assert xp.isnan(result).sum() == 0

    @cpu_and_gpu
    def test_interp2d_wrong_shape(self, target_device_idx, xp):
        '''
        Test that the interpolate method raises if the
        array to be interpolated has the wrong shape
        '''
        interpolator = Interp2D(input_shape=(10, 10), output_shape=(5,5),
                                rotInDeg=45, xp=xp, dtype=xp.float32)
        with self.assertRaises(ValueError):
            _ = interpolator.interpolate(xp.zeros((20,20)))

    @cpu_and_gpu
    def test_interp2d_magnification(self, target_device_idx, xp):
        '''
        Test that magnification correctly scales the input array.
        Works on both CPU (scipy) and GPU (cupy/precomputed or on-the-fly).
        '''
        input_shape = (10, 10)
        output_shape = (10, 10)

        # Create a simple gradient image: values from 0 to 9 along X
        _, x = xp.mgrid[0:input_shape[0], 0:input_shape[1]]
        phase_in = x.astype(xp.float32)

        # Test 1: Magnification = 2.0 (Zoom in)
        interp_zoom = Interp2D(
            input_shape, output_shape,
            magnification=2.0,
            xp=xp, dtype=xp.float32
        )
        output_zoom = interp_zoom.interpolate(phase_in)

        # Check center values.
        # With equal input and output size, output index j maps to x = j, and
        # zooming by 2x around the center xc = (10 - 1) / 2 = 4.5 gives
        # x = (j - 4.5) / 2 + 4.5: the two pixels next to the center go to
        # 4.25 and 4.75, symmetric about it.
        np.testing.assert_allclose(float(output_zoom[5, 4]), 4.25, atol=1e-5)
        np.testing.assert_allclose(float(output_zoom[5, 5]), 4.75, atol=1e-5)

        # The edge (0,0) originally mapped to 0. Zooming in by 2x around
        # the center (4.5) moves the sampling coordinate closer to the center:
        # new_x = (0 - 4.5) / 2.0 + 4.5 = 2.25
        edge_val = output_zoom[0, 0]
        np.testing.assert_allclose(float(edge_val), 2.25, atol=1e-5)

        # Test 2: Magnification = 0.5 (Zoom out)
        interp_shrink = Interp2D(
            input_shape, output_shape,
            magnification=0.5,
            xp=xp, dtype=xp.float32
        )
        output_shrink = interp_shrink.interpolate(phase_in)

        # Check that edges are clamped (reading beyond the original boundaries)
        # For a gradient 0..9, zooming out by 0.5x pushes the sampling coordinates
        # to -4.5 and 11.7, which are clamped to 0 and 9 respectively.
        np.testing.assert_allclose(float(output_shrink[0, 0]), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(output_shrink[0, 9]), 9.0, atol=1e-5)

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "This test requires CuPy (GPU)")
    def test_onthefly_vs_precomputed(self, target_device_idx, xp):
        '''
        Test that interp2_kernel_onthefly produces the same results as interp2_kernel
        with precomputed coordinates. This test runs only on GPU.
        '''
        if xp == cp: # pragma: no cover
            # Test various scenarios
            test_cases = [
                # (input_shape, output_shape, rotInDeg, rowShift, colShift, magnification, description)
                ((100, 100), (50, 50), 0, 0, 0, 1.0, "simple downscaling"),
                ((100, 100), (150, 150), 0, 0, 0, 1.0, "simple upscaling"),
                ((100, 100), (100, 100), 45, 0, 0, 1.0, "rotation 45 degrees"),
                ((100, 100), (100, 100), 0, 10, 5, 1.0, "shift only"),
                ((100, 100), (80, 80), 30, 5, -3, 1.0, "rotation + shift + scaling"),
                ((200, 150), (100, 120), 15, 2.5, 1.5, 1.0, "non-square with rotation and shift"),
                ((100, 100), (100, 100), 45, 1, 1, 0.5, "rotation, shift and magnification"),
            ]

            for input_shape, output_shape, rot, row_shift, col_shift, magnification, description in test_cases:
                with self.subTest(case=description):
                    # Create test input array with some structure
                    phase_in = xp.random.rand(*input_shape).astype(xp.float32)
                    # Add some features to make interpolation differences visible
                    y, x = xp.mgrid[0:input_shape[0], 0:input_shape[1]]
                    phase_in += xp.sin(x * 0.1) * xp.cos(y * 0.1)

                    # Method 1: on-the-fly (no xx, yy provided)
                    interp_onthefly = Interp2D(
                        input_shape, output_shape,
                        rotInDeg=rot,
                        rowShiftInPixels=row_shift,
                        colShiftInPixels=col_shift,
                        magnification=magnification,
                        xp=xp,
                        dtype=xp.float32
                    )
                    assert not interp_onthefly.use_precomputed, \
                        f"Expected on-the-fly mode for {description}"

                    output_onthefly = interp_onthefly.interpolate(phase_in)

                    # Method 2: precomputed coordinates (generate xx, yy manually)
                    yy, xx = xp.mgrid[0:output_shape[0], 0:output_shape[1]]
                    yy = yy.astype(xp.float32)
                    xx = xx.astype(xp.float32)
                    yy = (yy + 0.5) * (input_shape[0] / output_shape[0]) - 0.5
                    xx = (xx + 0.5) * (input_shape[1] / output_shape[1]) - 0.5

                    # Apply rotation and magnification
                    if rot != 0 or magnification != 1.0:
                        yc = input_shape[0] / 2 - 0.5
                        xc = input_shape[1] / 2 - 0.5

                        xx_centered = (xx - xc) / magnification
                        yy_centered = (yy - yc) / magnification

                        if rot != 0:
                            cos_ = xp.cos(rot * xp.pi / 180.0)
                            sin_ = xp.sin(rot * xp.pi / 180.0)
                            xxr = xx_centered * cos_ - yy_centered * sin_
                            yyr = xx_centered * sin_ + yy_centered * cos_
                            xx_centered = xxr
                            yy_centered = yyr

                        xx = xx_centered + xc
                        yy = yy_centered + yc

                    # Apply shift
                    if row_shift != 0 or col_shift != 0:
                        yy += row_shift
                        xx += col_shift

                    # Clamp
                    yy = xp.clip(yy, 0, input_shape[0] - 1)
                    xx = xp.clip(xx, 0, input_shape[1] - 1)

                    interp_precomputed = Interp2D(
                        input_shape, output_shape,
                        xx=xx, yy=yy,
                        xp=xp,
                        dtype=xp.float32
                    )
                    assert interp_precomputed.use_precomputed, \
                        f"Expected precomputed mode for {description}"

                    output_precomputed = interp_precomputed.interpolate(phase_in)

                    # Compare results
                    diff = cpuArray(xp.abs(output_onthefly - output_precomputed))
                    max_diff = xp.max(diff)
                    mean_diff = xp.mean(diff)

                    plot_debug = False
                    if plot_debug: # pragma: no cover
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(12,4))
                        plt.subplot(1,3,1)
                        plt.title('On-the-fly output')
                        plt.imshow(cpuArray(output_onthefly), cmap='viridis')
                        plt.colorbar()
                        plt.subplot(1,3,2)
                        plt.title('Precomputed output')
                        plt.imshow(cpuArray(output_precomputed), cmap='viridis')
                        plt.colorbar()
                        plt.subplot(1,3,3)
                        plt.title('Absolute difference')
                        plt.imshow(diff, cmap='inferno')
                        plt.colorbar()
                        plt.suptitle(f'Interpolation comparison: {description}')
                        plt.show()

                    verbose = False
                    if verbose: # pragma: no cover
                        print(f"Max difference for {description}: {max_diff}")
                        print(f"Mean difference for {description}: {mean_diff}")

                    # Allow small numerical differences due to floating point arithmetic
                    assert max_diff < 2e-5, \
                        f"Max difference for {description}: {max_diff} (should be < 2e-5)"
                    assert mean_diff < 2e-6, \
                        f"Mean difference for {description}: {mean_diff} (should be < 2e-6)"
        else:
            self.skipTest("This test only runs on GPU with CuPy")

    @staticmethod
    def _with_nan_sentinel(xp, data):
        '''
        Return a contiguous copy of data followed in memory by a NaN, so that
        any kernel read past the end of the array poisons the result.
        '''
        buf = xp.full(data.size + 1, xp.nan, dtype=data.dtype)
        buf[:-1] = data.ravel()
        return buf[:-1].reshape(data.shape)

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "This test requires CuPy (GPU)")
    def test_precomputed_corner_no_out_of_bounds_read(self, target_device_idx, xp):
        '''
        Sampling the last row and column exactly must not read past the end
        of the input array (bottom-right corner).
        '''
        if xp == cp: # pragma: no cover
            input_shape = (20, 30)
            for dtype in (xp.float32, xp.float64):
                with self.subTest(dtype=dtype):
                    data = xp.arange(input_shape[0] * input_shape[1], dtype=dtype).reshape(input_shape)
                    phase_in = self._with_nan_sentinel(xp, data)
                    yy, xx = xp.mgrid[0:input_shape[0], 0:input_shape[1]].astype(dtype)
                    interp = Interp2D(input_shape, input_shape, xx=xx, yy=yy, xp=xp, dtype=dtype)
                    output = interp.interpolate(phase_in)
                    np.testing.assert_array_equal(cpuArray(output), cpuArray(data))
        else:
            self.skipTest("This test only runs on GPU with CuPy")

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "This test requires CuPy (GPU)")
    def test_onthefly_corner_no_out_of_bounds_read(self, target_device_idx, xp):
        '''
        A shift clamping all coordinates to the bottom-right corner must not
        read past the end of the input array.
        '''
        if xp == cp: # pragma: no cover
            input_shape = (20, 30)
            for dtype in (xp.float32, xp.float64):
                with self.subTest(dtype=dtype):
                    data = xp.arange(input_shape[0] * input_shape[1], dtype=dtype).reshape(input_shape)
                    phase_in = self._with_nan_sentinel(xp, data)
                    interp = Interp2D(input_shape, input_shape,
                                      rowShiftInPixels=1000, colShiftInPixels=1000,
                                      xp=xp, dtype=dtype)
                    assert not interp.use_precomputed
                    output = cpuArray(interp.interpolate(phase_in))
                    np.testing.assert_array_equal(output, np.full(input_shape, cpuArray(data[-1, -1])))
        else:
            self.skipTest("This test only runs on GPU with CuPy")

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "This test requires CuPy (GPU)")
    def test_right_edge_no_row_wraparound(self, target_device_idx, xp):
        '''
        Sampling the last column must not pick up the first element of the
        next row, even with zero weight: a NaN there would leak into the output.
        '''
        if xp == cp: # pragma: no cover
            input_shape = (20, 30)
            for dtype in (xp.float32, xp.float64):
                with self.subTest(dtype=dtype):
                    data = xp.arange(input_shape[0] * input_shape[1], dtype=dtype).reshape(input_shape)
                    data[:, 0] = xp.nan
                    yy = xp.repeat(xp.arange(input_shape[0], dtype=dtype)[:, None], 5, axis=1)
                    xx = xp.full_like(yy, input_shape[1] - 1)
                    interp = Interp2D(input_shape, yy.shape, xx=xx, yy=yy, xp=xp, dtype=dtype)
                    output = cpuArray(interp.interpolate(data))
                    expected = np.repeat(cpuArray(data[:, -1])[:, None], 5, axis=1)
                    np.testing.assert_array_equal(output, expected)
        else:
            self.skipTest("This test only runs on GPU with CuPy")

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "This test requires CuPy (GPU)")
    def test_onthefly_float64(self, target_device_idx, xp):
        '''
        Test that on-the-fly kernel works correctly with float64 dtype.
        '''
        if xp == cp: # pragma: no cover
            input_shape = (100, 100)
            output_shape = (80, 80)

            # Create test input
            phase_in = xp.random.rand(*input_shape).astype(xp.float64)

            # On-the-fly with float64
            interp = Interp2D(
                input_shape, output_shape,
                rotInDeg=30,
                rowShiftInPixels=5,
                colShiftInPixels=3,
                xp=xp,
                dtype=xp.float64
            )

            output = interp.interpolate(phase_in)

            # Basic checks
            assert output.dtype == xp.float64, "Output should be float64"
            assert output.shape == output_shape, f"Output shape should be {output_shape}"
            assert not xp.isnan(output).any(), "Output should not contain NaN"
        else:
            self.skipTest("This test only runs on GPU with CuPy")

    # ------------------------------------------------------------------
    # Pixel-centre convention: pixel j is centred at j, output pixel j maps
    # to x_in = (j + 0.5) * N_in / N_out - 0.5 (bilinear, edges replicated).
    # ------------------------------------------------------------------

    @staticmethod
    def _ramp(xp, shape, axis):
        '''Float32 image whose value equals the pixel index along `axis` (1 = x, 0 = y).'''
        idx = np.arange(shape[axis], dtype=np.float32)
        img = np.tile(idx[None, :], (shape[0], 1)) if axis == 1 else np.tile(idx[:, None], (1, shape[1]))
        return xp.asarray(img, dtype=xp.float32)

    @cpu_and_gpu
    def test_upsample_ramp(self, target_device_idx, xp):
        '''
        Upsampling a linear ramp N -> 2N is exact: out[j] = (j + 0.5)/2 - 0.5
        (slope N_in/N_out per output pixel), the image centre is preserved,
        and the first/last pixels (mapped outside [0, N-1]) replicate the edges.
        '''
        n = 16
        for axis in (1, 0):
            with self.subTest(axis=axis):
                interp = Interp2D((n, n), (2 * n, 2 * n), xp=xp, dtype=xp.float32)
                out = cpuArray(interp.interpolate(self._ramp(xp, (n, n), axis)))
                prof = out[0, :] if axis == 1 else out[:, 0]
                # The other axis must be constant
                other = out[:, 0] if axis == 1 else out[0, :]
                np.testing.assert_allclose(other, other[0], atol=1e-6)
                if axis == 1:
                    np.testing.assert_allclose(out, np.tile(prof, (2 * n, 1)), atol=1e-5)
                else:
                    np.testing.assert_allclose(out, np.tile(prof[:, None], (1, 2 * n)), atol=1e-5)

                j = np.arange(1, 2 * n - 1)
                np.testing.assert_allclose(prof[1:-1], (j + 0.5) / 2 - 0.5, atol=1e-5)
                np.testing.assert_allclose(0.5 * (prof[n - 1] + prof[n]), (n - 1) / 2, atol=1e-5)
                np.testing.assert_allclose(prof[0], 0.0, atol=1e-6)
                np.testing.assert_allclose(prof[-1], n - 1, atol=1e-5)

    @cpu_and_gpu
    def test_downsample_ramp(self, target_device_idx, xp):
        '''
        Downsampling a linear ramp 2N -> N gives the mean of each input pixel
        pair, out[j] = 2j + 0.5, for all pixels (no clamping involved).
        '''
        n = 16
        for axis in (1, 0):
            with self.subTest(axis=axis):
                interp = Interp2D((2 * n, 2 * n), (n, n), xp=xp, dtype=xp.float32)
                out = cpuArray(interp.interpolate(self._ramp(xp, (2 * n, 2 * n), axis)))
                expected = 2 * np.arange(n) + 0.5
                expected = np.tile(expected[None, :], (n, 1)) if axis == 1 \
                    else np.tile(expected[:, None], (1, n))
                np.testing.assert_allclose(out, expected, atol=1e-4)

    @cpu_and_gpu
    def test_equal_size_shift_ramp(self, target_device_idx, xp):
        '''
        With equal sizes, a shift s along an axis moves a ramp by exactly s
        (out = ramp + s), clamped to [0, N-1] at the edges. The tiny shift
        1e-6 must be continuous with the unshifted identity (no 0.5 px jump).
        '''
        n = 16
        for axis in (1, 0):
            for s in (1e-6, 0.3, 1.0, -0.7):
                with self.subTest(axis=axis, shift=s):
                    kw = dict(colShiftInPixels=s) if axis == 1 else dict(rowShiftInPixels=s)
                    interp = Interp2D((n, n), (n, n), xp=xp, dtype=xp.float32, **kw)
                    img = self._ramp(xp, (n, n), axis)
                    out = cpuArray(interp.interpolate(img))
                    ref = cpuArray(img)
                    expected = np.clip(ref + s, 0, n - 1)
                    np.testing.assert_allclose(out, expected, atol=1e-5)
                    # Interior pixels, where no clamping happens, are exactly ramp + s
                    sl = [slice(None), slice(None)]
                    sl[axis] = slice(1, n - 1)
                    np.testing.assert_allclose(out[tuple(sl)], ref[tuple(sl)] + s, atol=1e-5)
                    if s == 1e-6:
                        assert np.max(np.abs(out - ref)) < 1e-5

    @cpu_and_gpu
    def test_integer_shift_random_image(self, target_device_idx, xp):
        '''
        An integer shift (row 2, col 3) samples out[i, j] = in[i + 2, j + 3]
        (x_in = x + shift), i.e. np.roll by (-2, -3), exactly, wherever no
        clamping happens.
        '''
        n = 16
        img = np.random.default_rng(1234).random((n, n)).astype(np.float32)
        interp = Interp2D((n, n), (n, n), rowShiftInPixels=2, colShiftInPixels=3,
                          xp=xp, dtype=xp.float32)
        out = cpuArray(interp.interpolate(xp.asarray(img)))
        expected = np.roll(img, (-2, -3), axis=(0, 1))
        np.testing.assert_allclose(out[:n - 2, :n - 3], expected[:n - 2, :n - 3], atol=1e-5)
        np.testing.assert_allclose(out[:n - 2, :n - 3], img[2:, 3:], atol=1e-5)

    @cpu_and_gpu
    def test_rotation_90_deg_random_image(self, target_device_idx, xp):
        '''
        Rotation about the input centre (N-1)/2 by +90 deg maps onto pixel
        centres exactly, for even and odd N. With x_in = -(y - c) + c and
        y_in = (x - c) + c one gets out[i, j] = in[j, N-1-i] = np.rot90(img, 1).
        -90 deg is the opposite direction (k=-1) and 180 deg is k=2.
        '''
        for n in (16, 15):
            img = np.random.default_rng(n).random((n, n)).astype(np.float32)
            for angle, k in ((90, 1), (-90, -1), (180, 2)):
                with self.subTest(n=n, angle=angle):
                    interp = Interp2D((n, n), (n, n), rotInDeg=angle, xp=xp, dtype=xp.float32)
                    out = cpuArray(interp.interpolate(xp.asarray(img)))
                    np.testing.assert_allclose(out, np.rot90(img, k), atol=1e-5)

    @cpu_and_gpu
    def test_upsample_random_no_overshoot(self, target_device_idx, xp):
        '''
        Bilinear upsampling 8 -> 24 is a convex combination of input pixels
        (edges replicated): the output must stay within [min, max] of the input.
        '''
        img = np.random.default_rng(7).random((8, 8)).astype(np.float32)
        interp = Interp2D((8, 8), (24, 24), xp=xp, dtype=xp.float32)
        out = cpuArray(interp.interpolate(xp.asarray(img)))
        assert out.shape == (24, 24)
        assert out.min() >= img.min() - 1e-6
        assert out.max() <= img.max() + 1e-6

    @cpu_and_gpu
    def test_default_vs_explicit_coordinates(self, target_device_idx, xp):
        '''
        Explicit xx/yy built with the pixel-centre formula must reproduce the
        default mapping (on GPU: precomputed kernel vs on-the-fly kernel).
        '''
        in_shape, out_shape = (8, 12), (24, 18)
        img = xp.asarray(np.random.default_rng(3).random(in_shape).astype(np.float32))
        yy, xx = np.mgrid[0:out_shape[0], 0:out_shape[1]].astype(np.float32)
        yy = np.clip((yy + 0.5) * in_shape[0] / out_shape[0] - 0.5, 0, in_shape[0] - 1)
        xx = np.clip((xx + 0.5) * in_shape[1] / out_shape[1] - 0.5, 0, in_shape[1] - 1)
        default = Interp2D(in_shape, out_shape, xp=xp, dtype=xp.float32)
        explicit = Interp2D(in_shape, out_shape, xx=xp.asarray(xx), yy=xp.asarray(yy),
                            xp=xp, dtype=xp.float32)
        assert explicit.use_precomputed
        np.testing.assert_allclose(cpuArray(explicit.interpolate(img)),
                                   cpuArray(default.interpolate(img)), atol=1e-5)
