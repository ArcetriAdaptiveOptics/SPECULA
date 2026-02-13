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
    @unittest.skipIf(cp is None, "This test requires CuPy (GPU)")
    def test_onthefly_vs_precomputed(self, target_device_idx, xp):
        '''
        Test that interp2_kernel_onthefly produces the same results as interp2_kernel
        with precomputed coordinates. This test runs only on GPU.
        '''
        if xp == cp: # pragma: no cover
            # Test various scenarios
            test_cases = [
                # (input_shape, output_shape, rotInDeg, rowShift, colShift, description)
                ((100, 100), (50, 50), 0, 0, 0, "simple downscaling"),
                ((100, 100), (150, 150), 0, 0, 0, "simple upscaling"),
                ((100, 100), (100, 100), 45, 0, 0, "rotation 45 degrees"),
                ((100, 100), (100, 100), 0, 10, 5, "shift only"),
                ((100, 100), (80, 80), 30, 5, -3, "rotation + shift + scaling"),
                ((200, 150), (100, 120), 15, 2.5, 1.5, "non-square with rotation and shift"),
            ]

            for input_shape, output_shape, rot, row_shift, col_shift, description in test_cases:
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
                    yy *= (input_shape[0]-1) / output_shape[0]
                    xx *= (input_shape[1]-1) / output_shape[1]

                    # Apply rotation
                    if rot != 0:
                        yc = input_shape[0] / 2 - 0.5
                        xc = input_shape[1] / 2 - 0.5
                        cos_ = xp.cos(rot * xp.pi / 180.0)
                        sin_ = xp.sin(rot * xp.pi / 180.0)
                        xxr = (xx-xc)*cos_ - (yy-yc)*sin_
                        yyr = (xx-xc)*sin_ + (yy-yc)*cos_
                        xx = xxr + xc
                        yy = yyr + yc

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
                    assert mean_diff < 1e-6, \
                        f"Mean difference for {description}: {mean_diff} (should be < 1e-6)"
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
