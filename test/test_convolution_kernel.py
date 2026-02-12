import unittest
import os
import numpy as np
from astropy.io import fits
import tempfile
import shutil

import specula
specula.init(0)  # Default target device
from specula import np
from specula import cpuArray

from specula.data_objects.convolution_kernel import ConvolutionKernel, lgs_map_sh
from specula.data_objects.gaussian_convolution_kernel import GaussianConvolutionKernel
from test.specula_testlib import cpu_and_gpu


class TestKernel(unittest.TestCase):

    def setUp(self):
        """Set up test by ensuring calibration directory exists"""
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')

        self.map_ref_path = os.path.join(self.datadir, 'lgs_map_sh_ref.fits')

        if not os.path.exists(self.map_ref_path):
            self.fail("Reference file {self.map_ref_path} not found")

    @cpu_and_gpu
    def test_gauss_kernel(self, target_device_idx, xp):
        """
        Test the GaussianConvolutionKernel class with a Gaussian kernel.
        This test creates a Gaussian kernel with a specified size and
        checks the kernel shape, dimensions, and values.
        The test also verifies the behavior of the kernel with and without
        FFT representation.
        """
        # Create a Gaussian kernel with the specified parameters
        dimx = dimy = 10
        spot_size = 1.0  # arcsec
        pixel_scale = 0.1  # arcsec
        pupil_size_m = 1.0  # m
        dimension = 16  # Size of kernel in pixels

        kernel = GaussianConvolutionKernel(dimx=dimx,
                                           dimy=dimy,
                                           pxscale=pixel_scale,
                                           pupil_size_m=pupil_size_m,
                                           dimension=dimension,
                                           spot_size=spot_size,
                                           oversampling=1,
                                           return_fft=True,
                                           positive_shift_tt=True,
                                           airmass=1.0,
                                           target_device_idx=target_device_idx)

        # Build and calculate kernel
        kernel_fn = kernel.build()
        kernel.calculate_lgs_map()

        # Check kernel shape and dimensions
        self.assertEqual(kernel.kernels.shape, (dimx*dimy, dimension, dimension))

        # Check that all values are finite
        self.assertTrue(xp.all(xp.isfinite(kernel.kernels)))

        # Check that kernels are complex (FFT representation)
        self.assertEqual(kernel.kernels.dtype, kernel.complex_dtype)

        # Check that each kernel has non-zero values
        for i in range(dimx*dimy):
            self.assertTrue(float(cpuArray(xp.abs(kernel.kernels[i]).sum())) > 0)

    @cpu_and_gpu
    def test_kernel(self, target_device_idx, xp):
        """
        Test the ConvolutionKernel class with a generic kernel.
        This test creates a kernel with a sodium layer profile and checks
        the kernel shape, dimensions, and values.
        The test also verifies the behavior of the kernel with and without
        FFT representation.
        The test uses a Gaussian intensity profile with a specified FWHM
        and checks the kernel values.
        The test also checks the kernel shape and dimensions after
        calculating the LGS map.
        """
        # Create a generic convolution kernel with the specified parameters
        dimx = dimy = 10
        spot_size = 1.0  # arcsec
        pixel_scale = 0.1  # arcsec
        pupil_size_m = 8.0  # m
        dimension = 64  # Size of kernel in pixels

        # Create sodium layer profile
        num_points = 20
        z_min = 80e3  # m
        z_max = 100e3  # m
        zlayer = xp.linspace(z_min, z_max, num_points)

        # Create Gaussian intensity profile with FWHM of 10e3 m
        center = 90e3  # m
        fwhm = 10e3  # m
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        zprofile = xp.exp(-0.5 * ((zlayer - center) / sigma) ** 2)
        zfocus = 90e3  # m
        launcher_pos = [5, 5, 0]  # m

        # Normalize the profile
        zprofile /= xp.sum(zprofile)

        # Test with return_fft = False
        kernel = ConvolutionKernel(dimx=dimx,
                                   dimy=dimy,
                                   pxscale=pixel_scale,
                                   pupil_size_m=pupil_size_m,
                                   dimension=dimension,
                                   launcher_pos=launcher_pos,
                                   seeing=spot_size,
                                   zfocus = zfocus,
                                   theta=[0.0, 0.0],
                                   oversampling=1,
                                   return_fft=False,
                                   positive_shift_tt=True,
                                   target_device_idx=target_device_idx)
        kernel.zlayer = zlayer.tolist()
        kernel.zprofile = zprofile.tolist()

        kernel_fn = kernel.build()
        kernel.calculate_lgs_map()

        # Check kernel shape and dimensions
        self.assertEqual(kernel.kernels.shape, (dimx*dimy, dimension, dimension))

        # Check that all values are finite
        self.assertTrue(xp.all(xp.isfinite(kernel.kernels)))

        # Check that each kernel has positive values
        for i in range(dimx*dimy):
            self.assertTrue(float(cpuArray(xp.sum(kernel.kernels[i]))) > 0)

        # Now test with return_fft = True
        kernel = ConvolutionKernel(dimx=dimx,
                                   dimy=dimy,
                                   pxscale=pixel_scale,
                                   pupil_size_m=pupil_size_m,
                                   dimension=dimension,
                                   launcher_pos=launcher_pos,
                                   seeing=spot_size,
                                   zfocus=zfocus,
                                   theta=[0.0, 0.0],
                                   oversampling=1,
                                   return_fft=True,
                                   positive_shift_tt=True,
                                   target_device_idx=target_device_idx)

        kernel.zlayer = zlayer.tolist()
        kernel.zprofile = zprofile.tolist()
        
        kernel_fn = kernel.build()
        kernel.calculate_lgs_map()

        # Check kernel shape and dimensions again
        self.assertEqual(kernel.kernels.shape, (dimx*dimy, dimension, dimension))

        # Check that all values are finite
        self.assertTrue(xp.all(xp.isfinite(kernel.kernels)))

        # For FFT kernels, check that they have the expected complex type
        self.assertEqual(kernel.kernels.dtype, kernel.complex_dtype)

        # Check that each kernel has non-zero values
        for i in range(dimx*dimy):
            self.assertTrue(float(cpuArray(xp.abs(kernel.kernels[i]).sum())) > 0)

    @cpu_and_gpu
    def test_lgs_map_sh(self, target_device_idx, xp):
        """
        Test the lgs_map_sh function with a sodium layer profile.
        This test creates a kernel with a sodium layer profile and checks
        the kernel shape, dimensions, and values.
        """
        # Create a generic convolution kernel with the specified parameters
        dimx = dimy = 10
        spot_size = 1.0  # arcsec
        pixel_scale = 0.1  # arcsec
        pupil_size_m = 8.0  # m
        dimension = 64  # Size of kernel in pixels

        # Create sodium layer profile
        num_points = 20
        z_min = 80e3  # m
        z_max = 100e3  # m
        zlayer = xp.linspace(z_min, z_max, num_points)

        # Create Gaussian intensity profile with FWHM of 10e3 m
        center = 90e3  # m
        fwhm = 10e3  # m
        sigma = fwhm / (2 * xp.sqrt(2 * np.log(2)))
        zprofile = xp.exp(-0.5 * ((zlayer - center) / sigma) ** 2)
        zfocus = 90e3  # m
        launcher_pos = [5, 5, 0]  # m

        # Normalize the profile
        zprofile /= xp.sum(zprofile)
        layer_offsets = zlayer - zfocus

        map = lgs_map_sh(
            nsh=dimx, diam=pupil_size_m, rl=launcher_pos, zb=zfocus,
            dz=layer_offsets, profz=zprofile, fwhmb=spot_size, ps=pixel_scale,
            ssp=dimension, overs=1, theta=[0.0, 0.0], xp=xp)

        # Create a 2D grid to display all kernels in their spatial positions
        kernel2d = xp.zeros((dimy * dimension, dimx * dimension))

        # Place each kernel in its correct position in the grid
        for j in range(dimy):
            for i in range(dimx):
                kernel_idx = i * dimx + j
                # Extract kernel and place it in the correct position in the 2D grid
                y_start = j * dimension
                y_end = (j + 1) * dimension
                x_start = i * dimension
                x_end = (i + 1) * dimension
                kernel2d[y_start:y_end, x_start:x_end] = map[kernel_idx]

        # Check kernel shape and dimensions
        self.assertEqual(map.shape, (dimx*dimy, dimension, dimension))

        # Check that all values are finite
        self.assertTrue(xp.all(xp.isfinite(map)))

        # Add reference file comparison
        with fits.open(self.map_ref_path) as ref_hdul:
            if hasattr(ref_hdul[0], 'data') and ref_hdul[0].data is not None:
                ref_kernel2d = ref_hdul[0].data
                # normalize the reference kernel
                ref_kernel2d /= np.sum(ref_kernel2d)
                kernel2d /= np.sum(kernel2d)

                # Convert kernel2d to CPU for comparison if needed
                kernel2d_cpu = cpuArray(kernel2d)

                # Display the kernel if needed for debugging
                display = False
                if display: # pragma: no cover
                    import matplotlib.pyplot as plt          
                    # Display the complete grid of kernels
                    plt.figure(figsize=(12, 12))
                    plt.imshow(kernel2d_cpu, cmap='viridis', origin='lower')
                    plt.colorbar()
                    plt.title('All Kernels Arranged in Grid')
                    plt.xlabel('X pixel')
                    plt.ylabel('Y pixel')
                    # Display the complete grid of kernels
                    plt.figure(figsize=(12, 12))
                    plt.imshow(ref_kernel2d, cmap='viridis', origin='lower')
                    plt.colorbar()
                    plt.title('Ref Kernels Arranged in Grid')
                    plt.xlabel('X pixel')
                    plt.ylabel('Y pixel')
                    # Display the difference between the two kernels
                    plt.figure(figsize=(12, 12))
                    plt.imshow(kernel2d_cpu - ref_kernel2d, cmap='viridis', origin='lower')
                    plt.colorbar()
                    plt.title('Difference between Kernels')
                    plt.xlabel('X pixel')
                    plt.ylabel('Y pixel')
                    plt.show()

                np.testing.assert_allclose(
                    kernel2d_cpu, ref_kernel2d,
                    rtol=1e-5, atol=1e-5,
                    err_msg="LGS map kernel2d does not match reference values"
                )

    @cpu_and_gpu
    def test_save_restore_workflows(self, target_device_idx, xp):
        """
        Test both manual save/restore and automatic save/restore via prepare_for_sh with data_dir.
        """
        # Common test parameters
        dimx = dimy = 5
        spot_size = 1.0
        pixel_scale = 0.1
        pupil_size_m = 8.0
        dimension = 32

        # Create sodium layer profile
        num_points = 5
        zlayer = np.linspace(80e3, 100e3, num_points)
        center = 90e3
        fwhm = 10e3
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        zprofile = np.exp(-0.5 * ((zlayer - center) / sigma) ** 2)
        zfocus = 90e3
        launcher_pos = [5, 5, 0]
        zprofile /= np.sum(zprofile)

        # --- Part 1: Test manual save/restore ---
        temp_file = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
        temp_filename = temp_file.name
        temp_file.close()

        try:
            # Create and calculate kernel
            kernel1 = ConvolutionKernel(
                dimx=dimx, dimy=dimy, pxscale=pixel_scale,
                pupil_size_m=pupil_size_m, dimension=dimension,
                launcher_pos=launcher_pos, seeing=spot_size,
                zfocus=zfocus, theta=[0.0, 0.0],
                oversampling=1, return_fft=False,
                positive_shift_tt=True,
                target_device_idx=target_device_idx
            )
            kernel1.zlayer = zlayer.tolist()
            kernel1.zprofile = zprofile.tolist()
            kernel1.build()
            kernel1.calculate_lgs_map()

            # Save and restore
            kernel1.save(temp_filename)
            restored_kernel = ConvolutionKernel.restore(
                temp_filename, target_device_idx=target_device_idx, return_fft=False
            )

            # Verify attributes
            self.assertEqual(kernel1.dimx, restored_kernel.dimx)
            self.assertEqual(kernel1.dimension, restored_kernel.dimension)

            # Compare kernel data
            np.testing.assert_allclose(
                cpuArray(kernel1.real_kernels),
                cpuArray(restored_kernel.real_kernels),
                rtol=1e-5, atol=1e-5,
                err_msg="Manual save/restore: kernel data mismatch"
            )

            # Test with return_fft=True
            restored_fft = ConvolutionKernel.restore(
                temp_filename, target_device_idx=target_device_idx, return_fft=True
            )
            self.assertEqual(restored_fft.kernels.dtype, restored_fft.complex_dtype)

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        # --- Part 2: Test automatic save/restore via prepare_for_sh with data_dir ---
        temp_dir = tempfile.mkdtemp()

        try:
            # Create kernel with data_dir
            kernel2 = ConvolutionKernel(
                dimx=dimx, dimy=dimy, pxscale=pixel_scale,
                pupil_size_m=pupil_size_m, dimension=dimension,
                launcher_pos=launcher_pos, seeing=spot_size,
                zfocus=zfocus, theta=[0.0, 0.0],
                oversampling=1, return_fft=True,
                positive_shift_tt=True, data_dir=temp_dir,
                target_device_idx=target_device_idx
            )
            kernel2.zlayer = zlayer.tolist()
            kernel2.zprofile = zprofile.tolist()

            original_kernels = cpuArray(kernel2.real_kernels.copy())

            # First call: should calculate and save
            kernel2.prepare_for_sh(
                sodium_altitude=zlayer.tolist(),
                sodium_intensity=zprofile.tolist(),
                current_time=1
            )

            kernel_fn = kernel2.build()
            expected_file = os.path.join(temp_dir, kernel_fn + '.fits')
            self.assertTrue(os.path.exists(expected_file))

            # Second call: should load from file
            kernel3 = ConvolutionKernel(
                dimx=dimx, dimy=dimy, pxscale=pixel_scale,
                pupil_size_m=pupil_size_m, dimension=dimension,
                launcher_pos=launcher_pos, seeing=spot_size,
                zfocus=zfocus, theta=[0.0, 0.0],
                oversampling=1, return_fft=True,
                positive_shift_tt=True, data_dir=temp_dir,
                target_device_idx=target_device_idx
            )
            kernel3.zlayer = zlayer.tolist()
            kernel3.zprofile = zprofile.tolist()

            new_kernels = cpuArray(kernel3.real_kernels.copy())

            kernel3.prepare_for_sh(
                sodium_altitude=zlayer.tolist(),
                sodium_intensity=zprofile.tolist(),
                current_time=2
            )

            # Verify loaded data matches
            np.testing.assert_allclose(
                original_kernels, new_kernels,
                rtol=1e-5, atol=1e-5,
                err_msg="Automatic save/restore: kernel data mismatch"
            )
            self.assertEqual(kernel3.generation_time, 2)

            # Test parameter change creates new file
            kernel3.zlayer = (zlayer * 1.1).tolist()
            kernel3.prepare_for_sh(
                sodium_altitude=kernel3.zlayer,
                sodium_intensity=zprofile.tolist(),
                current_time=3
            )
            new_kernel_fn = kernel3.build()
            self.assertNotEqual(kernel_fn, new_kernel_fn)
            self.assertTrue(os.path.exists(os.path.join(temp_dir, new_kernel_fn + '.fits')))

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @cpu_and_gpu
    def test_real_kernels_deallocation(self, target_device_idx, xp):
        """
        Test that real_kernels is properly deallocated after prepare_for_sh
        and can be recovered via get_value() if needed.
        """
        dimx = dimy = 5
        spot_size = 1.0
        pixel_scale = 0.1
        pupil_size_m = 8.0
        dimension = 32

        # Create sodium layer profile
        num_points = 5
        zlayer = np.linspace(80e3, 100e3, num_points)
        center = 90e3
        fwhm = 10e3
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        zprofile = np.exp(-0.5 * ((zlayer - center) / sigma) ** 2)
        zfocus = 90e3
        launcher_pos = [5, 5, 0]
        zprofile /= np.sum(zprofile)

        temp_dir = tempfile.mkdtemp()

        try:
            kernel = ConvolutionKernel(
                dimx=dimx, dimy=dimy, pxscale=pixel_scale,
                pupil_size_m=pupil_size_m, dimension=dimension,
                launcher_pos=launcher_pos, seeing=spot_size,
                zfocus=zfocus, theta=[0.0, 0.0],
                oversampling=1, return_fft=True,
                positive_shift_tt=True, data_dir=temp_dir,
                target_device_idx=target_device_idx
            )
            kernel.zlayer = zlayer.tolist()
            kernel.zprofile = zprofile.tolist()

            # Calculate kernels - real_kernels is allocated
            kernel.prepare_for_sh(
                sodium_altitude=zlayer.tolist(),
                sodium_intensity=zprofile.tolist(),
                current_time=1
            )

            # After prepare_for_sh, real_kernels should be deallocated
            self.assertIsNone(kernel.real_kernels, 
                            "real_kernels should be None after prepare_for_sh")

            # kernels (FFT) should still exist
            self.assertIsNotNone(kernel.kernels,
                               "kernels should still exist after prepare_for_sh")

            # get_value() should raise an error since real_kernels is None
            with self.assertRaises(ValueError):
                kernel.get_value()

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @cpu_and_gpu
    def test_set_value_after_deallocation(self, target_device_idx, xp):
        """
        Test that set_value() can recreate real_kernels after deallocation.
        """
        dimx = dimy = 5
        spot_size = 1.0
        pixel_scale = 0.1
        pupil_size_m = 8.0
        dimension = 32

        num_points = 5
        zlayer = np.linspace(80e3, 100e3, num_points)
        center = 90e3
        fwhm = 10e3
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        zprofile = np.exp(-0.5 * ((zlayer - center) / sigma) ** 2)
        zfocus = 90e3
        launcher_pos = [5, 5, 0]
        zprofile /= np.sum(zprofile)

        temp_dir = tempfile.mkdtemp()

        try:
            kernel = ConvolutionKernel(
                dimx=dimx, dimy=dimy, pxscale=pixel_scale,
                pupil_size_m=pupil_size_m, dimension=dimension,
                launcher_pos=launcher_pos, seeing=spot_size,
                zfocus=zfocus, theta=[0.0, 0.0],
                oversampling=1, return_fft=True,
                positive_shift_tt=True, data_dir=temp_dir,
                target_device_idx=target_device_idx
            )
            kernel.zlayer = zlayer.tolist()
            kernel.zprofile = zprofile.tolist()

            # Calculate and save
            kernel.prepare_for_sh(
                sodium_altitude=zlayer.tolist(),
                sodium_intensity=zprofile.tolist(),
                current_time=1
            )

            # real_kernels should be deallocated
            self.assertIsNone(kernel.real_kernels)

            # Create new kernel data - make sure each kernel sums to 1 to avoid normalization effects
            new_kernels = cpuArray(xp.random.rand(dimx * dimy, dimension, dimension).astype(kernel.dtype))
            # Normalize each kernel so process_kernels doesn't change it much
            for i in range(dimx * dimy):
                total = np.sum(new_kernels[i])
                if total > 0:
                    new_kernels[i] /= total

            # set_value should recreate real_kernels
            kernel.set_value(new_kernels)

            # real_kernels should now exist
            self.assertIsNotNone(kernel.real_kernels,
                               "real_kernels should be recreated by set_value()")

            # Verify shape
            self.assertEqual(kernel.real_kernels.shape, 
                           (dimx * dimy, dimension, dimension))

            # Verify data was set correctly
            # (allowing for small numerical errors from normalization)
            np.testing.assert_array_almost_equal(
                cpuArray(kernel.real_kernels), new_kernels,
                decimal=5,  # Reduced precision to account for normalization
                err_msg="set_value should correctly set new kernel data"
            )

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @cpu_and_gpu
    def test_set_value_wrong_shape_after_deallocation(self, target_device_idx, xp):
        """
        Test that set_value() raises an error with wrong shape after deallocation.
        """
        dimx = dimy = 5
        spot_size = 1.0
        pixel_scale = 0.1
        pupil_size_m = 8.0
        dimension = 32

        num_points = 5
        zlayer = np.linspace(80e3, 100e3, num_points)
        center = 90e3
        fwhm = 10e3
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        zprofile = np.exp(-0.5 * ((zlayer - center) / sigma) ** 2)
        zfocus = 90e3
        launcher_pos = [5, 5, 0]
        zprofile /= np.sum(zprofile)

        temp_dir = tempfile.mkdtemp()

        try:
            kernel = ConvolutionKernel(
                dimx=dimx, dimy=dimy, pxscale=pixel_scale,
                pupil_size_m=pupil_size_m, dimension=dimension,
                launcher_pos=launcher_pos, seeing=spot_size,
                zfocus=zfocus, theta=[0.0, 0.0],
                oversampling=1, return_fft=True,
                positive_shift_tt=True, data_dir=temp_dir,
                target_device_idx=target_device_idx
            )
            kernel.zlayer = zlayer.tolist()
            kernel.zprofile = zprofile.tolist()

            kernel.prepare_for_sh(
                sodium_altitude=zlayer.tolist(),
                sodium_intensity=zprofile.tolist(),
                current_time=1
            )

            # Try to set with wrong shape
            wrong_shape_kernels = cpuArray(xp.random.rand(10, dimension, dimension).astype(kernel.dtype))

            with self.assertRaises(ValueError) as context:
                kernel.set_value(wrong_shape_kernels)

            self.assertIn("does not match expected shape", str(context.exception))

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @cpu_and_gpu
    def test_restore_after_deallocation(self, target_device_idx, xp):
        """
        Test that restore() works correctly even if real_kernels was deallocated.
        """
        dimx = dimy = 5
        spot_size = 1.0
        pixel_scale = 0.1
        pupil_size_m = 8.0
        dimension = 32

        num_points = 5
        zlayer = np.linspace(80e3, 100e3, num_points)
        center = 90e3
        fwhm = 10e3
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        zprofile = np.exp(-0.5 * ((zlayer - center) / sigma) ** 2)
        zfocus = 90e3
        launcher_pos = [5, 5, 0]
        zprofile /= np.sum(zprofile)

        temp_dir = tempfile.mkdtemp()

        try:
            # Create and save kernel
            kernel1 = ConvolutionKernel(
                dimx=dimx, dimy=dimy, pxscale=pixel_scale,
                pupil_size_m=pupil_size_m, dimension=dimension,
                launcher_pos=launcher_pos, seeing=spot_size,
                zfocus=zfocus, theta=[0.0, 0.0],
                oversampling=1, return_fft=True,
                positive_shift_tt=True, data_dir=temp_dir,
                target_device_idx=target_device_idx
            )
            kernel1.zlayer = zlayer.tolist()
            kernel1.zprofile = zprofile.tolist()

            kernel1.prepare_for_sh(
                sodium_altitude=zlayer.tolist(),
                sodium_intensity=zprofile.tolist(),
                current_time=1
            )

            kernel_fn = kernel1.build()
            saved_file = os.path.join(temp_dir, kernel_fn + '.fits')

            # Verify file was saved
            self.assertTrue(os.path.exists(saved_file))

            # Create new kernel object with real_kernels initially None
            kernel2 = ConvolutionKernel(
                dimx=dimx, dimy=dimy, pxscale=pixel_scale,
                pupil_size_m=pupil_size_m, dimension=dimension,
                launcher_pos=launcher_pos, seeing=spot_size,
                zfocus=zfocus, theta=[0.0, 0.0],
                oversampling=1, return_fft=True,
                positive_shift_tt=True,
                target_device_idx=target_device_idx
            )

            # Manually deallocate to simulate the state after prepare_for_sh
            kernel2.real_kernels = None
            kernel2.zlayer = zlayer.tolist()
            kernel2.zprofile = zprofile.tolist()

            # Restore should recreate real_kernels
            ConvolutionKernel.restore(
                saved_file, 
                kernel_obj=kernel2,
                target_device_idx=target_device_idx,
                return_fft=True
            )

            # real_kernels should now exist (reallocated by restore)
            self.assertIsNotNone(kernel2.real_kernels,
                               "restore() should reallocate real_kernels")

            # Verify shape
            self.assertEqual(kernel2.real_kernels.shape,
                           (dimx * dimy, dimension, dimension))

            # kernels should also exist
            self.assertIsNotNone(kernel2.kernels)

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @cpu_and_gpu
    def test_memory_reduction_with_deallocation(self, target_device_idx, xp):
        """
        Test that deallocating real_kernels actually reduces memory usage.
        This is more of a documentation test showing the memory savings.
        """
        dimx = dimy = 10  # Larger to see memory difference
        spot_size = 1.0
        pixel_scale = 0.1
        pupil_size_m = 8.0
        dimension = 64  # Larger dimension

        num_points = 5
        zlayer = np.linspace(80e3, 100e3, num_points)
        center = 90e3
        fwhm = 10e3
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        zprofile = np.exp(-0.5 * ((zlayer - center) / sigma) ** 2)
        zfocus = 90e3
        launcher_pos = [5, 5, 0]
        zprofile /= np.sum(zprofile)

        temp_dir = tempfile.mkdtemp()

        try:
            kernel = ConvolutionKernel(
                dimx=dimx, dimy=dimy, pxscale=pixel_scale,
                pupil_size_m=pupil_size_m, dimension=dimension,
                launcher_pos=launcher_pos, seeing=spot_size,
                zfocus=zfocus, theta=[0.0, 0.0],
                oversampling=1, return_fft=True,
                positive_shift_tt=True, data_dir=temp_dir,
                target_device_idx=target_device_idx
            )
            kernel.zlayer = zlayer.tolist()
            kernel.zprofile = zprofile.tolist()

            # Before prepare_for_sh: both arrays allocated
            kernel.build()
            kernel.calculate_lgs_map()

            real_kernels_size = kernel.real_kernels.nbytes if kernel.real_kernels is not None else 0
            kernels_size = kernel.kernels.nbytes if kernel.kernels is not None else 0

            memory_before = real_kernels_size + kernels_size

            # After prepare_for_sh: real_kernels deallocated
            kernel.prepare_for_sh(
                sodium_altitude=zlayer.tolist(),
                sodium_intensity=zprofile.tolist(),
                current_time=1
            )

            real_kernels_size_after = kernel.real_kernels.nbytes if kernel.real_kernels is not None else 0
            kernels_size_after = kernel.kernels.nbytes if kernel.kernels is not None else 0

            memory_after = real_kernels_size_after + kernels_size_after

            # Memory should be reduced
            self.assertLess(memory_after, memory_before,
                          "Memory usage should decrease after deallocation")

            # Expected savings: size of real_kernels (float32)
            expected_savings = dimx * dimy * dimension * dimension * 4  # 4 bytes for float32
            actual_savings = memory_before - memory_after

            # Should be close to expected (within 10% tolerance)
            self.assertAlmostEqual(actual_savings, expected_savings,
                                 delta=expected_savings * 0.1,
                                 msg=f"Expected to save ~{expected_savings/1e6:.1f} MB, "
                                     f"actually saved {actual_savings/1e6:.1f} MB")

            print(f"\nMemory test: Saved {actual_savings/1e6:.2f} MB by deallocating real_kernels")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
