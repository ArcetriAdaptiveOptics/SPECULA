import unittest
import os
import numpy as np
from astropy.io import fits

import specula
specula.init(0)  # Default target device
from specula import cp, np
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
        spotsize = 1.0  # arcsec
        pixel_scale = 0.1  # arcsec
        pupil_size_m = 1.0  # m
        dimension = 16  # Size of kernel in pixels

        kernel = GaussianConvolutionKernel(
            convolGaussSpotSize=spotsize,
            dimx=dimx,
            dimy=dimy,
            target_device_idx=target_device_idx
        )

        # Set required parameters
        kernel.pxscale = pixel_scale
        kernel.pupil_size_m = pupil_size_m
        kernel.dimension = dimension
        kernel.oversampling = 1
        kernel.return_fft = True
        kernel.positive_shift_tt = True

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
        spotsize = 1.0  # arcsec
        pixel_scale = 0.1  # arcsec
        pupil_size_m = 8.0  # m
        dimension = 64  # Size of kernel in pixels

        # Create sodium layer profile
        num_points = 20
        z_min = 80e3  # m
        z_max = 100e3  # m
        zlayer = np.linspace(z_min, z_max, num_points)

        # Create Gaussian intensity profile with FWHM of 10e3 m
        center = 90e3  # m
        fwhm = 10e3  # m
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        zprofile = np.exp(-0.5 * ((zlayer - center) / sigma) ** 2)
        zfocus = 90e3  # m
        launcher_pos = [5, 5, 0]  # m

        # Normalize the profile
        zprofile /= np.sum(zprofile)

        kernel = ConvolutionKernel(
            dimx=dimx,
            dimy=dimy,
            target_device_idx=target_device_idx
        )
        
        # Set required parameters
        kernel.pxscale = pixel_scale
        kernel.pupil_size_m = pupil_size_m
        kernel.dimension = dimension
        kernel.seeing = spotsize
        kernel.zlayer = zlayer.tolist()
        kernel.zprofile = zprofile.tolist()
        kernel.zfocus = zfocus
        kernel.launcher_pos = launcher_pos

        kernel.return_fft = False

        # Test with return_fft = False
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
        kernel.return_fft = True
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
        spotsize = 1.0  # arcsec
        pixel_scale = 0.1  # arcsec
        pupil_size_m = 8.0  # m
        dimension = 64  # Size of kernel in pixels
        
        # Create sodium layer profile
        num_points = 20
        z_min = 80e3  # m
        z_max = 100e3  # m
        zlayer = np.linspace(z_min, z_max, num_points)
        
        # Create Gaussian intensity profile with FWHM of 10e3 m
        center = 90e3  # m
        fwhm = 10e3  # m
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        zprofile = np.exp(-0.5 * ((zlayer - center) / sigma) ** 2)
        zfocus = 90e3  # m
        launcher_pos = [5, 5, 0]  # m
        
        # Normalize the profile
        zprofile /= np.sum(zprofile)
        layer_offsets = zlayer - zfocus
        
        map = lgs_map_sh(
            nsh=dimx, diam=pupil_size_m, rl=launcher_pos, zb=zfocus,
            dz=layer_offsets, profz=zprofile, fwhmb=spotsize, ps=pixel_scale,
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
                if display:
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
                print("LGS map kernel2d matches reference values")