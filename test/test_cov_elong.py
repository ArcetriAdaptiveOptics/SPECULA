import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.lib.calc_noise_cov_elong import calc_noise_cov_elong
from specula.lib.make_mask import make_mask
from astropy.io import fits
from test.specula_testlib import cpu_and_gpu

class TestCovTRunc(unittest.TestCase):

    @cpu_and_gpu
    def test_cov_trunc(self, target_device_idx, xp):
        # Parameters
        diameter_in_m = 8
        zenith_angle_in_deg = 30.
        na_thickness_in_m = 10e3
        launcher_coord_in_m = [3.0, 3.0, 0.]
        n_sub_aps = 4
        mask_sub_aps = make_mask(n_sub_aps, xp=xp)
        sub_aps_index = xp.where(mask_sub_aps)
        # Convert to 1D indices for compatibility
        sub_aps_index_1D = xp.ravel_multi_index(sub_aps_index, mask_sub_aps.shape)
        sub_aps_fov = 5.0
        sh_spot_fwhm = 1.0
        sigma_noise2 = 1.0
        t_g_parameter = 0.0
        h_in_m = 90e3

        # Compute covariance matrix with Python function
        cov = calc_noise_cov_elong(
                diameter_in_m, zenith_angle_in_deg, na_thickness_in_m, launcher_coord_in_m,
                sub_aps_index_1D, n_sub_aps, sub_aps_fov, sh_spot_fwhm, sigma_noise2,
                t_g_parameter, h_in_m=h_in_m, only_diag=False, verbose=False, display=False
        )

        # Compute covariance matrix with Python function (only diagonal)
        cov_only_diag = calc_noise_cov_elong(
                diameter_in_m, zenith_angle_in_deg, na_thickness_in_m, launcher_coord_in_m,
                sub_aps_index_1D, n_sub_aps, sub_aps_fov, sh_spot_fwhm, sigma_noise2,
                t_g_parameter, h_in_m=h_in_m, only_diag=True, verbose=False, display=False
        )

        # Load reference FITS (IDL result)
        testdir = os.path.dirname(__file__)
        ref_path = os.path.join(testdir, 'data', 'cov_sh_ref.fits')
        cov_idl_all = fits.getdata(ref_path)
        cov_idl = cov_idl_all[0]
        cov_only_diag_idl = cov_idl_all[1]

        verbose = False
        if verbose:
            print("diagonal of covariance matrix (Python):")
            print(cov.diagonal())
            print("diagonal of covariance matrix (IDL):")
            print(cov_idl.diagonal())
            print(cov_only_diag_idl.diagonal())
            print("difference of diagonal of covariance matrix (Python - IDL):")
            print(cov.diagonal() - cov_idl.diagonal())

        display = False
        if display:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(cov)
            plt.colorbar()
            plt.title("Python Covariance Matrix")
            plt.figure(figsize=(10, 8))
            plt.imshow(cov_idl)
            plt.colorbar()
            plt.title("IDL Covariance Matrix")
            plt.figure(figsize=(10, 8))
            plt.imshow(cov_idl-cov)
            plt.colorbar()
            plt.title("Difference Covariance Matrix (IDL - Python)")
            plt.show()

        # Compare shapes
        self.assertEqual(cov.shape, cov_idl.shape)

        # Compare diagonal values
        np.testing.assert_allclose(
                cpuArray(cov.diagonal()), cpuArray(cov_idl.diagonal()),
                rtol=1e-3, atol=1e-6
        )

        # Compare off-diagonal values
        np.testing.assert_allclose(
                cpuArray(cov[~np.eye(cov.shape[0], dtype=bool)]),
                cpuArray(cov_idl[~np.eye(cov_idl.shape[0], dtype=bool)]),
                rtol=1e-3, atol=1e-6
        )

        # Compare values (allowing for small numerical differences)
        np.testing.assert_allclose(
                cpuArray(cov), cpuArray(cov_idl),
                rtol=1e-3, atol=1e-6
        )

        # Compare values of only diagonal
        np.testing.assert_allclose(
                cpuArray(cov_only_diag), cpuArray(cov_only_diag_idl),
                rtol=1e-3, atol=1e-6
        )