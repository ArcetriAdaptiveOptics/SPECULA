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

    @cpu_and_gpu
    def test_cov_with_eta_not_one(self, target_device_idx, xp):
        """Test covariance computation with eta_is_not_one=True"""
        diameter_in_m = 8
        zenith_angle_in_deg = 30.
        na_thickness_in_m = 10e3
        launcher_coord_in_m = [3.0, 3.0, 0.]
        n_sub_aps = 4
        mask_sub_aps = make_mask(n_sub_aps, xp=xp)
        sub_aps_index = xp.where(mask_sub_aps)
        sub_aps_index_1D = xp.ravel_multi_index(sub_aps_index, mask_sub_aps.shape)
        sub_aps_fov = 5.0
        sh_spot_fwhm = 1.0
        sigma_noise2 = 1.0
        t_g_parameter = 0.0
        h_in_m = 90e3

        # Should not crash with eta_is_not_one=True
        cov = calc_noise_cov_elong(
            diameter_in_m, zenith_angle_in_deg, na_thickness_in_m, launcher_coord_in_m,
            sub_aps_index_1D, n_sub_aps, sub_aps_fov, sh_spot_fwhm, sigma_noise2,
            t_g_parameter, h_in_m=h_in_m, eta_is_not_one=True,
            only_diag=False, verbose=False, display=False
        )

        # Check that output is valid
        self.assertEqual(cov.shape[0], cov.shape[1])
        self.assertEqual(cov.shape[0], 2 * len(cpuArray(sub_aps_index_1D)))
        # Check that diagonal is positive
        diag_cpu = cpuArray(np.diag(cov))
        self.assertTrue(np.all(diag_cpu > 0))

        # Check that not all cross-terms are zero (indicating that elongation is being captured by the covariance)
        # Being the launcher at [3.0, 3.0, 0] (diagonal), we expect strong coupling between X and Y modes,
        # so the cross-terms should not be all zero.
        n_valid = len(cpuArray(sub_aps_index_1D))
        cross_terms = cpuArray(np.diag(cov, k=n_valid)) # Extract the cross-terms (cov[i, i + n_valid])
                                                        # that represent the coupling between X and Y modes
                                                        # of the same sub-aperture

        # At least one cross-term should be significantly different from zero
        self.assertTrue(np.any(np.abs(cross_terms) > 1e-3))

    @cpu_and_gpu
    def test_cov_with_theta(self, target_device_idx, xp):
        """Test covariance computation with theta parameter"""
        diameter_in_m = 8
        zenith_angle_in_deg = 30.
        na_thickness_in_m = 10e3
        launcher_coord_in_m = [3.0, 3.0, 0.]
        n_sub_aps = 4
        mask_sub_aps = make_mask(n_sub_aps, xp=xp)
        sub_aps_index = xp.where(mask_sub_aps)
        sub_aps_index_1D = xp.ravel_multi_index(sub_aps_index, mask_sub_aps.shape)
        sub_aps_fov = 5.0
        sh_spot_fwhm = 1.0
        sigma_noise2 = 1.0
        t_g_parameter = 0.0
        h_in_m = 90e3

        # Test with theta as list
        theta_list = [0.5, 0.5]
        cov = calc_noise_cov_elong(
            diameter_in_m, zenith_angle_in_deg, na_thickness_in_m, launcher_coord_in_m,
            sub_aps_index_1D, n_sub_aps, sub_aps_fov, sh_spot_fwhm, sigma_noise2,
            t_g_parameter, h_in_m=h_in_m, theta=theta_list, eta_is_not_one=True,
            only_diag=False, verbose=False, display=False
        )

        self.assertEqual(cov.shape[0], 2 * len(cpuArray(sub_aps_index_1D)))
        self.assertTrue(np.all(cpuArray(np.diag(cov)) > 0))

    @cpu_and_gpu
    def test_cov_with_truncation(self, target_device_idx, xp):
        """Test covariance computation with t_g_parameter > 0"""
        diameter_in_m = 8
        zenith_angle_in_deg = 30.
        na_thickness_in_m = 10e3
        launcher_coord_in_m = [3.0, 3.0, 0.]
        n_sub_aps = 4
        mask_sub_aps = make_mask(n_sub_aps, xp=xp)
        sub_aps_index = xp.where(mask_sub_aps)
        sub_aps_index_1D = xp.ravel_multi_index(sub_aps_index, mask_sub_aps.shape)
        sub_aps_fov = 5.0
        sh_spot_fwhm = 1.0
        sigma_noise2 = 1.0
        t_g_parameter = 0.3  # 30% truncation
        h_in_m = 90e3

        # Test with full matrix
        cov_full = calc_noise_cov_elong(
            diameter_in_m, zenith_angle_in_deg, na_thickness_in_m, launcher_coord_in_m,
            sub_aps_index_1D, n_sub_aps, sub_aps_fov, sh_spot_fwhm, sigma_noise2,
            t_g_parameter, h_in_m=h_in_m, only_diag=False, verbose=False
        )

        # Test with diagonal only
        cov_diag = calc_noise_cov_elong(
            diameter_in_m, zenith_angle_in_deg, na_thickness_in_m, launcher_coord_in_m,
            sub_aps_index_1D, n_sub_aps, sub_aps_fov, sh_spot_fwhm, sigma_noise2,
            t_g_parameter, h_in_m=h_in_m, only_diag=True, verbose=False
        )

        self.assertEqual(cov_full.shape[0], 2 * len(cpuArray(sub_aps_index_1D)))
        self.assertEqual(cov_diag.shape[0], 2 * len(cpuArray(sub_aps_index_1D)))
        self.assertTrue(np.all(cpuArray(np.diag(cov_full)) > 0))
        self.assertTrue(np.all(cpuArray(np.diag(cov_diag)) > 0))

        # Check that off-diagonal elements of the full matrix are small (due to truncation)
        cov_diag_cpu = cpuArray(cov_diag)
        off_diagonal_elements = cov_diag_cpu[~np.eye(cov_diag_cpu.shape[0], dtype=bool)]
        np.testing.assert_allclose(off_diagonal_elements, 0.0, atol=1e-10)

        # Check that the trace of the full matrix is not zero (indicating that it has valid values)
        self.assertTrue(np.sum(np.diag(cov_full)) > 0)

    @cpu_and_gpu
    def test_coord_sub_aps_consistency(self, target_device_idx, xp):
        """Test that coord_sub_aps is properly defined in all code paths"""
        diameter_in_m = 8
        zenith_angle_in_deg = 30.
        na_thickness_in_m = 10e3
        launcher_coord_in_m = [3.0, 3.0, 0.]
        n_sub_aps = 4
        mask_sub_aps = make_mask(n_sub_aps, xp=xp)
        sub_aps_index = xp.where(mask_sub_aps)
        sub_aps_index_1D = xp.ravel_multi_index(sub_aps_index, mask_sub_aps.shape)
        sub_aps_fov = 5.0
        sh_spot_fwhm = 1.0
        sigma_noise2 = 1.0
        t_g_parameter = 0.0
        h_in_m = 90e3

        # Test both code paths (with and without eta_is_not_one)
        for eta_flag in [False, True]:
            for diag_flag in [False, True]:
                cov = calc_noise_cov_elong(
                    diameter_in_m, zenith_angle_in_deg, na_thickness_in_m,
                    launcher_coord_in_m, sub_aps_index_1D, n_sub_aps,
                    sub_aps_fov, sh_spot_fwhm, sigma_noise2, t_g_parameter,
                    h_in_m=h_in_m, eta_is_not_one=eta_flag,
                    only_diag=diag_flag, verbose=False
                )
                # Should not crash and return valid matrix
                self.assertIsNotNone(cov)
                self.assertTrue(np.all(np.isfinite(cpuArray(cov))))

    @cpu_and_gpu
    def test_different_sub_aps_sizes(self, target_device_idx, xp):
        """Test with different number of subapertures"""
        diameter_in_m = 8
        zenith_angle_in_deg = 30.
        na_thickness_in_m = 10e3
        launcher_coord_in_m = [3.0, 3.0, 0.]
        sub_aps_fov = 5.0
        sh_spot_fwhm = 1.0
        sigma_noise2 = 1.0
        t_g_parameter = 0.0
        h_in_m = 90e3

        for n_sub_aps in [4, 8, 16]:
            mask_sub_aps = make_mask(n_sub_aps, xp=xp)
            sub_aps_index = xp.where(mask_sub_aps)
            sub_aps_index_1D = xp.ravel_multi_index(sub_aps_index, mask_sub_aps.shape)

            cov = calc_noise_cov_elong(
                diameter_in_m, zenith_angle_in_deg, na_thickness_in_m,
                launcher_coord_in_m, sub_aps_index_1D, n_sub_aps,
                sub_aps_fov, sh_spot_fwhm, sigma_noise2, t_g_parameter,
                h_in_m=h_in_m, only_diag=False, verbose=False
            )

            expected_size = 2 * len(cpuArray(sub_aps_index_1D))
            self.assertEqual(cov.shape[0], expected_size)
            self.assertTrue(np.all(cpuArray(np.diag(cov)) > 0))

    @cpu_and_gpu
    def test_xy_symmetry_physics(self, target_device_idx, xp):
        """Check that a launcher on the X-axis does not produce elongation in Y on the X-axis"""
        diameter_in_m = 8.0
        zenith_angle_in_deg = 30.
        na_thickness_in_m = 10e3
        launcher_coord_in_m = [5.0, 0.0, 0.0] # Launcher purely on X-axis
        n_sub_aps = 4
        mask_sub_aps = make_mask(n_sub_aps, xp=xp)
        sub_aps_index = xp.where(mask_sub_aps)
        sub_aps_index_1D = xp.ravel_multi_index(sub_aps_index, mask_sub_aps.shape)

        cov_full = calc_noise_cov_elong(
            diameter_in_m, zenith_angle_in_deg, na_thickness_in_m, launcher_coord_in_m,
            sub_aps_index_1D, n_sub_aps, sub_aps_fov=5.0, sh_spot_fwhm=1.0, 
            sigma_noise2=1.0, t_g_parameter=0.0, h_in_m=90e3, only_diag=False, verbose=False
        )

        n_valid = len(cpuArray(sub_aps_index_1D))

        # Y coordinates of sub-apertures. unravel_index returns (Y, X) coordinates.
        y_coords, x_coords = xp.unravel_index(sub_aps_index_1D, (n_sub_aps, n_sub_aps))

        # Look for sub-apertures that are on the horizontal center line (Y-axis) of the grid.
        # For n_sub_aps=4, the center Y would be between indices 1 and 2 (0-based), so we can check for those.
        center_y_idx = n_sub_aps // 2
        on_axis_mask = y_coords == center_y_idx

        # For these sub-apertures, the cross-term (cov_full[i, i + n_valid]) MUST be ~0
        for i in range(n_valid):
            if on_axis_mask[i]:
                cross_term = cov_full[i, i + n_valid]
                # Check that the cross-term is numerically zero
                np.testing.assert_allclose(cpuArray(cross_term), 0.0, atol=1e-7)
