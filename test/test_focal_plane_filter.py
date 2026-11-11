import specula
specula.init(0)  # Default target device

import unittest

from scipy.special import j0, j1

from specula import np, cpuArray, RAD2ASEC
from specula.lib.calc_psf import calc_psf
from specula.lib.make_mask import make_mask
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.focal_plane_filter import FocalPlaneFilter

from test.specula_testlib import cpu_and_gpu

class TestFocalPlaneFilter(unittest.TestCase):

    def setUp(self):
        # Basic simulation parameters
        self.pixel_pupil = 120
        self.pixel_pitch = 0.05
        self.wavelength_nm = 500
        self.fov = 2.0

        self.simul_params = SimulParams(
            pixel_pupil=self.pixel_pupil,
            pixel_pitch=self.pixel_pitch
        )

        # make a round mask for the pupil
        self.mask = make_mask(self.pixel_pupil, obsratio=0.0, xp=np)

    @cpu_and_gpu
    def test_output_shape(self, target_device_idx, xp):
        """Test that output ElectricField has expected shape"""
        fpf = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        fpf.inputs['in_ef'].set(ef)
        fpf.setup()
        fpf.check_ready(1)
        fpf.prepare_trigger(1)
        fpf.trigger_code()
        fpf.post_trigger()
        out_ef = fpf.outputs['out_ef']
        self.assertEqual(out_ef.A.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(out_ef.phaseInNm.shape, (self.pixel_pupil, self.pixel_pupil))

    @cpu_and_gpu
    def test_psf_with_and_without_obstruction(self, target_device_idx, xp):
        """Test PSF with and without a central obstruction using calc_psf"""
        # No filter (no obstruction)
        fpf_nofilter = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            fp_obs=0.0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        fpf_nofilter.inputs['in_ef'].set(ef)
        fpf_nofilter.setup()
        fpf_nofilter.check_ready(1)
        fpf_nofilter.prepare_trigger(1)
        fpf_nofilter.trigger_code()
        fpf_nofilter.post_trigger()
        ef_nofilter = fpf_nofilter.outputs['out_ef']

        # With filter: central obstruction of 2 lambda/D
        fp_obs = 2.0  # in lambda/D

        fpf_obs = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            fp_obs=fp_obs,
            target_device_idx=target_device_idx
        )
        fpf_obs.inputs['in_ef'].set(ef)
        fpf_obs.setup()
        fpf_obs.check_ready(1)
        fpf_obs.prepare_trigger(1)
        fpf_obs.trigger_code()
        fpf_obs.post_trigger()
        ef_obs = fpf_obs.outputs['out_ef']

        # Compute PSF for both cases using calc_psf
        nm2rad = 2*xp.pi/self.wavelength_nm
        psf = calc_psf(ef.phaseInNm*nm2rad, ef.A, xp=xp, complex_dtype=xp.complex64, normalize=True)
        psf_nofilter = calc_psf(ef_nofilter.phaseInNm*nm2rad, ef_nofilter.A, xp=xp, complex_dtype=xp.complex64, normalize=True)
        psf_obs = calc_psf(ef_obs.phaseInNm*nm2rad, ef_obs.A, xp=xp, complex_dtype=xp.complex64, normalize=True)

        max_psf = float(psf.max())
        max_psf_nofilter = float(psf_nofilter.max())
        max_psf_obs = float(psf_obs.max())

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cpuArray(ef_nofilter.A), cmap='gray')
            plt.colorbar()
            plt.title('Amplitude without obstruction')
            plt.subplot(1,2,2)
            plt.imshow(cpuArray(ef_obs.A), cmap='gray')
            plt.colorbar()
            plt.title('Amplitude with 2 lambda/D obstruction')
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cpuArray(ef_nofilter.phaseInNm), cmap='twilight')
            plt.colorbar()
            plt.title('Phase without obstruction')
            plt.subplot(1,2,2)
            plt.imshow(cpuArray(ef_obs.phaseInNm), cmap='twilight')
            plt.colorbar()
            plt.title('Phase with 2 lambda/D obstruction')
            plt.show()
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(cpuArray(psf), cmap='viridis', norm=colors.LogNorm(vmin=1e-6*max_psf, vmax=max_psf))
            plt.colorbar()
            plt.title('PSF input wavefront')
            plt.subplot(1,3,2)
            plt.imshow(cpuArray(psf_nofilter), cmap='viridis', norm=colors.LogNorm(vmin=1e-6*max_psf_nofilter, vmax=max_psf_nofilter))
            plt.colorbar()
            plt.title('PSF without obstruction')
            plt.subplot(1,3,3)
            plt.imshow(cpuArray(psf_obs), cmap='viridis', norm=colors.LogNorm(vmin=1e-6*max_psf_obs, vmax=max_psf_obs))
            plt.colorbar()
            plt.title('PSF with 2 lambda/D obstruction')
            plt.show()

        # Check shapes
        self.assertEqual(psf_nofilter.shape, psf_obs.shape)

        # Check that the mask has an effect (PSFs must differ)
        diff = np.abs(psf_nofilter - psf_obs).sum()
        self.assertGreater(cpuArray(diff), 0.0, "Obstruction mask does not affect the PSF!")

    @cpu_and_gpu
    def test_phase_preservation(self, target_device_idx, xp):
        """Test that a flat input phase results in a flat output phase (no mask)"""
        fpf = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            fp_obs=0.0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        fpf.inputs['in_ef'].set(ef)
        fpf.setup()
        fpf.check_ready(1)
        fpf.prepare_trigger(1)
        fpf.trigger_code()
        fpf.post_trigger()
        out_ef = fpf.outputs['out_ef']
        # Output phase should be (almost) constant for a flat input
        mask = ef.A > 0
        valid_phases = out_ef.phaseInNm[mask]
        min_phase = float(xp.min(valid_phases))
        max_phase = float(xp.max(valid_phases))

        # max and min phase should be close to zero (within 5 nm)
        self.assertLess(np.abs(max_phase), 5)
        self.assertLess(np.abs(min_phase), 5)

    @cpu_and_gpu
    def test_amplitude_preservation(self, target_device_idx, xp):
        """Test that a flat input amplitude results in a nonzero output amplitude (no mask)"""
        fpf = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            fp_obs=0.0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = 1
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        fpf.inputs['in_ef'].set(ef)
        fpf.setup()
        fpf.check_ready(1)
        fpf.prepare_trigger(1)
        fpf.trigger_code()
        fpf.post_trigger()
        out_ef = fpf.outputs['out_ef']
        # Output amplitude should not be all zeros and should be approximately the same as the input one
        self.assertGreater(float(out_ef.A.sum()), 0.0)
        self.assertLess(float(out_ef.A.max()), 2.0*float(ef.A.max()))

    @cpu_and_gpu
    def test_s0_scaling_with_obstruction(self, target_device_idx, xp):
        """Test that S0 is scaled correctly when using obstruction"""
        # Test with obstruction - S0 should decrease
        fpf_obs = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            fp_obs=2.0,  # 2 lambda/D obstruction
            target_device_idx=target_device_idx
        )

        # Test without obstruction - S0 should remain similar
        fpf_no_obs = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            fp_obs=0.0,
            target_device_idx=target_device_idx
        )

        # Create input electric field
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=100.0, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.S0 = 100.0
        ef.generation_time = 1

        # Test with obstruction
        fpf_obs.inputs['in_ef'].set(ef)
        fpf_obs.setup()
        fpf_obs.check_ready(1)
        fpf_obs.prepare_trigger(1)
        fpf_obs.trigger_code()
        fpf_obs.post_trigger()
        s0_with_obs = fpf_obs.outputs['out_ef'].S0

        # Test without obstruction
        fpf_no_obs.inputs['in_ef'].set(ef)
        fpf_no_obs.setup()
        fpf_no_obs.check_ready(1)
        fpf_no_obs.prepare_trigger(1)
        fpf_no_obs.trigger_code()
        fpf_no_obs.post_trigger()
        s0_no_obs = fpf_no_obs.outputs['out_ef'].S0

        # S0 with obstruction should be less than without obstruction
        self.assertLess(s0_with_obs, s0_no_obs, "S0 should decrease with obstruction!")

        # Both should be less than or equal to original S0
        self.assertLessEqual(s0_with_obs, 100.0)
        self.assertLessEqual(s0_no_obs, 100.0)

    @cpu_and_gpu
    def test_interpolation_activation(self, target_device_idx, xp):
        """Test that interpolation is activated when FoV is large"""

        # Small FoV - should not trigger interpolation
        fpf_small_fov = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=1.0,  # Small FoV
            target_device_idx=target_device_idx
        )

        # Large FoV - should trigger interpolation
        fpf_large_fov = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=5.0,  # Large FoV
            fov_errinf=0.1,
            fov_errsup=10.0,
            target_device_idx=target_device_idx
        )

        # Create input electric field
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        # Setup both filters
        fpf_small_fov.inputs['in_ef'].set(ef)
        fpf_small_fov.setup()

        fpf_large_fov.inputs['in_ef'].set(ef)
        fpf_large_fov.setup()

        # Check interpolation flags
        self.assertFalse(fpf_small_fov.ef_interpolator.do_interpolation, "Small FoV should not require interpolation")
        self.assertTrue(fpf_large_fov.ef_interpolator.do_interpolation, "Large FoV should require interpolation")

        # Check fov_res values
        self.assertEqual(fpf_small_fov.fov_res, 1.0, "Small FoV should have fov_res = 1")
        self.assertGreater(fpf_large_fov.fov_res, 1.0, "Large FoV should have fov_res > 1")

    @cpu_and_gpu
    def test_fft_sampling_with_interpolation(self, target_device_idx, xp):
        """Test that FFT sampling changes correctly with interpolation"""

        # Large FoV that requires interpolation
        fpf = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=5.0,  # Large FoV
            fov_errinf=0.1,
            fov_errsup=10.0,
            target_device_idx=target_device_idx
        )

        # Create input electric field
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        fpf.inputs['in_ef'].set(ef)
        fpf.setup()

        # Check that fft_sampling is larger than pixel_pupil when interpolation is needed
        if fpf.ef_interpolator.do_interpolation:
            self.assertGreater(fpf.fft_sampling, self.pixel_pupil, 
                             "FFT sampling should be larger than pixel_pupil when interpolation is used")
            
            # Check that the interpolated field has the correct size
            self.assertEqual(fpf.ef_interpolator.interpolated_ef().A.shape[0], fpf.fft_sampling)
            self.assertEqual(fpf.ef_interpolator.interpolated_ef().A.shape[1], fpf.fft_sampling)
            
    @cpu_and_gpu
    def test_transmission_calculation(self, target_device_idx, xp):
        """Test that transmission is calculated correctly"""

        # Test with different FoV sizes
        fovs = [1.0, 0.5, 0.2, 0.1]
        transmissions = []

        for fov in fovs:
            fpf = FocalPlaneFilter(
                simul_params=self.simul_params,
                wavelengthInNm=self.wavelength_nm,
                fov=fov,
                fov_errinf=0.001,
                fov_errsup=1000.0,
                target_device_idx=target_device_idx
            )

            # Create input electric field
            ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=100.0, target_device_idx=target_device_idx)
            ef.A[:] = xp.array(self.mask)
            ef.phaseInNm[:] = 0.0
            ef.S0 = 100.0
            ef.generation_time = 1

            fpf.inputs['in_ef'].set(ef)
            fpf.setup()
            fpf.check_ready(1)
            fpf.prepare_trigger(1)
            fpf.trigger_code()
            fpf.post_trigger()

            transmission = fpf.outputs['out_ef'].S0 / ef.S0
            transmissions.append(float(transmission))

        # Transmission should decrease as obstruction increases
        for i in range(1, len(transmissions)):
            self.assertLess(transmissions[i], transmissions[i-1],
                          f"Transmission should decrease with larger FoV: {fovs[i]} vs {fovs[i-1]}")

        # Transmission should be between 0 and 1
        for t in transmissions:
            self.assertGreater(t, 0.0, "Transmission should be positive")
            self.assertLessEqual(t, 1.0, "Transmission should not exceed 1.0")

class TestFocalPlaneFilterObstruction(unittest.TestCase):
    """Units and physical meaning of fp_obs (diameter of the obstruction in lambda/D)."""

    def setUp(self):
        self.pixel_pupil = 120
        self.pixel_pitch = 0.05
        self.wavelength_nm = 500
        self.simul_params = SimulParams(
            pixel_pupil=self.pixel_pupil,
            pixel_pitch=self.pixel_pitch
        )
        self.mask = make_mask(self.pixel_pupil, obsratio=0.0, xp=np)
        # lambda/D in arcsec, with D = pixel_pupil * pixel_pitch
        self.ld_arcsec = (self.wavelength_nm * 1e-9 / (self.pixel_pupil * self.pixel_pitch)
                          * RAD2ASEC)

    def _make_filter(self, fov, fp_obs, target_device_idx):
        return FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=fov,
            fp_obs=fp_obs,
            target_device_idx=target_device_idx
        )

    @staticmethod
    def _equivalent_diameter_ld(n_pixels, fft_res):
        """Diameter in lambda/D of the disk with the same area as n_pixels."""
        return 2 * np.sqrt(n_pixels / np.pi) / fft_res

    def _run_filter(self, fpf, ef):
        fpf.inputs['in_ef'].set(ef)
        fpf.setup()
        fpf.check_ready(1)
        fpf.prepare_trigger(1)
        fpf.trigger_code()
        fpf.post_trigger()
        return fpf.outputs['out_ef']

    @cpu_and_gpu
    def test_obstruction_diameter_in_lambda_over_d(self, target_device_idx, xp):
        """The zero pixels inside the field stop form a disk of diameter fp_obs [lambda/D].

        Expected: the area-equivalent diameter of the obstruction equals fp_obs. The
        focal plane has fft_res = 3 pixels per lambda/D, also when the pupil is
        interpolated (fov_res > 1), so a disk edge is quantised to ~1/3 lambda/D
        and the tolerance is 0.25 lambda/D (measured deviations are below 0.15).
        The fov values cover fov_res = 1, 2 and 3 (different fp_masking too).
        """
        expected_fov_res = {2.0: 1, 3.0: 2, 6.0: 3}
        for fov, fov_res in expected_fov_res.items():
            no_obs = cpuArray(self._make_filter(fov, 0.0, target_device_idx).fp_mask)
            for fp_obs in (2.0, 4.0, 10.0):
                with self.subTest(fov=fov, fp_obs=fp_obs):
                    fpf = self._make_filter(fov, fp_obs, target_device_idx)
                    self.assertEqual(fpf.fov_res, fov_res)
                    mask = cpuArray(fpf.fp_mask)
                    # Zero pixels inside the field stop (open without obstruction)
                    n_obstructed = np.count_nonzero((no_obs == 1) & (mask == 0))
                    diameter = self._equivalent_diameter_ld(n_obstructed, fpf.fft_res)
                    self.assertAlmostEqual(diameter, fp_obs, delta=0.25)

    @cpu_and_gpu
    def test_field_stop_diameter_in_arcsec(self, target_device_idx, xp):
        """Open pixels plus obstruction have an area-equivalent diameter equal to fov [arcsec].

        Expected: within 1%, independently of fov_res and of the obstruction.
        """
        for fov in (1.0, 2.0, 3.0, 6.0):
            for fp_obs in (0.0, 4.0):
                with self.subTest(fov=fov, fp_obs=fp_obs):
                    fpf = self._make_filter(fov, fp_obs, target_device_idx)
                    mask = cpuArray(fpf.fp_mask)
                    no_obs = cpuArray(self._make_filter(fov, 0.0, target_device_idx).fp_mask)
                    # Field stop = open pixels + obstructed pixels (the no-obstruction disk)
                    n_stop = np.count_nonzero(mask) + np.count_nonzero((no_obs == 1) & (mask == 0))
                    diameter_arcsec = (self._equivalent_diameter_ld(n_stop, fpf.fft_res)
                                       * self.ld_arcsec)
                    self.assertAlmostEqual(diameter_arcsec / fov, 1.0, delta=0.01)

    @cpu_and_gpu
    def test_obstruction_larger_than_field_stop_raises(self, target_device_idx, xp):
        """fp_obs >= field stop diameter must raise ValueError, a smaller one must not.

        For this geometry (D = 6 m, 500 nm) a 2 arcsec field stop is ~116 lambda/D.
        """
        stop_ld = 2.0 / self.ld_arcsec
        self.assertAlmostEqual(stop_ld, 116.3, delta=0.5)
        with self.assertRaises(ValueError):
            self._make_filter(2.0, stop_ld + 10, target_device_idx)
        with self.assertRaises(ValueError):
            self._make_filter(2.0, 130.0, target_device_idx)
        fpf = self._make_filter(2.0, stop_ld - 10, target_device_idx)  # no exception
        self.assertGreater(np.count_nonzero(cpuArray(fpf.fp_mask)), 0)

    @cpu_and_gpu
    def test_blocked_energy_is_airy_encircled_energy(self, target_device_idx, xp):
        """The focal plane mask blocks the Airy encircled energy within radius fp_obs / 2.

        Expected: for a flat wavefront on a circular pupil, the PSF fraction inside an
        obstruction of diameter d lambda/D is EE = 1 - J0(v)^2 - J1(v)^2 with
        v = pi * d / 2 (v = pi D theta / lambda at theta = d/2 lambda/D); for d = 2
        EE = 0.826, for d = 4 EE = 0.906. The PSF is computed independently with numpy,
        sampled at fft_res = 3 px per lambda/D (disk area quantised, ~0.13 lambda/D
        on the diameter), so the tolerance is 0.03. fov = 3 arcsec uses fov_res = 2.
        """
        for fov in (2.0, 3.0):
            no_obs = self._make_filter(fov, 0.0, target_device_idx)
            n = no_obs.fft_totsize
            n_pup = no_obs.fft_sampling
            pad = (n - n_pup) // 2
            pupil = np.zeros((n, n))
            pupil[pad:pad + n_pup, pad:pad + n_pup] = make_mask(n_pup, xp=np)
            psf = np.abs(np.fft.fft2(pupil)) ** 2
            psf /= psf.sum()
            open_no_obs = np.fft.fftshift(cpuArray(no_obs.fp_mask)).astype(float)

            for fp_obs in (2.0, 4.0):
                with self.subTest(fov=fov, fp_obs=fp_obs):
                    fpf = self._make_filter(fov, fp_obs, target_device_idx)
                    open_obs = np.fft.fftshift(cpuArray(fpf.fp_mask)).astype(float)
                    blocked = 1 - (psf * open_obs).sum() / (psf * open_no_obs).sum()
                    v = np.pi * fp_obs / 2
                    expected = 1 - j0(v) ** 2 - j1(v) ** 2
                    self.assertAlmostEqual(blocked, expected, delta=0.03)

    @cpu_and_gpu
    def test_s0_loss_is_at_least_the_blocked_energy(self, target_device_idx, xp):
        """The S0 lost through the filter is not smaller than the Airy encircled energy.

        Expected: the output field is cropped to the pupil, so it carries at most the
        energy that passes the focal plane mask: 1 - T_obs / T_noobs >= EE(fp_obs / 2),
        with EE = 1 - J0(pi)^2 - J1(pi)^2 = 0.826 for fp_obs = 2 lambda/D. It is not
        equal to EE because the unfiltered pupil halo outside the pupil is also lost
        (measured ~0.875), but it must stay below 1.
        """
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1,
                           target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        t_no_obs = float(self._run_filter(self._make_filter(2.0, 0.0, target_device_idx), ef).S0)
        t_obs = float(self._run_filter(self._make_filter(2.0, 2.0, target_device_idx), ef).S0)
        self.assertGreater(t_no_obs, 0.99)
        self.assertLessEqual(t_no_obs, 1.0)

        lost = 1 - t_obs / t_no_obs
        expected = 1 - j0(np.pi) ** 2 - j1(np.pi) ** 2
        self.assertGreaterEqual(lost, expected - 0.01)
        self.assertLess(lost, 1.0)
