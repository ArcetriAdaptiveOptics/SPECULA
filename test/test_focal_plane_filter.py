import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cpuArray
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
        fp_obs = 2 * (self.wavelength_nm * 1e-9) / (self.pixel_pupil * self.pixel_pitch) * 206265  # in arcsec

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
        self.assertFalse(fpf_small_fov._do_interpolation, "Small FoV should not require interpolation")
        self.assertTrue(fpf_large_fov._do_interpolation, "Large FoV should require interpolation")

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
        if fpf._do_interpolation:
            self.assertGreater(fpf.fft_sampling, self.pixel_pupil, 
                             "FFT sampling should be larger than pixel_pupil when interpolation is used")
            
            # Check that the interpolated field has the correct size
            self.assertEqual(fpf._wf_interpolated.A.shape[0], fpf.fft_sampling)
            self.assertEqual(fpf._wf_interpolated.A.shape[1], fpf.fft_sampling)
            
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