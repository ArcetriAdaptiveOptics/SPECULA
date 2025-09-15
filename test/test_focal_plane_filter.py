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

    @cpu_and_gpu
    def setUp(self, target_device_idx, xp):
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
        mask = make_mask(self.pixel_pupil, xp=xp)

        # Flat wavefront
        self.ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        self.ef.generation_time = 1
        self.ef.A[:] = mask
        self.ef.phaseInNm[:] = 0.0

    @cpu_and_gpu
    def test_output_shape(self, target_device_idx, xp):
        """Test that output ElectricField has expected shape"""
        fpf = FocalPlaneFilter(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            target_device_idx=target_device_idx
        )
        fpf.inputs['in_ef'].set(self.ef)
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
        fpf_nofilter.inputs['in_ef'].set(self.ef)
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
        fpf_obs.inputs['in_ef'].set(self.ef)
        fpf_obs.setup()
        fpf_obs.check_ready(1)
        fpf_obs.prepare_trigger(1)
        fpf_obs.trigger_code()
        fpf_obs.post_trigger()
        ef_obs = fpf_obs.outputs['out_ef']

        # Compute PSF for both cases using calc_psf
        psf = calc_psf(self.ef.phaseInNm, self.ef.A, xp=xp, normalize=True)
        psf_nofilter = calc_psf(ef_nofilter.phaseInNm, ef_nofilter.A, xp=xp, normalize=True)
        psf_obs = calc_psf(ef_obs.phaseInNm, ef_obs.A, xp=xp, normalize=True)

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
        self.ef.phaseInNm[:] = 0.0
        fpf.inputs['in_ef'].set(self.ef)
        fpf.setup()
        fpf.check_ready(1)
        fpf.prepare_trigger(1)
        fpf.trigger_code()
        fpf.post_trigger()
        out_ef = fpf.outputs['out_ef']
        # Output phase should be (almost) constant for a flat input
        idx = xp.where(self.ef.A > 0)
        min_phase = float(xp.min(out_ef.phaseInNm[idx]))
        max_phase = float(xp.max(out_ef.phaseInNm[idx]))

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
        self.ef.A[:] = 1.0
        fpf.inputs['in_ef'].set(self.ef)
        fpf.setup()
        fpf.check_ready(1)
        fpf.prepare_trigger(1)
        fpf.trigger_code()
        fpf.post_trigger()
        out_ef = fpf.outputs['out_ef']
        # Output amplitude should not be all zeros and should be approximately the same as the input one
        self.assertGreater(float(out_ef.A.sum()), 0.0)
        self.assertLess(float(out_ef.A.max()), 2.0*float(self.ef.A.max()))