import specula
specula.init(0)

import unittest
import numpy as np
from specula.processing_objects.extended_source import ExtendedSource
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.data_objects.electric_field import ElectricField
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from test.specula_testlib import cpu_and_gpu

class TestExtendedSource(unittest.TestCase):

    debug_plot = False  # Set to True to enable plotting for debugging
    simul_params = SimulParams(
        pixel_pupil=160,
        pixel_pitch=0.05,
        zenithAngleInDeg=30.0
    )
    size_obj = 0.2
    sampling_lambda_over_d = 1.0
    wavelengthInNm = 589.0

    @cpu_and_gpu
    def test_point_source(self, target_device_idx, xp):
        src = ExtendedSource(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelengthInNm,
            source_type='POINT_SOURCE',
            sampling_lambda_over_d=self.sampling_lambda_over_d,
            size_obj=None,
            sampling_type='CARTESIAN'
        )
        src.compute()
        if self.debug_plot:
            src.plot_source()
        self.assertEqual(src.npoints, 1)
        self.assertTrue(np.allclose(src.coeff_flux, 1.0))

    @cpu_and_gpu
    def test_tophat_cartesian(self, target_device_idx, xp):
        src = ExtendedSource(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelengthInNm,
            source_type='TOPHAT',
            sampling_lambda_over_d=self.sampling_lambda_over_d,
            size_obj=self.size_obj,
            sampling_type='CARTESIAN'
        )
        src.compute()
        if self.debug_plot:
            src.plot_source()
        self.assertGreater(src.npoints, 1)
        self.assertAlmostEqual(np.sum(src.coeff_flux), 1.0, places=6)

    @cpu_and_gpu
    def test_gauss_cartesian(self, target_device_idx, xp):
        src = ExtendedSource(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelengthInNm,
            source_type='GAUSS',
            sampling_lambda_over_d=self.sampling_lambda_over_d,
            size_obj=self.size_obj,
            sampling_type='CARTESIAN'
        )
        src.compute()
        if self.debug_plot:
            src.plot_source()
        self.assertGreater(src.npoints, 1)
        self.assertAlmostEqual(np.sum(src.coeff_flux), 1.0, places=6)

    @cpu_and_gpu
    def test_gauss_cartesian_3d(self, target_device_idx, xp):
        src = ExtendedSource(
            simul_params=self.simul_params,
            focus_height=90000.0,
            layer_height=[70000.0, 80000.0, 90000.0, 100000.0, 110000.0],
            intensity_profile=[0.1, 0.23, 0.34, 0.23, 0.1],
            wavelengthInNm=self.wavelengthInNm,
            source_type='GAUSS',
            sampling_lambda_over_d=self.sampling_lambda_over_d,
            size_obj=self.size_obj,
            sampling_type='CARTESIAN'
        )
        src.compute()
        if self.debug_plot:
            src.plot_source()
        self.assertGreater(src.npoints, 1)
        self.assertAlmostEqual(float(np.sum(src.coeff_flux)), 1.0, places=6)

    @cpu_and_gpu   
    def test_extended_source_in_pyramid(self, target_device_idx, xp):
        pixel_pupil = self.simul_params.pixel_pupil
        pixel_pitch = self.simul_params.pixel_pitch
        output_resolution = 80
        t = 1
        
        # Create an extended source
        src = ExtendedSource(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelengthInNm,
            source_type='GAUSS',
            sampling_lambda_over_d=self.sampling_lambda_over_d,
            size_obj=self.size_obj,
            sampling_type='CARTESIAN'
        )
        src.compute()

        # Pass it to the pyramid
        pyr = ModulatedPyramid(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelengthInNm,
            fov=2.0,
            pup_diam=30,
            output_resolution=output_resolution,
            mod_amp=3.0
        )

        # Flat wavefr
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        pyr.inputs['in_ef'].set(ef)
        pyr.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr.setup()
        pyr.check_ready(t)

        # Check that the extended source is loaded and parameters are consistent
        self.assertEqual(pyr.mod_steps, src.npoints)
        self.assertEqual(pyr.ttexp.shape[1], src.npoints)
        self.assertEqual(pyr.flux_factor_vector.shape[0], src.npoints)
        self.assertAlmostEqual(float(np.sum(specula.cpuArray(pyr.flux_factor_vector))), 1.0, places=6)
        self.assertEqual(pyr.ttexp.shape[2:], pyr.tilt_x.shape)

        # Optionally, plot for debug
        if self.debug_plot:
            src.plot_source()

        # Set up the electric field and trigger the pyramid computation
        t = 1
        flat_ef = ElectricField(self.simul_params.pixel_pupil,
                                self.simul_params.pixel_pupil,
                                self.simul_params.pixel_pitch,
                                S0=1,
                                target_device_idx=target_device_idx)
        flat_ef.generation_time = t
        pyr.inputs['in_ef'].set(flat_ef)
        pyr.setup()
        pyr.check_ready(t)
        pyr.trigger()
        pyr.post_trigger()

        # pyramid output
        intensity = pyr.outputs['out_i'].i.copy()
        # Check shape of the output intensity
        self.assertEqual(intensity.shape, (output_resolution, output_resolution))
        # Check that the max intensity is non-zero and minimum intensity is non-negative
        self.assertGreater(np.max(intensity), 0)
        self.assertGreaterEqual(np.min(intensity), 0)
        
    @cpu_and_gpu   
    def test_extended_source_psf_update(self, target_device_idx, xp):
        # Create PSF-based extended source
        psf = np.random.random((64, 64))
        psf /= np.sum(psf)  # Normalize
        
        src = ExtendedSource(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelengthInNm,
            source_type='FROM_PSF',
            sampling_lambda_over_d=self.sampling_lambda_over_d,
            initial_psf=psf,
            pixel_scale_psf=0.1,
            sampling_type='CARTESIAN'
        )

        # Create a new PSF to update with
        new_psf = BaseValue(target_device_idx=target_device_idx)
        new_psf.value = xp.random.random((64, 64))
        new_psf.value /= xp.sum(new_psf.value)
        new_psf.generation_time = 1

        # Connect PSF input
        src.inputs['psf'].set(new_psf)

        # Setup and trigger
        src.setup()
        src.check_ready(1)
        original_flux = src.coeff_flux.copy()

        src.trigger()  # This should update the PSF and recompute
        src.post_trigger()

        # Verify the flux coefficients changed
        self.assertFalse(np.allclose(original_flux, src.coeff_flux))