import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cpuArray
from specula.lib.make_mask import make_mask
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.coronagraph import Coronagraph

from test.specula_testlib import cpu_and_gpu


class SimpleCoronagraph(Coronagraph):
    """Simple concrete implementation of abstract Coronagraph with unity masks"""
    
    def make_apodizer(self):
        """Return unity apodizer (no apodization)"""
        return 1.0
    
    def make_focal_plane_mask(self):
        """Return unity focal plane mask"""
        return self.xp.ones((self.fft_totsize, self.fft_totsize),
                           dtype=self.complex_dtype)
    
    def make_pupil_plane_mask(self):
        """Return unity pupil plane mask"""
        return self.xp.ones((self.fft_sampling, self.fft_sampling),
                           dtype=self.complex_dtype)


class TestAbstractCoronagraph(unittest.TestCase):

    def setUp(self):
        # Basic simulation parameters
        self.pixel_pupil = 40
        self.pixel_pitch = 0.05
        self.wavelength_nm = 500
        self.fov = 10.0

        self.simul_params = SimulParams(
            pixel_pupil=self.pixel_pupil,
            pixel_pitch=self.pixel_pitch
        )
        # make a round mask for the pupil
        self.mask = make_mask(self.pixel_pupil, obsratio=0.0, xp=np)

    def get_coro_field(self, coro, in_ef):
        coro.inputs['in_ef'].set(in_ef)
        coro.setup()
        coro.check_ready(1)
        coro.prepare_trigger(1)
        coro.trigger_code()
        coro.post_trigger()
        return coro.outputs['out_ef']

    @cpu_and_gpu
    def test_output_field_size(self, target_device_idx, xp):
        """Test that output ElectricField has expected size"""
        coro = SimpleCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        out_ef = self.get_coro_field(coro, ef)
        
        # Check that output field has the same size as input pupil
        self.assertEqual(out_ef.A.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(out_ef.phaseInNm.shape, (self.pixel_pupil, self.pixel_pupil))

    @cpu_and_gpu
    def test_unity_masks_preserve_field(self, target_device_idx, xp):
        """Test that unity masks approximately preserve the input field"""
        coro = SimpleCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            center_on_pixel=True,
            target_device_idx=target_device_idx
        )

        # Flat wavefront with amplitude 1
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        out_ef = self.get_coro_field(coro, ef)
        
        # With unity masks, the amplitude should be approximately preserved in the pupil plane
        # (allowing for some numerical precision loss during FFT/IFFT operations)
        amplitude_diff = xp.abs(out_ef.A - ef.A).max()
        self.assertLess(cpuArray(amplitude_diff), 0.1, 
                       "Unity masks should approximately preserve amplitude")

    @cpu_and_gpu
    def test_phase_shift_center_on_pixel_true(self, target_device_idx, xp):
        """Test that phase_shift is 1.0 when center_on_pixel is True"""
        coro = SimpleCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            center_on_pixel=True,
            target_device_idx=target_device_idx
        )

        # Trigger setup to initialize phase_shift
        in_ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                              self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        in_ef.A[:] = xp.array(self.mask)
        in_ef.phaseInNm[:] = 0.0
        in_ef.generation_time = 1
        
        coro.inputs['in_ef'].set(in_ef)
        coro.setup()

        # When center_on_pixel is True, phase_shift should be 1.0 (scalar)
        self.assertEqual(coro.phase_shift, 1.0,
                        "phase_shift should be 1.0 when center_on_pixel is True")

    @cpu_and_gpu
    def test_phase_shift_center_on_pixel_false(self, target_device_idx, xp):
        """Test that phase_shift is not 1.0 when center_on_pixel is False"""
        coro = SimpleCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            center_on_pixel=False,
            target_device_idx=target_device_idx
        )

        # Trigger setup to initialize phase_shift
        in_ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                              self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        in_ef.A[:] = xp.array(self.mask)
        in_ef.phaseInNm[:] = 0.0
        in_ef.generation_time = 1
        
        coro.inputs['in_ef'].set(in_ef)
        coro.setup()

        # When center_on_pixel is False, phase_shift should be an array, not 1.0
        self.assertNotEqual(coro.phase_shift, 1.0,
                           "phase_shift should not be 1.0 when center_on_pixel is False")
        # Check that it is an array with the appropriate shape
        self.assertEqual(coro.phase_shift.shape, (2 * coro.fft_totsize, 2 * coro.fft_totsize),
                        "phase_shift should have shape (2*fft_totsize, 2*fft_totsize)")

    @cpu_and_gpu
    def test_unity_focal_plane_mask_shape(self, target_device_idx, xp):
        """Test that focal plane mask has the expected shape with unity mask"""
        coro = SimpleCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            target_device_idx=target_device_idx
        )

        self.assertEqual(coro.fp_mask.shape, (coro.fft_totsize, coro.fft_totsize))
        # Check that it's indeed unity (all ones)
        self.assertTrue(xp.allclose(coro.fp_mask, 1.0),
                       "Focal plane mask should be all ones")

    @cpu_and_gpu
    def test_unity_pupil_plane_mask_shape(self, target_device_idx, xp):
        """Test that pupil plane mask has the expected shape with unity mask"""
        coro = SimpleCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov,
            target_device_idx=target_device_idx
        )

        self.assertEqual(coro.pupil_mask.shape, (coro.fft_sampling, coro.fft_sampling))
        # Check that it's indeed unity (all ones)
        self.assertTrue(xp.allclose(coro.pupil_mask, 1.0),
                       "Pupil plane mask should be all ones")
