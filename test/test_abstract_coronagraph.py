import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cpuArray
from specula.lib.make_mask import make_mask
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.abstract_coronagraph import Coronagraph

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
    def test_output_field_size_with_smaller_fov(self, target_device_idx, xp):
        """Test that output ElectricField has expected size"""
        coro = SimpleCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov*0.8,
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
    def test_rebinning_preserves_complex_amplitude_and_phase(self, target_device_idx, xp):
        """Test that toccd rebinning handles complex fields properly by splitting amp and phase"""
        coro = SimpleCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            fov=self.fov * 0.5,  # Force rebinning by changing the field of view
            target_device_idx=target_device_idx
        )

        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        
        # Set a constant amplitude
        ef.A[:] = 1.0
        
        # Create a spatial phase gradient (e.g., a ramp). 
        # This is crucial: if toccd rebins complex numbers directly, 
        # linear interpolation of different phases will cause the amplitude to collapse.
        x = xp.linspace(-xp.pi, xp.pi, self.pixel_pupil)
        X, Y = xp.meshgrid(x, x)
        ef.phaseInNm[:] = (X / (2 * xp.pi)) * self.wavelength_nm 
        ef.generation_time = 1

        # Perform the propagation through the coronagraph
        # (In the old code without the PR, this would fail on GPU with a TypeError
        # and return incorrect amplitudes on CPU)
        out_ef = self.get_coro_field(coro, ef)
        
        # Verify that the mean amplitude has not collapsed due to 
        # incorrect summations over complex numbers
        mean_amplitude = xp.mean(out_ef.A)
        self.assertTrue(
            mean_amplitude > 0.9, 
            f"Mean amplitude dropped to {mean_amplitude}. Probable rebinning error on complex arrays."
        )
    
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

        # Check that it is an array with the appropriate shape
        self.assertEqual(coro.phase_shift.shape, (coro.fft_totsize, coro.fft_totsize),
                        "phase_shift should have shape (fft_totsize, fft_totsize)")
