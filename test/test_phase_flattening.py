import specula
specula.init(0)  # Default target device

import tempfile
import os
import gc
import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.phase_flattening import PhaseFlattening

from test.specula_testlib import cpu_and_gpu

class TestPhaseFlattening(unittest.TestCase):

    @cpu_and_gpu
    def test_phase_flattening_basic(self, target_device_idx, xp):
        """Test phase flattening with random values"""
        pixel_pupil = 6
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)

        # Create phase flattener (reuse for both subtests)
        phase_flattener = PhaseFlattening(
            simul_params=simul_params,
            target_device_idx=target_device_idx
        )

        # === SUBTEST 1: Random values with circular pupil ===
        ef_in = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch,
                              S0=1, target_device_idx=target_device_idx, precision=1)

        # Set circular pupil mask
        y, x = xp.ogrid[:pixel_pupil, :pixel_pupil]
        center = pixel_pupil // 2
        radius = pixel_pupil // 3
        mask = (x - center)**2 + (y - center)**2 <= radius**2

        ef_in.A[:] = mask.astype(ef_in.dtype)
        valid_mask = cpuArray(ef_in.A) > 0
  
        # Set phase with known mean
        phase_random = xp.random.randn(pixel_pupil, pixel_pupil) * 10
        phase_random -= xp.mean(phase_random[valid_mask])
        ef_in.phaseInNm[:] = 100.0 + phase_random  # Mean phase is 100

        # Process first case
        phase_flattener.inputs['in_ef'].set(ef_in)

        t = 1
        ef_in.generation_time = t
        phase_flattener.check_ready(t)
        phase_flattener.setup()
        phase_flattener.trigger()
        phase_flattener.post_trigger()

        ef_out = phase_flattener.outputs['out_ef']

        # Check that amplitude and S0 are preserved
        assert np.allclose(cpuArray(ef_out.A), cpuArray(ef_in.A))
        assert ef_out.S0 == ef_in.S0

        # Check that mean phase of valid pixels is close to zero
        if np.any(valid_mask):
            mean_phase = np.mean(cpuArray(ef_out.phaseInNm)[valid_mask])
            assert abs(mean_phase) < 1e-5, f"Mean phase should be ~0, got {mean_phase}"

        # Check expected output values
        actual_output = cpuArray(ef_out.phaseInNm[valid_mask])
        expected_output = phase_random[valid_mask]

        assert np.allclose(actual_output, cpuArray(expected_output), rtol=1e-04, atol=1e-07), \
            f"Expected {expected_output}, got {actual_output}"

        # Check precision
        assert ef_out.field.dtype == np.float64, \
            f"Expected float64, got {ef_out.field.dtype}"

    @cpu_and_gpu
    def test_phase_flattening_edge_cases(self, target_device_idx, xp):
        """Test edge cases"""
        pixel_pupil = 8
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)
        
        # === SUBTEST 1: Checkerboard mask (preserves invalid pixels) ===
        phase_flattener = PhaseFlattening(
            simul_params=simul_params,
            target_device_idx=target_device_idx
        )
        
        ef_in1 = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, 
                              S0=1, target_device_idx=target_device_idx)
        
        # Set checkerboard mask
        mask = xp.zeros((pixel_pupil, pixel_pupil))
        mask[::2, ::2] = 1  # Only some pixels are valid
        mask[1::2, 1::2] = 1
        
        ef_in1.A[:] = mask.astype(ef_in1.dtype)
        
        # Set different phase values for valid and invalid pixels
        ef_in1.phaseInNm[:] = 50.0  # Valid pixels will have mean removed
        ef_in1.phaseInNm[mask == 0] = 999.0  # Invalid pixels should stay unchanged

        # Store original invalid pixel values
        original_invalid_phase = cpuArray(ef_in1.phaseInNm[mask == 0].copy())

        phase_flattener.inputs['in_ef'].set(ef_in1)

        t = 1
        ef_in1.generation_time = t
        phase_flattener.check_ready(t)
        phase_flattener.setup()
        phase_flattener.trigger()
        phase_flattener.post_trigger()

        ef_out1 = phase_flattener.outputs['out_ef']

        # Check that invalid pixels are unchanged
        output_invalid_phase = cpuArray(ef_out1.phaseInNm[cpuArray(mask) == 0])
        assert np.allclose(output_invalid_phase, original_invalid_phase), \
            "Invalid pixels should remain unchanged"

        # === SUBTEST 2: All zero amplitude ===
        ef_in2 = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, 
                              S0=1, target_device_idx=target_device_idx)

        ef_in2.A[:] = 0  # All pixels invalid
        ef_in2.phaseInNm[:] = 123.456  # Arbitrary phase values

        original_phase = cpuArray(ef_in2.phaseInNm.copy())

        phase_flattener.inputs['in_ef'].set(ef_in2)

        t = 2
        ef_in2.generation_time = t
        phase_flattener.check_ready(t)
        phase_flattener.trigger()
        phase_flattener.post_trigger()

        ef_out2 = phase_flattener.outputs['out_ef']

        # Phase should be unchanged since no valid pixels
        assert np.allclose(cpuArray(ef_out2.phaseInNm), original_phase), \
            "Phase should be unchanged when no valid pixels"

    @cpu_and_gpu
    def test_phase_flattening_memory_efficiency(self, target_device_idx, xp):
        """Test that phase flattening doesn't cause memory reallocation"""
        pixel_pupil = 10
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)

        # Create phase flattener
        phase_flattener = PhaseFlattening(
            simul_params=simul_params,
            target_device_idx=target_device_idx
        )
        
        # Create input
        ef_in = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, 
                             S0=1, target_device_idx=target_device_idx)
        ef_in.A[:] = 1
        ef_in.phaseInNm[:] = 100

        phase_flattener.inputs['in_ef'].set(ef_in)

        # First trigger to setup
        t = 1
        ef_in.generation_time = t
        phase_flattener.check_ready(t)
        phase_flattener.setup()
        phase_flattener.trigger()
        phase_flattener.post_trigger()

        # Store output field ID
        output_field_id = id(phase_flattener.outputs['out_ef'].field)

        # Second trigger - should not reallocate
        t = 2
        ef_in.generation_time = t
        phase_flattener.check_ready(t)
        phase_flattener.trigger()
        phase_flattener.post_trigger()

        # Check no reallocation occurred
        assert id(phase_flattener.outputs['out_ef'].field) == output_field_id, \
            "Output field should not be reallocated"