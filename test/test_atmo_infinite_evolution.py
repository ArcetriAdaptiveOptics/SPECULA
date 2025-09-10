
import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray
from specula import np

from specula.base_time_obj import BaseTimeObj
from specula.data_objects.source import Source
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_infinite_evolution import AtmoInfiniteEvolution
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu


class TestAtmoInfiniteEvolution(unittest.TestCase):

    @cpu_and_gpu
    def test_infinite_evolution_layer_size(self, target_device_idx, xp):
        '''
        Test that the output layer size is the correct one
        '''
        pixel_pupil = 160
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=0.05, time_step=1)

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.5], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        lgs1_source = Source( polar_coordinates=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)

        atmo = AtmoInfiniteEvolution(simul_params,
                             L0=23,  # [m] Outer scale
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
               obj.post_trigger()

        for ii in range(len(atmo.outputs['layer_list'])):
            layer = atmo.outputs['layer_list'][ii]
            assert layer.size == (atmo.pixel_layer_size[ii], atmo.pixel_layer_size[ii])

    @cpu_and_gpu
    def test_wrong_seeing_length_is_checked(self, target_device_idx, xp):

        simul_params = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=[0.65, 0.1], target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo = AtmoInfiniteEvolution(simul_params,
                             L0=23,  # [m] Outer scale
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        for obj in [seeing, wind_speed, wind_direction]:
            obj.setup()

        with self.assertRaises(ValueError):
            atmo.setup()

    @cpu_and_gpu
    def test_wrong_wind_speed_length_is_checked(self, target_device_idx, xp):

        simul_params = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.2, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[8.5, 5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo = AtmoInfiniteEvolution(simul_params,
                             L0=23,  # [m] Outer scale
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        for obj in [seeing, wind_speed, wind_direction]:
            obj.setup()

        with self.assertRaises(ValueError):
            atmo.setup()

    @cpu_and_gpu
    def test_wrong_wind_speed_direction_is_checked(self, target_device_idx, xp):

        simul_params = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.2, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[90, 0, 90], target_device_idx=target_device_idx)

        atmo = AtmoInfiniteEvolution(simul_params,
                             L0=23,  # [m] Outer scale
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        for obj in [seeing, wind_speed, wind_direction]:
            obj.setup()

        with self.assertRaises(ValueError):
            atmo.setup()

    @cpu_and_gpu
    def test_extra_delta_time(self, target_device_idx, xp):

        simul_params = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        delta_time = 1.0
        delta_t = BaseTimeObj().seconds_to_t(delta_time)
        extra_delta_time = 0.1

        atmo = AtmoInfiniteEvolution(simul_params,
                             L0=23,  # [m] Outer scale
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             extra_delta_time=extra_delta_time,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(0)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
               obj.post_trigger()

            for obj in objlist:
                obj.check_ready(delta_t)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
               obj.post_trigger()

        assert atmo.delta_time == delta_time + extra_delta_time

    @cpu_and_gpu
    def test_scale_coeff_with_different_seeing(self, target_device_idx, xp):
        """
        Test that scale_coeff is applied correctly with different seeing values
        """
        pixel_pupil = 160
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=0.05, time_step=1)

        # Test with two different seeing values
        seeing_values = [0.65, 1.3]  # arcsec - second value is double the first

        results = []

        for seeing_val in seeing_values:
            # Create components with specific seeing
            seeing = WaveGenerator(constant=seeing_val, target_device_idx=target_device_idx)
            wind_speed = WaveGenerator(constant=[5.5, 2.5], target_device_idx=target_device_idx)
            wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

            atmo = AtmoInfiniteEvolution(simul_params,
                                    L0=23,  # [m] Outer scale
                                    heights=[30.0, 26500.0], # [m] layer heights at 0 zenith angle
                                    Cn2=[0.5, 0.5], # Cn2 weights (total must be eq 1)
                                    fov=120.0,
                                    target_device_idx=target_device_idx)

            atmo.inputs['seeing'].set(seeing.output)
            atmo.inputs['wind_direction'].set(wind_direction.output)
            atmo.inputs['wind_speed'].set(wind_speed.output)

            # Setup and run one iteration
            for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
                for obj in objlist:
                    obj.setup()

                for obj in objlist:
                    obj.check_ready(1)

                for obj in objlist:
                    obj.trigger()

                for obj in objlist:
                    obj.post_trigger()

            # Store results for comparison
            result = {
                'seeing': seeing_val,
                'scale_coeff': float(atmo.scale_coeff),
                'ref_r0': float(atmo.ref_r0),
                'layer_phases': [cpuArray(layer.phaseInNm) for layer in atmo.layer_list]
            }
            results.append(result)

        # Verify scaling relationships
        seeing1, seeing2 = seeing_values
        result1, result2 = results

        print(f"Seeing 1: {seeing1} arcsec, scale_coeff: {result1['scale_coeff']:.6f}")
        print(f"Seeing 2: {seeing2} arcsec, scale_coeff: {result2['scale_coeff']:.6f}")

        # Test 1: Verify that scale_coeff is scaling with seeing
        seeing_ratio = result1['scale_coeff'] / result2['scale_coeff']
        expected_seeing_ratio = seeing1**(5/6) / seeing2**(5/6)
        self.assertAlmostEqual(seeing_ratio, expected_seeing_ratio, places=2,
            msg=f"scale_coeff ratio {seeing_ratio:.3f} should equal seeing_ratio^(6/5) = {expected_seeing_ratio:.3f}")

        # Test 2: Verify that the actual phase scaling in layers follows the seeing relationship
        # Since scale_coeff is the same, the phases should be identical (current implementation)
        for i in range(len(atmo.layer_list)):
            phase1 = result1['layer_phases'][i]
            phase2 = result2['layer_phases'][i]

            # Check that phases have the same RMS (since scale_coeff is the same)
            rms1 = float(xp.sqrt(xp.mean(phase1**2)))
            rms2 = float(xp.sqrt(xp.mean(phase2**2)))

            print(f"Layer {i}: RMS phase seeing={seeing1}: {rms1:.3f}, seeing={seeing2}: {rms2:.3f}")

            # Check that the RMS ratio matches the expected seeing ratio
            self.assertAlmostEqual(rms1/rms2, expected_seeing_ratio, places=2,
                msg=f"RMS phase ratio {rms1/rms2:.3f} should equal seeing_ratio^(6/5) = {expected_seeing_ratio:.3f}")

    @cpu_and_gpu
    def test_pupil_distances_are_scaled_by_airmass(self, target_device_idx, xp):
        """
        Test that pupil_distances are correctly computed as heights * airmass
        """
        pixel_pupil = 160
        zenith = 30.0  # degrees
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=0.05, zenithAngleInDeg=zenith, time_step=1)
        heights = [1000.0, 5000.0, 12000.0]
        airmass = 1.0 / np.cos(np.radians(zenith))
        atmo = AtmoInfiniteEvolution(simul_params,
                                    L0=23,
                                    heights=heights,
                                    Cn2=[1/3, 1/3, 1/3],
                                    fov=120.0,
                                    target_device_idx=target_device_idx)
        expected = cpuArray(heights) * airmass
        np.testing.assert_allclose(atmo.pupil_distances, expected, rtol=1e-8)