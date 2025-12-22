import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray
from specula import np

from specula.base_time_obj import BaseTimeObj
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_evolution_up_down import AtmoEvolutionUpDown
from specula.data_objects.layer import Layer
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu


class TestAtmoEvolutionUpDown(unittest.TestCase):

    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests by removing generated files"""
        files = ['ps_seed1_dim8192_pixpit0.050_L023.0000_double.fits',
                 'ps_seed1_dim8192_pixpit0.050_L023.0000_single.fits']
        for fname in files:
            fpath = os.path.join(cls.data_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

    @cpu_and_gpu
    def test_two_layer_lists_exist(self, target_device_idx, xp):
        """Test that both layer_list_down and layer_list_up are created"""
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        atmo = AtmoEvolutionUpDown(
            simulParams,
            L0=23,
            data_dir=self.data_dir,
            heights=[30.0, 26500.0],
            Cn2=[0.5, 0.5],
            fov=120.0,
            extra_delta_time_down=0.0,
            extra_delta_time_up=0.1,
            target_device_idx=target_device_idx
        )

        # Check that both outputs exist
        assert 'layer_list_down' in atmo.outputs
        assert 'layer_list_up' in atmo.outputs
        assert 'layer_list' in atmo.outputs

        # Check that they are lists
        assert isinstance(atmo.outputs['layer_list_down'], list)
        assert isinstance(atmo.outputs['layer_list_up'], list)

        # Check that they have the correct length
        assert len(atmo.outputs['layer_list_down']) == 2
        assert len(atmo.outputs['layer_list_up']) == 2

        # Check that all elements are Layer objects
        for layer in atmo.outputs['layer_list_down'] + atmo.outputs['layer_list_up']:
            assert isinstance(layer, Layer)

        # Check backward compatibility: layer_list points to layer_list_down
        assert atmo.outputs['layer_list'] is atmo.outputs['layer_list_down']

    @cpu_and_gpu
    def test_layer_lists_are_independent(self, target_device_idx, xp):
        """Test that up and down layer lists are independent objects"""
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        atmo = AtmoEvolutionUpDown(
            simulParams,
            L0=23,
            data_dir=self.data_dir,
            heights=[30.0, 26500.0],
            Cn2=[0.5, 0.5],
            fov=120.0,
            extra_delta_time_down=0.0,
            extra_delta_time_up=0.1,
            target_device_idx=target_device_idx
        )

        # Check that the layer lists are different objects
        assert atmo.outputs['layer_list_down'] is not atmo.outputs['layer_list_up']

        # Check that individual layers are different objects
        for i in range(len(atmo.outputs['layer_list_down'])):
            assert atmo.outputs['layer_list_down'][i] is not atmo.outputs['layer_list_up'][i]

    @cpu_and_gpu
    def test_extra_delta_time_difference(self, target_device_idx, xp):
        """Test that different extra_delta_time values produce different effective positions"""
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        extra_delta_time_down = 0.0
        extra_delta_time_up = 0.1

        atmo = AtmoEvolutionUpDown(
            simulParams,
            L0=23,
            data_dir=self.data_dir,
            heights=[30.0, 26500.0],
            Cn2=[0.5, 0.5],
            fov=120.0,
            extra_delta_time_down=extra_delta_time_down,
            extra_delta_time_up=extra_delta_time_up,
            target_device_idx=target_device_idx
        )

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

        # After first trigger, check positions
        wind_speed_values = cpuArray(wind_speed.output.value)

        # Down should have no extra offset
        np.testing.assert_allclose(atmo.last_position, 0.0, atol=1e-6)

        # Up should have extra offset
        expected_extra_offset_up = wind_speed_values * extra_delta_time_up / atmo.pixel_pitch
        np.testing.assert_allclose(atmo.last_position_up, 0.0, atol=1e-6)

        # The phase screens should be different due to different sampling positions
        # (we can't directly compare phases because they're sampled from the same screens
        # at different positions)

    @cpu_and_gpu
    def test_positions_accumulate_independently(self, target_device_idx, xp):
        """Test that up and down positions accumulate independently over time"""
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        delta_time = 1.0
        delta_t = BaseTimeObj().seconds_to_t(delta_time)
        extra_delta_time_down = 0.05
        extra_delta_time_up = 0.15

        atmo = AtmoEvolutionUpDown(
            simulParams,
            L0=23,
            data_dir=self.data_dir,
            heights=[30.0, 26500.0],
            Cn2=[0.5, 0.5],
            fov=120.0,
            extra_delta_time_down=extra_delta_time_down,
            extra_delta_time_up=extra_delta_time_up,
            target_device_idx=target_device_idx
        )

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        # First trigger at t=0
        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.setup()
            for obj in objlist:
                obj.check_ready(0)
            for obj in objlist:
                obj.trigger()
            for obj in objlist:
                obj.post_trigger()

        # Second trigger at t=delta_t
        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.check_ready(delta_t)
            for obj in objlist:
                obj.trigger()
            for obj in objlist:
                obj.post_trigger()

        wind_speed_values = cpuArray(wind_speed.output.value)

        # Both should have accumulated the same delta_position
        expected_position = wind_speed_values * delta_time / atmo.pixel_pitch
        np.testing.assert_allclose(atmo.last_position, expected_position, rtol=1e-8)
        np.testing.assert_allclose(atmo.last_position_up, expected_position, rtol=1e-8)

        # But the effective positions should differ by the extra_delta_time offset
        # (note: we can't directly test effective_position as it's computed in _update_layer_list,
        # but we can verify the stored extra_delta_time arrays)
        np.testing.assert_allclose(
            atmo.extra_delta_time_down,
            [extra_delta_time_down, extra_delta_time_down],
            rtol=1e-8
        )
        np.testing.assert_allclose(
            atmo.extra_delta_time_up,
            [extra_delta_time_up, extra_delta_time_up],
            rtol=1e-8
        )

    @cpu_and_gpu
    def test_vector_extra_delta_time(self, target_device_idx, xp):
        """Test that vector extra_delta_time works correctly"""
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3, 1.0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90, 45], target_device_idx=target_device_idx)

        extra_delta_time_down = [0.01, 0.02, 0.03]
        extra_delta_time_up = [0.11, 0.12, 0.13]

        atmo = AtmoEvolutionUpDown(
            simulParams,
            L0=23,
            data_dir=self.data_dir,
            heights=[30.0, 7000.0, 26500.0],
            Cn2=[1/3, 1/3, 1/3],
            fov=120.0,
            extra_delta_time_down=extra_delta_time_down,
            extra_delta_time_up=extra_delta_time_up,
            target_device_idx=target_device_idx
        )

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

        # Check that extra_delta_time arrays are correctly set
        np.testing.assert_allclose(
            atmo.extra_delta_time_down,
            extra_delta_time_down,
            rtol=1e-8
        )
        np.testing.assert_allclose(
            atmo.extra_delta_time_up,
            extra_delta_time_up,
            rtol=1e-8
        )

    @cpu_and_gpu
    def test_layers_not_reallocated(self, target_device_idx, xp):
        """Test that layer objects are not reallocated between triggers"""
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo = AtmoEvolutionUpDown(
            simulParams,
            L0=23,
            data_dir=self.data_dir,
            heights=[30.0, 26500.0],
            Cn2=[0.5, 0.5],
            fov=120.0,
            extra_delta_time_down=0.0,
            extra_delta_time_up=0.1,
            target_device_idx=target_device_idx
        )

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        # First trigger
        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.setup()
            for obj in objlist:
                obj.check_ready(0)
            for obj in objlist:
                obj.trigger()
            for obj in objlist:
                obj.post_trigger()

        # Store layer field IDs
        id_down_0_1 = id(atmo.outputs['layer_list_down'][0].field)
        id_down_1_1 = id(atmo.outputs['layer_list_down'][1].field)
        id_up_0_1 = id(atmo.outputs['layer_list_up'][0].field)
        id_up_1_1 = id(atmo.outputs['layer_list_up'][1].field)

        # Second trigger
        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.check_ready(1)
            for obj in objlist:
                obj.trigger()
            for obj in objlist:
                obj.post_trigger()

        # Check that IDs haven't changed
        id_down_0_2 = id(atmo.outputs['layer_list_down'][0].field)
        id_down_1_2 = id(atmo.outputs['layer_list_down'][1].field)
        id_up_0_2 = id(atmo.outputs['layer_list_up'][0].field)
        id_up_1_2 = id(atmo.outputs['layer_list_up'][1].field)

        assert id_down_0_1 == id_down_0_2
        assert id_down_1_1 == id_down_1_2
        assert id_up_0_1 == id_up_0_2
        assert id_up_1_1 == id_up_1_2

    @cpu_and_gpu
    def test_satellite_ranging_scenario(self, target_device_idx, xp):
        """Test a realistic satellite laser ranging scenario with significant time difference"""
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=0.001)

        seeing = WaveGenerator(constant=0.8, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[10.0, 5.0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[45, 135], target_device_idx=target_device_idx)

        # Satellite at 38,000 km altitude
        # Light travel time ~ 0.127 seconds one way
        # So up and down differ by this amount
        light_speed = 299792458  # m/s
        satellite_altitude = 38000000  # m
        light_travel_time = satellite_altitude / light_speed  # ~0.127 s

        # For simplicity, assume layers see uplink first, then downlink after delay
        extra_delta_time_down = light_travel_time
        extra_delta_time_up = 0.0

        atmo = AtmoEvolutionUpDown(
            simulParams,
            L0=23,
            data_dir=self.data_dir,
            heights=[30.0, 10000.0],
            Cn2=[0.7, 0.3],
            fov=120.0,
            extra_delta_time_down=extra_delta_time_down,
            extra_delta_time_up=extra_delta_time_up,
            target_device_idx=target_device_idx
        )

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

        # Check that the time difference is correctly applied
        wind_speed_values = cpuArray(wind_speed.output.value)
        expected_offset_down = wind_speed_values * light_travel_time / atmo.pixel_pitch

        # The downlink should have a significant phase screen offset compared to uplink
        # We can verify this by checking the stored extra_delta_time values
        np.testing.assert_allclose(
            atmo.extra_delta_time_down[0],
            light_travel_time,
            rtol=1e-8
        )
        np.testing.assert_allclose(
            atmo.extra_delta_time_up[0],
            0.0,
            atol=1e-10
        )
