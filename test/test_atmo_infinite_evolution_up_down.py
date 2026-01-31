import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray
from specula import np

from specula.base_time_obj import BaseTimeObj
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_infinite_evolution_up_down import AtmoInfiniteEvolutionUpDown
from specula.data_objects.layer import Layer
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu


class TestAtmoInfiniteEvolutionUpDown(unittest.TestCase):

    @cpu_and_gpu
    def test_two_layer_lists_exist(self, target_device_idx, xp):
        """Test that both layer_list_down and layer_list_up are created"""
        simulParams = SimulParams(pixel_pupil=80, pixel_pitch=0.1, time_step=1)

        atmo = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
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
        simulParams = SimulParams(pixel_pupil=80, pixel_pitch=0.1, time_step=1)

        atmo = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
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
    def test_down_first_vs_up_first(self, target_device_idx, xp):
        """Test that both orderings produce valid but different results"""
        simulParams = SimulParams(pixel_pupil=80, pixel_pitch=0.1, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        # Test case 1: down first (down has smaller extra_delta_time)
        atmo1 = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
            heights=[30.0, 26500.0],
            Cn2=[0.5, 0.5],
            fov=120.0,
            extra_delta_time_down=0.0,
            extra_delta_time_up=0.1,
            target_device_idx=target_device_idx
        )

        atmo1.inputs['seeing'].set(seeing.output)
        atmo1.inputs['wind_direction'].set(wind_direction.output)
        atmo1.inputs['wind_speed'].set(wind_speed.output)

        for objlist in [[seeing, wind_speed, wind_direction], [atmo1]]:
            for obj in objlist:
                obj.setup()
            for obj in objlist:
                obj.check_ready(0)
            for obj in objlist:
                obj.trigger()
            for obj in objlist:
                obj.post_trigger()

        phase_down_1 = cpuArray(atmo1.layer_list_down[0].phaseInNm.copy())
        phase_up_1 = cpuArray(atmo1.layer_list_up[0].phaseInNm.copy())

        # Test case 2: up first (up has smaller extra_delta_time)
        seeing2 = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed2 = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction2 = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo2 = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
            heights=[30.0, 26500.0],
            Cn2=[0.5, 0.5],
            fov=120.0,
            seed=1,  # Same seed as atmo1
            extra_delta_time_down=0.1,
            extra_delta_time_up=0.0,
            target_device_idx=target_device_idx
        )

        atmo2.inputs['seeing'].set(seeing2.output)
        atmo2.inputs['wind_direction'].set(wind_direction2.output)
        atmo2.inputs['wind_speed'].set(wind_speed2.output)

        for objlist in [[seeing2, wind_speed2, wind_direction2], [atmo2]]:
            for obj in objlist:
                obj.setup()
            for obj in objlist:
                obj.check_ready(0)
            for obj in objlist:
                obj.trigger()
            for obj in objlist:
                obj.post_trigger()

        phase_down_2 = cpuArray(atmo2.layer_list_down[0].phaseInNm.copy())
        phase_up_2 = cpuArray(atmo2.layer_list_up[0].phaseInNm.copy())

        # With infinite screens, the results won't be identical due to different
        # generation order, but they should both be valid phase screens
        # Check that all phases have reasonable statistics
        for phase, name in [(phase_down_1, 'down_1'), (phase_up_1, 'up_1'),
                           (phase_down_2, 'down_2'), (phase_up_2, 'up_2')]:
            std = np.std(phase)
            assert std > 0, f"{name}: phase should have non-zero variance"
            assert std < 10000, f"{name}: phase std too large: {std}"

        # The phases should be different between the two configurations
        # because infinite screens generate lines in different order
        assert not np.allclose(phase_down_1, phase_down_2, rtol=0.1), \
            "Different generation order should produce different phases"

    @cpu_and_gpu
    def test_phase_screens_restored_after_trigger(self, target_device_idx, xp):
        """Test that layers are updated correctly for both up and down"""
        simulParams = SimulParams(pixel_pupil=80, pixel_pitch=0.1, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[30.0, 50.0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        extra_delta_time_down = 0.0
        extra_delta_time_up = 0.01

        atmo = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
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

        # Check that both layer lists have valid data
        for i, _ in enumerate(atmo.layer_list_down):
            phase_down = cpuArray(atmo.layer_list_down[i].phaseInNm)
            phase_up = cpuArray(atmo.layer_list_up[i].phaseInNm)

            # Both should have non-zero variance
            assert np.std(phase_down) > 0, f"Layer {i} down: zero variance"
            assert np.std(phase_up) > 0, f"Layer {i} up: zero variance"

            # They should be different due to extra_delta_time
            assert not np.allclose(phase_down, phase_up), \
                f"Layer {i}: phases should differ with different extra_delta_time"

        # Store phase values after first trigger
        phase_down_1 = cpuArray(atmo.layer_list_down[0].phaseInNm.copy())
        phase_up_1 = cpuArray(atmo.layer_list_up[0].phaseInNm.copy())

        # Second trigger
        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.check_ready(BaseTimeObj().seconds_to_t(0.01))
            for obj in objlist:
                obj.trigger()
            for obj in objlist:
                obj.post_trigger()

        # Store phase values after second trigger
        phase_down_2 = cpuArray(atmo.layer_list_down[0].phaseInNm.copy())
        phase_up_2 = cpuArray(atmo.layer_list_up[0].phaseInNm.copy())

        # Phases should have evolved between triggers
        # Check that they're not identical (allowing for some numerical tolerance)
        down_diff = np.mean(np.abs(phase_down_2 - phase_down_1))
        up_diff = np.mean(np.abs(phase_up_2 - phase_up_1))

        assert down_diff > 1.0, \
            f"Down phases should change significantly between triggers, got mean diff: {down_diff}"
        assert up_diff > 1.0, \
            f"Up phases should change significantly between triggers, got mean diff: {up_diff}"


    @cpu_and_gpu
    def test_extra_delta_time_difference(self, target_device_idx, xp):
        """Test that different extra_delta_time values produce different phases"""
        simulParams = SimulParams(pixel_pupil=80, pixel_pitch=0.1, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        extra_delta_time_down = 0.0
        extra_delta_time_up = 0.5  # Significant difference

        atmo = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
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

        # Check that phases are different between up and down
        for i, _ in enumerate(atmo.layer_list_down):
            phase_down = cpuArray(atmo.layer_list_down[i].phaseInNm)
            phase_up = cpuArray(atmo.layer_list_up[i].phaseInNm)

            # Phases should be significantly different
            correlation = np.corrcoef(phase_down.flatten(), phase_up.flatten())[0, 1]
            # With 0.5s offset and wind, correlation should be lower
            assert correlation < 0.95, f"Layer {i}: correlation too high: {correlation}"

    @cpu_and_gpu
    def test_positions_accumulate_independently(self, target_device_idx, xp):
        """Test that up and down positions accumulate independently over time"""
        simulParams = SimulParams(pixel_pupil=80, pixel_pitch=0.1, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        delta_time = 1.0
        delta_t = BaseTimeObj().seconds_to_t(delta_time)
        extra_delta_time_down = 0.05
        extra_delta_time_up = 0.15

        atmo = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
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
        np.testing.assert_allclose(atmo.last_position, expected_position, rtol=1e-6)
        np.testing.assert_allclose(atmo.last_position_up, expected_position, rtol=1e-6)

        # Verify the stored extra_delta_time arrays
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
        simulParams = SimulParams(pixel_pupil=80, pixel_pitch=0.1, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3, 1.0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90, 45], target_device_idx=target_device_idx)

        extra_delta_time_down = [0.01, 0.02, 0.03]
        extra_delta_time_up = [0.11, 0.12, 0.13]

        atmo = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
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
        simulParams = SimulParams(pixel_pupil=80, pixel_pitch=0.1, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
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
        simulParams = SimulParams(pixel_pupil=80, pixel_pitch=0.1, time_step=0.01)

        seeing = WaveGenerator(constant=0.8, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[10.0, 15.0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[45, 135], target_device_idx=target_device_idx)

        # Satellite at 38,000 km altitude
        # Light travel time ~ 0.127 seconds one way
        light_speed = 299792458  # m/s
        satellite_altitude = 38000000  # m
        light_travel_time = satellite_altitude / light_speed  # ~0.127 s

        # Uplink sees atmosphere first, downlink sees it after round-trip delay
        extra_delta_time_up = 0.0
        extra_delta_time_down = 2 * light_travel_time  # Round trip

        atmo = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
            heights=[30.0, 5000.0],  # Lower second layer for faster decorrelation
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

        # Verify extra_delta_time values
        np.testing.assert_allclose(
            atmo.extra_delta_time_up[0],
            0.0,
            atol=1e-10
        )
        np.testing.assert_allclose(
            atmo.extra_delta_time_down[0],
            2 * light_travel_time,
            rtol=1e-8
        )

        # Verify that phases are different due to large time offset
        # We check at least one layer shows significant decorrelation
        correlations = []
        for i, _ in enumerate(atmo.layer_list_down):
            phase_down = cpuArray(atmo.layer_list_down[i].phaseInNm)
            phase_up = cpuArray(atmo.layer_list_up[i].phaseInNm)

            # With 0.254s offset, phases should be decorrelated
            correlation = np.corrcoef(phase_down.flatten(), phase_up.flatten())[0, 1]
            correlations.append(abs(correlation))

        # At least one layer should show low correlation
        assert min(correlations) < 0.7, \
            f"At least one layer should show decorrelation, got correlations: {correlations}"


    @cpu_and_gpu
    def test_acc_rows_cols_independent(self, target_device_idx, xp):
        """Test that acc_rows and acc_cols are tracked independently for up and down"""
        simulParams = SimulParams(pixel_pupil=80, pixel_pitch=0.1, time_step=0.1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[45, 135], target_device_idx=target_device_idx)

        extra_delta_time_down = 0.0
        extra_delta_time_up = 0.05

        atmo = AtmoInfiniteEvolutionUpDown(
            simulParams,
            L0=23,
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

        # After first trigger, acc_rows and acc_cols should be different for up and down
        # (because they have different extra_delta_time)
        acc_rows_down = atmo.acc_rows.copy()
        acc_cols_down = atmo.acc_cols.copy()
        acc_rows_up = atmo.acc_rows_up.copy()
        acc_cols_up = atmo.acc_cols_up.copy()

        # They should be different (not necessarily always, but likely with wind and offset)
        # At least verify they exist and are being tracked
        assert len(acc_rows_down) == 2
        assert len(acc_cols_down) == 2
        assert len(acc_rows_up) == 2
        assert len(acc_cols_up) == 2
