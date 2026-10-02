import os
import glob
import specula
specula.init(0)  # Default target device
from specula.loop_control import LoopControl

import unittest

from specula import cpuArray
from specula import np

from specula.data_objects.source import Source
from specula.base_time_obj import BaseTimeObj
from specula.base_processing_obj import BaseProcessingObj
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.atmo_evolution_up_down import AtmoEvolutionUpDown
from specula.processing_objects.base_slicer import BaseSlicer
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.layer import Layer
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu


class TestAtmoEvolution(unittest.TestCase):

    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests by removing generated files"""
        pattern = 'ps_seed*_pixpit0.050_L023.0000_*.fits'
        for fpath in glob.glob(os.path.join(cls.data_dir, pattern)):
            if os.path.exists(fpath):
                os.remove(fpath)

    @cpu_and_gpu
    def test_atmo(self, target_device_idx, xp):
        '''Test that a basic AtmoEvolution and AtmoPropagation setup executes without exceptions'''
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.5], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        lgs1_source = Source( polar_coordinates=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=self.data_dir,
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        prop = AtmoPropagation(simulParams,                               
                               source_dict = {'on_axis_source': on_axis_source,
                                               'lgs1_source': lgs1_source},
                               target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])

        # No exceptions should be raised during the loop execution, and outputs should be generated

        loop = LoopControl()
        loop.add(seeing, idx=0)
        loop.add(wind_speed, idx=0)
        loop.add(wind_direction, idx=0)
        loop.add(atmo, idx=1)
        loop.run(run_time=1, dt=1)

        assert 'out_on_axis_source_ef' in prop.outputs
        assert 'out_lgs1_source_ef' in prop.outputs

    @cpu_and_gpu
    def test_that_wrong_Cn2_total_is_detected(self, target_device_idx, xp):

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05)

        with self.assertRaises(ValueError):
            atmo = AtmoEvolution(simulParams,
                                L0=23,  # [m] Outer scale
                                data_dir=self.data_dir,
                                heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                                Cn2 = [0.2, 0.2], # Cn2 weights (total must be eq 1)
                                fov = 120.0,
                                target_device_idx=target_device_idx)

        # Total is 1, no exception raised.
        atmo = AtmoEvolution(simulParams,
                            L0=23,  # [m] Outer scale
                            data_dir=self.data_dir,
                            heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                            Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                            fov = 120.0,
                            target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_layer_list_type_length_and_element_types(self, target_device_idx, xp):

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05)

        atmo = AtmoEvolution(simulParams,
                            L0=23,  # [m] Outer scale
                            data_dir=self.data_dir,
                            heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                            Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                            fov = 120.0,
                            target_device_idx=target_device_idx)
            
        assert isinstance(atmo.outputs['layer_list'], list)
        assert len(atmo.outputs['layer_list']) == 2
        
        for layer in atmo.outputs['layer_list']:
            assert isinstance(layer, Layer)

    @cpu_and_gpu
    def test_atmo_evolution_layers_are_not_reallocated(self, target_device_idx, xp):

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=self.data_dir,
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        loop = LoopControl()
        loop.add(seeing, idx=0)
        loop.add(wind_speed, idx=0)
        loop.add(wind_direction, idx=0)
        loop.add(atmo, idx=1)
        loop.start(run_time=2, dt=1)
        loop.iter()

        id_a1 = id(atmo.outputs['layer_list'][0].field)
        id_b1 = id(atmo.outputs['layer_list'][1].field)

        loop.iter()

        id_a2 = id(atmo.outputs['layer_list'][0].field)
        id_b2 = id(atmo.outputs['layer_list'][1].field)

        assert id_a1 == id_a2
        assert id_b1 == id_b2

    @cpu_and_gpu
    def test_wrong_seeing_length_is_checked(self, target_device_idx, xp):

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=[0.65, 0.1], target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=self.data_dir,
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

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.2, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[8.5, 5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=self.data_dir,
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

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.2, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[90, 0, 90], target_device_idx=target_device_idx)

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=self.data_dir,
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

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        delta_time = 1.0
        delta_t = BaseTimeObj().seconds_to_t(delta_time)
        extra_delta_time = 0.1

        atmo = AtmoEvolution(simulParams,
                            L0=23,  # [m] Outer scale
                            data_dir=self.data_dir,
                            heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                            Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                            fov = 120.0,
                            extra_delta_time=extra_delta_time,
                            target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        loop = LoopControl()
        loop.add(seeing, idx=0)
        loop.add(wind_speed, idx=0)
        loop.add(wind_direction, idx=0)
        loop.add(atmo, idx=1)
        loop.start(run_time=delta_time*2, dt=delta_time)
        loop.iter()

        # After first trigger, last_position should be approximately zero
        np.testing.assert_allclose(cpuArray(atmo.last_position), 0.0, atol=1e-6)

        # last_effective_position should contain the extra_offset
        wind_speed_values = cpuArray(wind_speed.output.value)
        expected_extra_offset = wind_speed_values * extra_delta_time / atmo.pixel_pitch
        np.testing.assert_allclose(
            cpuArray(atmo.last_effective_position), expected_extra_offset, rtol=1e-8
        )

        # Second trigger
        loop.iter()

        # After second trigger, verify that:
        # 1. delta_time does not contain extra_delta_time
        assert atmo.delta_time == delta_time

        # 2. last_position has accumulated only delta_position (not extra_offset)
        expected_last_position = wind_speed_values * delta_time / atmo.pixel_pitch
        np.testing.assert_allclose(
            cpuArray(atmo.last_position), expected_last_position, rtol=1e-8
        )

        # 3. last_effective_position = last_position + extra_offset
        expected_effective_position = expected_last_position + expected_extra_offset
        np.testing.assert_allclose(
            cpuArray(atmo.last_effective_position), expected_effective_position, rtol=1e-8
        )

    @cpu_and_gpu
    def test_extra_delta_time_vector(self, target_device_idx, xp):

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)

        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.3, 1.0, 1.0],
                                target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 90, 180, 90],
                                    target_device_idx=target_device_idx)

        delta_time = 1.0
        delta_t = BaseTimeObj().seconds_to_t(delta_time)
        extra_delta_time = [0.1, 0.2, 0.3, 0.4]

        atmo = AtmoEvolution(simulParams,
                            L0=23,  # [m] Outer scale
                            data_dir=self.data_dir,
                            heights=[30.0, 7000.0, 10000.0, 26500.0],  # [m] layer heights at 0 zenith angle
                            Cn2=[0.25, 0.25, 0.25, 0.25],  # Cn2 weights (total must be eq 1)
                            fov=120.0,
                            extra_delta_time=extra_delta_time,
                            target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        loop = LoopControl()
        loop.add(seeing, idx=0)
        loop.add(wind_speed, idx=0)
        loop.add(wind_direction, idx=0)
        loop.add(atmo, idx=1)
        loop.start(run_time=delta_time*2, dt=delta_time)
        loop.iter()

        # After first trigger, last_position should be approximately zero
        np.testing.assert_allclose(cpuArray(atmo.last_position), 0.0, atol=1e-6)
        
        # last_effective_position should contain the extra_offset
        wind_speed_values = cpuArray(wind_speed.output.value)
        expected_extra_offset = wind_speed_values * np.array(extra_delta_time) / atmo.pixel_pitch
        np.testing.assert_allclose(
            cpuArray(atmo.last_effective_position), expected_extra_offset, rtol=1e-8
        )

        loop.iter()

        # After second trigger, verify that:
        # 1. delta_time does not contain extra_delta_time
        assert atmo.delta_time == delta_time
        
        # 2. last_position has accumulated only delta_position (not extra_offset)
        expected_last_position = wind_speed_values * delta_time / atmo.pixel_pitch
        np.testing.assert_allclose(
            cpuArray(atmo.last_position), expected_last_position, rtol=1e-8
        )
        
        # 3. last_effective_position = last_position + extra_offset
        expected_effective_position = expected_last_position + expected_extra_offset
        np.testing.assert_allclose(
            cpuArray(atmo.last_effective_position), expected_effective_position, rtol=1e-8
        )

    @cpu_and_gpu
    def test_pupil_distances_are_scaled_by_airmass(self, target_device_idx, xp):
        """
        Test that pupil_distances are correctly computed as heights * airmass
        """
        pixel_pupil = 160
        zenith = 30.0  # degrees
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil, pixel_pitch=0.05, zenithAngleInDeg=zenith, time_step=1
        )
        heights = [1000.0, 5000.0, 12000.0]
        airmass = 1.0 / np.cos(np.radians(zenith))
        atmo = AtmoEvolution(simul_params,
                             L0=23,
                             data_dir=self.data_dir,
                             heights=heights,
                             Cn2=[1/3, 1/3, 1/3],
                             fov=120.0,
                             target_device_idx=target_device_idx)
        expected = cpuArray(heights) * airmass
        np.testing.assert_allclose(atmo.pupil_distances, expected, rtol=1e-8)

    @unittest.skipIf(specula.cp is None, 'GPU not available')
    def test_cuda_graph_matches_cpu(self):
        """
        Test that on GPU the evolution is captured in a CUDA graph and gives the
        same layers as the CPU implementation, including rotations and screen cycling,
        for AtmoEvolution and for both layer lists of AtmoEvolutionUpDown
        """
        simul_params = SimulParams(pixel_pupil=32, pixel_pitch=0.05, time_step=0.01)
        classes = {AtmoEvolution: dict(extra_delta_time=0.013),
                   AtmoEvolutionUpDown: dict(extra_delta_time_down=0.013,
                                             extra_delta_time_up=[0.0, 0.03])}
        for cls, kwargs in classes.items():
            layers = {}
            for target_device_idx in [-1, 0]:
                seeing = WaveGenerator(constant=0.8, amp=0.3, freq=5.0,
                                       target_device_idx=target_device_idx)
                wind_speed = WaveGenerator(constant=[25.5, 30.0], amp=[5.0, 5.0], freq=[3.0, 3.0],
                                           target_device_idx=target_device_idx)
                wind_direction = WaveGenerator(constant=[90, -212.7], amp=[20.0, 20.0],
                                               freq=[2.0, 2.0], target_device_idx=target_device_idx)
                atmo = cls(simul_params, L0=23, data_dir=self.data_dir,
                           heights=[0, 10000], Cn2=[0.5, 0.5], fov=60.0,
                           pixel_phasescreens=256, **kwargs,
                           target_device_idx=target_device_idx, precision=0)
                atmo.inputs['seeing'].set(seeing.output)
                atmo.inputs['wind_speed'].set(wind_speed.output)
                atmo.inputs['wind_direction'].set(wind_direction.output)

                loop = LoopControl()
                for obj in [seeing, wind_speed, wind_direction]:
                    loop.add(obj, idx=0)
                loop.add(atmo, idx=1)
                loop.start(run_time=0.4, dt=simul_params.time_step)
                layers[target_device_idx] = []
                for _ in range(40):
                    loop.iter()
                    layers[target_device_idx] += [cpuArray(l.phaseInNm).copy()
                                                  for layer_list in atmo.layer_lists
                                                  for l in layer_list]
                    # The scale coefficient must follow the current (time-varying) seeing
                    expected_scale = cpuArray(seeing.output.value)[0]**(5/6) * atmo.seeing_scale_factor
                    np.testing.assert_allclose(cpuArray(atmo.scale_coef), expected_scale, rtol=1e-10)
                    # Rotation matrices must follow the current (time-varying) wind direction
                    theta = np.radians(cpuArray(wind_direction.output.value))
                    np.testing.assert_allclose(cpuArray(atmo.rot_matrix[:, 0, 0]), np.cos(theta), atol=1e-10)
                    np.testing.assert_allclose(cpuArray(atmo.rot_matrix[:, 0, 1]), np.sin(theta), atol=1e-10)
                assert (atmo.cuda_graph is not None) == (target_device_idx >= 0)
                assert len(atmo.layer_lists) == (2 if cls is AtmoEvolutionUpDown else 1)
                if cls is AtmoEvolutionUpDown:
                    # Different extra delta times: the up list is at a different position
                    assert not np.allclose(cpuArray(atmo.win_matrix[0, :, 1, 2]),
                                           cpuArray(atmo.win_matrix[1, :, 1, 2]))

            for gpu_layer, cpu_layer in zip(layers[0], layers[-1]):
                np.testing.assert_allclose(gpu_layer, cpu_layer, rtol=1e-10, atol=1e-8)

    @cpu_and_gpu
    def test_shared_stream(self, target_device_idx, xp):
        """
        Test that the evolution can run in the device stream shared with
        other objects (allow_parallel=False), with the same layers as in its own stream
        """
        simul_params = SimulParams(pixel_pupil=32, pixel_pitch=0.05, time_step=0.01)
        layers = {}
        for allow_parallel in [True, False]:
            seeing = WaveGenerator(constant=0.8, amp=0.3, freq=5.0,
                                   target_device_idx=target_device_idx)
            wind_speed = WaveGenerator(constant=[25.5, 30.0], amp=[5.0, 5.0], freq=[3.0, 3.0],
                                       target_device_idx=target_device_idx)
            wind_direction = WaveGenerator(constant=[90, -212.7], amp=[20.0, 20.0],
                                           freq=[2.0, 2.0], target_device_idx=target_device_idx)
            atmo = AtmoEvolution(simul_params, L0=23, data_dir=self.data_dir,
                                 heights=[0, 10000], Cn2=[0.5, 0.5], fov=60.0,
                                 pixel_phasescreens=256, target_device_idx=target_device_idx,
                                 precision=0)
            atmo.inputs['seeing'].set(seeing.output)
            atmo.inputs['wind_speed'].set(wind_speed.output)
            atmo.inputs['wind_direction'].set(wind_direction.output)

            loop = LoopControl()
            for obj in [seeing, wind_speed, wind_direction]:
                loop.add(obj, idx=0)
            loop.add(atmo, idx=1)
            loop.start(run_time=0.2, dt=simul_params.time_step)
            atmo.build_stream(allow_parallel=allow_parallel)
            assert (atmo.cuda_graph is not None) == (target_device_idx >= 0)
            if target_device_idx >= 0:
                assert (atmo.stream is BaseProcessingObj.device_stream(0)) == (not allow_parallel)
            layers[allow_parallel] = []
            for _ in range(20):
                loop.iter()
                layers[allow_parallel] += [cpuArray(l.phaseInNm).copy() for l in atmo.layer_list]

        # Not bit-identical: if the phase screens are not cached, the first object
        # generates them and the second one loads them from file
        for shared, own in zip(layers[False], layers[True]):
            np.testing.assert_allclose(shared, own, rtol=1e-12, atol=1e-9)

    @cpu_and_gpu
    def test_zero_and_negative_seeing(self, target_device_idx, xp):
        """Test that seeing <= 0 gives zero layers, without NaNs"""
        simul_params = SimulParams(pixel_pupil=32, pixel_pitch=0.05, time_step=0.01)
        for seeing_value in [0.0, -1.0]:
            seeing = WaveGenerator(constant=seeing_value, target_device_idx=target_device_idx)
            wind_speed = WaveGenerator(constant=[10.0], target_device_idx=target_device_idx)
            wind_direction = WaveGenerator(constant=[33.3], target_device_idx=target_device_idx)
            atmo = AtmoEvolution(simul_params, L0=23, data_dir=self.data_dir, heights=[0],
                                 Cn2=[1.0], pixel_phasescreens=256,
                                 target_device_idx=target_device_idx)
            atmo.inputs['seeing'].set(seeing.output)
            atmo.inputs['wind_speed'].set(wind_speed.output)
            atmo.inputs['wind_direction'].set(wind_direction.output)

            loop = LoopControl()
            for obj in [seeing, wind_speed, wind_direction]:
                loop.add(obj, idx=0)
            loop.add(atmo, idx=1)
            loop.start(run_time=0.02, dt=simul_params.time_step)
            loop.iter()
            np.testing.assert_array_equal(cpuArray(atmo.layer_list[0].phaseInNm), 0)

    @cpu_and_gpu
    def test_fov_in_m(self, target_device_idx, xp):
        """Test that fov_in_m sets the size of all layers, ignoring fov, and that
        the evolution runs with it (captured in a CUDA graph on GPU)"""
        simul_params = SimulParams(pixel_pupil=32, pixel_pitch=0.05, time_step=0.01)
        fov_in_m = 4.03
        expected_size = int(fov_in_m / simul_params.pixel_pitch / 2.0) * 2
        kwargs = dict(L0=23, data_dir=self.data_dir, heights=[0, 10000], Cn2=[0.5, 0.5],
                      fov=60.0, pixel_phasescreens=256, target_device_idx=target_device_idx)

        # With fov only, the layer sizes depend on the height
        atmo_fov = AtmoEvolution(simul_params, **kwargs)
        assert len(set(atmo_fov.pixel_layer)) == 2

        atmo = AtmoEvolution(simul_params, fov_in_m=fov_in_m, **kwargs)
        np.testing.assert_array_equal(atmo.pixel_layer, expected_size)
        for layer in atmo.layer_list:
            assert layer.phaseInNm.shape == (expected_size, expected_size)

        seeing = WaveGenerator(constant=0.8, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[25.5, 30.0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[90, 33.3], target_device_idx=target_device_idx)
        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        loop = LoopControl()
        for obj in [seeing, wind_speed, wind_direction]:
            loop.add(obj, idx=0)
        loop.add(atmo, idx=1)
        loop.start(run_time=0.03, dt=simul_params.time_step)
        for _ in range(3):
            loop.iter()
        assert (atmo.cuda_graph is not None) == (target_device_idx >= 0)
        for layer in atmo.layer_list:
            phase = cpuArray(layer.phaseInNm)
            assert np.all(np.isfinite(phase)) and np.any(phase != 0)

    @cpu_and_gpu
    def test_matches_original_algorithm(self, target_device_idx, xp):
        """Regression test of positions, seeing scale and rotation convention: the layers
        must match the original algorithm (window slicing with linear interpolation,
        rot90() and ndimage rotate()) for fractional, negative, multiple of 90 degrees
        and larger than 360 degrees wind directions, with layers of different sizes"""
        from scipy.ndimage import rotate
        simul_params = SimulParams(pixel_pupil=32, pixel_pitch=0.05, time_step=0.01)
        directions = [0.0, 90.0, 33.3, -212.7, 405.5]
        speeds = [10.3, 7.1, 12.9, 5.55, 9.0]
        seeing_value = 0.8
        n = len(directions)
        atmo = AtmoEvolution(simul_params, L0=23, data_dir=self.data_dir,
                             heights=[2000.0 * i for i in range(n)], Cn2=[1.0 / n] * n,
                             fov=60.0, pixel_phasescreens=256,
                             target_device_idx=target_device_idx, precision=0)
        assert len(set(atmo.pixel_layer)) > 1
        seeing = WaveGenerator(constant=seeing_value, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=speeds, target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=directions, target_device_idx=target_device_idx)
        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        loop = LoopControl()
        for obj in [seeing, wind_speed, wind_direction]:
            loop.add(obj, idx=0)
        loop.add(atmo, idx=1)
        n_steps = 5
        loop.start(run_time=simul_params.time_step * n_steps, dt=simul_params.time_step)
        for _ in range(n_steps):
            loop.iter()

        # Original seeing scale coefficient, no zenith angle
        r0 = 0.9759 * 0.5 / (seeing_value * 4.848)
        scale = (simul_params.pixel_pitch / r0) ** (5. / 6.)
        for ii, layer in enumerate(atmo.layer_list):
            # The first step has zero delta time
            position = speeds[ii] * simul_params.time_step * (n_steps - 1) / simul_params.pixel_pitch
            pos = int(np.floor(position))
            rem = position - pos
            size = layer.phaseInNm.shape[0]
            screen = cpuArray(atmo.phasescreens[ii])
            expected = (1.0 - rem) * screen[0:size, pos:pos + size] \
                       + rem * screen[0:size, pos + 1:pos + size + 1]
            wdf, wdi = np.modf(directions[ii] / 90.0)
            expected = np.rot90(expected, int(wdi))
            if wdf != 0:
                expected = rotate(expected, wdf * 90, reshape=False, order=1)
            expected *= scale
            np.testing.assert_allclose(cpuArray(layer.phaseInNm), expected, rtol=1e-10,
                                       atol=1e-10 * np.abs(expected).max())

    @cpu_and_gpu
    def test_reallocated_input_raises_with_cuda_graph(self, target_device_idx, xp):
        """With a CUDA graph, an input reallocated by its producer (here BaseSlicer,
        which rebinds its output value at each step) raises an error, instead of
        being silently ignored. Without a graph (CPU) it works."""
        simul_params = SimulParams(pixel_pupil=32, pixel_pitch=0.05, time_step=0.01)
        seeing = WaveGenerator(constant=0.8, target_device_idx=target_device_idx)
        all_speeds = WaveGenerator(constant=[25.5, 30.0, 12.0], target_device_idx=target_device_idx)
        wind_speed = BaseSlicer(indices=[0, 1], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[90, 33.3], target_device_idx=target_device_idx)
        wind_speed.inputs['in_value'].set(all_speeds.output)
        atmo = AtmoEvolution(simul_params, L0=23, data_dir=self.data_dir, heights=[0, 10000],
                             Cn2=[0.5, 0.5], pixel_phasescreens=256,
                             target_device_idx=target_device_idx)
        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_speed'].set(wind_speed.outputs['out_value'])
        atmo.inputs['wind_direction'].set(wind_direction.output)

        loop = LoopControl()
        for obj in [seeing, all_speeds, wind_direction]:
            loop.add(obj, idx=0)
        loop.add(wind_speed, idx=1)
        loop.add(atmo, idx=2)
        loop.start(run_time=0.03, dt=simul_params.time_step)
        if atmo.cuda_graph:
            with self.assertRaisesRegex(RuntimeError, 'wind_speed has been reallocated'):
                for _ in range(3):
                    loop.iter()
        else:
            for _ in range(3):
                loop.iter()
            self.assertTrue(np.any(cpuArray(atmo.layer_list[0].phaseInNm) != 0))
        self.assertEqual(atmo.cuda_graph is not None, target_device_idx >= 0)
