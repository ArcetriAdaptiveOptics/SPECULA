
import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray
from specula import np

from specula.data_objects.source import Source
from specula.base_time_obj import BaseTimeObj
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.layer import Layer
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu


class TestAtmoEvolution(unittest.TestCase):

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

        for objlist in [[seeing, wind_speed, wind_direction], [atmo], [prop]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()

        ef_onaxis = cpuArray(prop.outputs['out_on_axis_source_ef'])
        ef_offaxis = cpuArray(prop.outputs['out_lgs1_source_ef'])

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

        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()

        id_a1 = id(atmo.outputs['layer_list'][0].field)
        id_b1 = id(atmo.outputs['layer_list'][1].field)

        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.check_ready(2)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()

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
    def test_reverse_atmo_layer_list(self, target_device_idx, xp):
        '''Test that reverse_atmo_layer_list reverses the order of atmo_layer_list in AtmoPropagation'''
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)
        seeing = WaveGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[5.5, 2.5, 10.0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 10, -10], target_device_idx=target_device_idx)

        atmo = AtmoEvolution(simulParams,
                             L0=23,
                             data_dir=self.data_dir,
                             heights=[30.0, 5000.0, 20000.0],
                             Cn2=[0.5, 0.25, 0.25],
                             fov=0.0,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        atmo.setup()

        original_layers = list(atmo.outputs['layer_list'])

        prop = AtmoPropagation(simulParams,
                               source_dict={'src': Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)},
                               reverse_atmo_layer_list=True,
                               target_device_idx=target_device_idx)
        prop.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])
        prop.setup()

        reversed_layers = prop.atmo_layer_list

        assert reversed_layers != original_layers
        assert reversed_layers == original_layers[::-1]

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
        atmo = AtmoEvolution(simul_params,
                             L0=23,
                             data_dir=self.data_dir,
                             heights=heights,
                             Cn2=[1/3, 1/3, 1/3],
                             fov=120.0,
                             target_device_idx=target_device_idx)
        expected = cpuArray(heights) * airmass
        np.testing.assert_allclose(atmo.pupil_distances, expected, rtol=1e-8)