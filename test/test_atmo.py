
import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray

from specula.data_objects.source import Source
from specula.processing_objects.func_generator import FuncGenerator
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.layer import Layer

from test.specula_testlib import cpu_and_gpu


class TestAtmo(unittest.TestCase):

    @cpu_and_gpu
    def test_atmo(self, target_device_idx, xp):
        
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = FuncGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = FuncGenerator(constant=5.5, target_device_idx=target_device_idx)
        wind_direction = FuncGenerator(constant=0, target_device_idx=target_device_idx)

        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        lgs1_source = Source( polar_coordinates=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)

        atmo = AtmoEvolution(L0=23,  # [m] Outer scale
                             pixel_pupil=160,
                             pixel_pitch=0.05,
                             data_dir=data_dir,
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)


        prop = AtmoPropagation(pixel_pupil=160,
                               pixel_pitch=0.05,
                               source_dict = {'on_axis_source': on_axis_source,
                                               'lgs1_source': lgs1_source},
                               target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])
        prop.inputs['common_layer_list'].set([])

        for obj in [seeing, wind_speed, wind_direction, atmo, prop]:
            obj.setup(1, 1)
        
        for obj in [seeing, wind_speed, wind_direction, atmo, prop]:
            obj.check_ready(1)
       
        for obj in [seeing, wind_speed, wind_direction, atmo, prop]:
            obj.trigger()

        for obj in [seeing, wind_speed, wind_direction, atmo, prop]:
            obj.post_trigger()
            
        ef_onaxis = cpuArray(prop.outputs['out_on_axis_source_ef'])
        ef_offaxis = cpuArray(prop.outputs['out_lgs1_source_ef'])

    @cpu_and_gpu
    def test_that_wrong_Cn2_total_is_detected(self, target_device_idx, xp):

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        lgs1_source = Source( polar_coordinates=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)

        with self.assertRaises(ValueError):
            atmo = AtmoEvolution(L0=23,  # [m] Outer scale
                                pixel_pupil=160,
                                pixel_pitch=0.05,
                                data_dir=data_dir,
                                heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                                Cn2 = [0.2, 0.2], # Cn2 weights (total must be eq 1)
                                fov = 120.0,
                                target_device_idx=target_device_idx)

        # Total is 1, no exception raised.
        atmo = AtmoEvolution(L0=23,  # [m] Outer scale
                            pixel_pupil=160,
                            pixel_pitch=0.05,
                            data_dir=data_dir,
                            heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                            Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                            fov = 120.0,
                            target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_layer_list_type_length_and_element_types(self, target_device_idx, xp):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        lgs1_source = Source( polar_coordinates=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)

        atmo = AtmoEvolution(L0=23,  # [m] Outer scale
                            pixel_pupil=160,
                            pixel_pitch=0.05,
                            data_dir=data_dir,
                            heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                            Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                            fov = 120.0,
                            target_device_idx=target_device_idx)
            
        assert isinstance(atmo.outputs['layer_list'], list)
        assert len(atmo.outputs['layer_list']) == 2
        
        for layer in atmo.outputs['layer_list']:
            assert isinstance(layer, Layer)
