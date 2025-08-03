
import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray

from specula.data_objects.source import Source
from specula.processing_objects.func_generator import FuncGenerator
from specula.processing_objects.atmo_infinite_evolution import AtmoInfiniteEvolution
from specula.data_objects.layer import Layer
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu


class TestAtmoInfiniteEvolution(unittest.TestCase):

    @cpu_and_gpu
    def test_infinite_evolution_layer_size(self, target_device_idx, xp):
        '''
        Test that the output layer size is the correct one
        '''
        pixel_pupil = 160
        simulParams = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=0.05, time_step=1)
    
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = FuncGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = FuncGenerator(constant=[5.5, 2.5], target_device_idx=target_device_idx)
        wind_direction = FuncGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        lgs1_source = Source( polar_coordinates=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)

        atmo = AtmoInfiniteEvolution(simulParams,
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
