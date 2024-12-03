
import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import cp, np
from specula import cpuArray

from specula.data_objects.source import Source
from specula.processing_objects.func_generator import FuncGenerator
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation


class TestAtmo(unittest.TestCase):

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_atmo_gpu(self):
        self._test_atmo(target_device_idx=0, xp=cp)

    def test_atmo_cpu(self):
        self._test_atmo(target_device_idx=-1, xp=np)
        
    def _test_atmo(self, target_device_idx, xp):
        
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = FuncGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = FuncGenerator(constant=5.5, target_device_idx=target_device_idx)
        wind_direction = FuncGenerator(constant=0, target_device_idx=target_device_idx)

        on_axis_source = Source(polar_coordinate=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        lgs1_source = Source( polar_coordinate=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)

        atmo = AtmoEvolution(L0=23,  # [m] Outer scale
                             pixel_pupil=160,
                             pixel_pitch=0.05,
                             data_dir=data_dir,
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             source_dict = {'on_axis_source': on_axis_source,
                                            'lgs1_source': lgs1_source},
                             target_device_idx=target_device_idx)


        prop = AtmoPropagation(pixel_pupil=160,
                               pixel_pitch=0.05,
                               source_dict = {'on_axis_source': on_axis_source,
                                               'lgs1_source': lgs1_source},
                               target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop.inputs['layer_list'].set(atmo.outputs['layer_list'])

        for obj in [seeing, wind_speed, wind_direction, atmo, prop]:
            obj.run_check(1)
        
        for obj in [seeing, wind_speed, wind_direction, atmo, prop]:
            obj.check_ready(1)
       
        for obj in [seeing, wind_speed, wind_direction, atmo, prop]:
            obj.trigger()

        for obj in [seeing, wind_speed, wind_direction, atmo, prop]:
            obj.post_trigger()
            
        ef_onaxis = cpuArray(prop.outputs['out_on_axis_source_ef'])
        ef_offaxis = cpuArray(prop.outputs['out_lgs1_source_ef'])
        

if __name__ == '__main__':
    unittest.main()