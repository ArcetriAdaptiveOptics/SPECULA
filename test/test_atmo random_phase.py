
import os
import specula
specula.init(0)  # Default target device

import unittest

from specula.data_objects.source import Source
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.atmo_random_phase import AtmoRandomPhase

from test.specula_testlib import cpu_and_gpu


class TestAtmo(unittest.TestCase):

    @cpu_and_gpu
    def test_output_ef_have_the_correct_names(self, target_device_idx, xp):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        lgs1_source = Source( polar_coordinates=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)

        tmpParams = SimulParams(root_dir = '', pixel_pupil=160, pixel_pitch=0.05)
        atmo = AtmoRandomPhase(tmpParams, 
                            L0=23,  # [m] Outer scale
                            data_dir=data_dir,
                            source_dict = {'on_axis_source': on_axis_source,
                                            'lgs1_source': lgs1_source},
                            target_device_idx=target_device_idx)
        
        assert len(atmo.outputs) == 2
        assert 'out_on_axis_source_ef' in atmo.outputs
        assert 'out_lgs1_source_ef' in atmo.outputs
        