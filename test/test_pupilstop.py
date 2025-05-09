

import specula
specula.init(0)  # Default target device

import os
import unittest
import numpy as np

from specula import cpuArray
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu

class TestPixels(unittest.TestCase):
   
    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), 'data', 'pupilstop.fits')

    def tearDown(self):
        try:
            pass
            #os.unlink(self.filename)
        except:
            pass

    @cpu_and_gpu
    def test_pupilstop_save_restore_roundtrip(self, target_device_idx, xp):
        
        pixel_pupil = 3
        pixel_pitch = 0.05
        mask = np.arange(9).reshape((3, 3))
        tempParams = SimulParams(pixel_pupil, pixel_pitch)
        stop1 = Pupilstop(tempParams, input_mask=mask, target_device_idx=target_device_idx)
        stop1.save(self.filename)

        stop2 = Pupilstop.restore(self.filename)

        assert stop1.pixel_pupil == stop2.pixel_pupil
        assert stop1.pixel_pitch == stop2.pixel_pitch
        np.testing.assert_array_equal(cpuArray(stop1.A), cpuArray(stop2.A))
        assert stop2.phaseInNm.sum() == 0

       
