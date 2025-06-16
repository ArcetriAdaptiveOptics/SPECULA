

import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest

from specula.data_objects.slopes import Slopes
from specula.base_value import BaseValue
from specula.processing_objects.im_rec_calibrator import ImRecCalibrator

from test.specula_testlib import cpu_and_gpu

class TestImRecCalibrator(unittest.TestCase):

    def test_existing_im_file_is_detected(self):

        data_dir = tempfile.gettempdir()
        im_filename = 'test_im.fits'
        im_path = os.path.join(data_dir, im_filename)
        open(im_path, 'a').close()
        
        slopes = Slopes(2)
        cmd = BaseValue(value=2)
        calibrator = ImRecCalibrator(nmodes=10, data_dir=data_dir, rec_tag='x', im_tag='test_im')
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)
        
        with self.assertRaises(FileExistsError):
            calibrator.setup()
        
    def test_existing_rec_file_is_detected(self):

        data_dir = tempfile.gettempdir()
        rec_filename = 'test_rec.fits'
        rec_path = os.path.join(data_dir, rec_filename)
        open(rec_path, 'a').close()
        
        slopes = Slopes(2)
        cmd = BaseValue(value=2)
        calibrator = ImRecCalibrator(nmodes=10, data_dir=data_dir, rec_tag='test_rec')
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)
        
        with self.assertRaises(FileExistsError):
            calibrator.setup()

    def test_existing_im_file_is_not_detected_if_not_requested(self):

        data_dir = tempfile.gettempdir()
        im_filename = 'test_im.fits'
        rec_filename = 'test_rec.fits'
        im_path = os.path.join(data_dir, im_filename)
        rec_path = os.path.join(data_dir, rec_filename)
        open(im_path, 'a').close()
        if os.path.exists(rec_path):
            os.unlink(rec_path)

        slopes = Slopes(2)
        cmd = BaseValue(value=2)
        calibrator = ImRecCalibrator(nmodes=10, data_dir=data_dir, rec_tag='test_rec')
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)

        # Does not raise        
        calibrator.setup()
    
    @cpu_and_gpu
    def test_triggered_by_slopes_only(self, target_device_idx, xp):

        data_dir = tempfile.gettempdir()
        rec_filename = 'test_rec.fits'
        rec_path = os.path.join(data_dir, rec_filename)
        if os.path.exists(rec_path):
            os.unlink(rec_path)
        
        slopes = Slopes(2, target_device_idx=target_device_idx)
        cmd = BaseValue(value=xp.zeros(2), target_device_idx=target_device_idx)
        calibrator = ImRecCalibrator(nmodes=10, data_dir=data_dir, rec_tag='test_rec')
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)
        calibrator.setup()
        
        slopes.generation_time = 1
        cmd.generation_time = 1

        calibrator.check_ready(t=1)
        calibrator.trigger()
        calibrator.post_trigger()
        
        assert calibrator.outputs['out_im'][0].generation_time == 1

        # Do not advance slopes.generation_time
        slopes.generation_time = 1
        cmd.generation_time = 2

        calibrator.check_ready(t=2)
        calibrator.trigger()
        calibrator.post_trigger()
        
        # Check that trigger was not executed
        assert calibrator.outputs['out_im'][0].generation_time == 1

        # Advance both
        slopes.generation_time = 3
        cmd.generation_time = 3

        calibrator.check_ready(t=3)
        calibrator.trigger()
        calibrator.post_trigger()
        
        # Check that trigger was executed
        assert calibrator.outputs['out_im'][0].generation_time == 3


