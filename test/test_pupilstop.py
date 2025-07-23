
import specula
specula.init(0)  # Default target device

import tempfile
import os
import gc
import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu

class TestPupilstop(unittest.TestCase):

    @cpu_and_gpu
    def test_input_mask(self, target_device_idx, xp):
        pixel_pupil = 20
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        mask_diam = 18.0
        obs_diam = 0.1
        shiftXYinPixel = (0.0, 0.0)
        rotInDeg = 0.0
        magnification = 1.0

        # first create a Pupilstop object
        pupilstop0 = Pupilstop(simul_params,
                              mask_diam=mask_diam,
                              obs_diam=obs_diam,
                              shiftXYinPixel=shiftXYinPixel,
                              rotInDeg=rotInDeg,
                              magnification=magnification,
                              target_device_idx=target_device_idx)

        # make a second Pupilstop object from the Amplitude of the first one
        pupilstop1 = Pupilstop(simul_params,
                              input_mask=pupilstop0.A,
                              target_device_idx=target_device_idx)

        # Check that the two objects have the same data
        assert np.allclose(cpuArray(pupilstop0.A), cpuArray(pupilstop1.A))
        
    @cpu_and_gpu
    def test_save_and_restore(self, target_device_idx, xp):
        pixel_pupil = 20
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        mask_diam = 18.0
        obs_diam = 0.1
        shiftXYinPixel = (0.0, 0.0)
        rotInDeg = 0.0
        magnification = 1.0

        # first create a Pupilstop object
        pupilstop = Pupilstop(simul_params,
                              mask_diam=mask_diam,
                              obs_diam=obs_diam,
                              shiftXYinPixel=shiftXYinPixel,
                              rotInDeg=rotInDeg,
                              magnification=magnification,
                              target_device_idx=target_device_idx)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "pupilstop_test.fits")
            pupilstop.save(filename)

            # Restore from file
            pupilstop2 = Pupilstop.restore(filename, target_device_idx=target_device_idx)

            # Check that the restored object has the data as expected
            assert np.allclose(cpuArray(pupilstop.A), cpuArray(pupilstop2.A))
            assert pupilstop.pixel_pitch == pupilstop2.pixel_pitch
            assert pupilstop.magnification == pupilstop2.magnification

            # Force cleanup for Windows
            del pupilstop2
            gc.collect()

    @cpu_and_gpu
    def test_PASSATA_pupilstop_file(self, target_device_idx, xp):
        '''Test that old pupilstop files from PASSATA are loaded correctly'''

        filename = os.path.join(os.path.dirname(__file__), 'data', 'PASSATA_pupilstop_64pix.fits')

        # From custom PASSATA method
        pupilstop = Pupilstop.restore_from_passata(filename, target_device_idx=target_device_idx)
        assert pupilstop.A.shape == (64,64)
        self.assertAlmostEqual(pupilstop.pixel_pitch, 0.01)

        # From generic method - both must work
        pupilstop = Pupilstop.restore(filename, target_device_idx=target_device_idx)
        assert pupilstop.A.shape == (64,64)
        self.assertAlmostEqual(pupilstop.pixel_pitch, 0.01)

    @cpu_and_gpu
    def test_wrong_file_fails(self, target_device_idx, xp):

        filename = os.path.join(os.path.dirname(__file__), 'data', 'ref_phase.fits')

        with self.assertRaises(ValueError):
            pupilstop = Pupilstop.restore_from_passata(filename, target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            Pupilstop.restore(filename, target_device_idx=target_device_idx)
