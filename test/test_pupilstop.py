import specula
specula.init(0)  # Default target device

import tempfile
import os
import gc
import unittest
import pytest

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

    # This decorator suppress the warning in pytest output,
    # but the "with pytest.warns(...)" instruction below still checks
    # that it is raised.
    @pytest.mark.filterwarnings('ignore:.*Detected PASSATA pupilstop file.*:RuntimeWarning')
    @cpu_and_gpu
    def test_PASSATA_pupilstop_file(self, target_device_idx, xp):
        '''Test that old pupilstop files from PASSATA are loaded correctly'''

        filename = os.path.join(os.path.dirname(__file__), 'data', 'PASSATA_pupilstop_64pix.fits')

        # From custom PASSATA method
        with pytest.warns(RuntimeWarning):
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
    
    @cpu_and_gpu
    def test_set_value(self, target_device_idx, xp):
        pixel_pupil = 20
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx)

        # Create a new amplitude mask
        new_mask = xp.ones((pixel_pupil, pixel_pupil), dtype=pupilstop.dtype)

        # Set the new mask
        pupilstop.set_value(new_mask)

        # Check that the mask was set correctly
        np.testing.assert_array_equal(cpuArray(pupilstop.A), cpuArray(new_mask))
        
    @cpu_and_gpu
    def test_set_value_shape_mismatch(self, target_device_idx, xp):
        pixel_pupil = 20
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx)

        # Create a new mask with a different shape
        new_mask = xp.ones((pixel_pupil + 1, pixel_pupil), dtype=pupilstop.dtype)

        # Expect an assertion error due to shape mismatch
        with self.assertRaises(AssertionError):
            pupilstop.set_value(new_mask)
            
    @cpu_and_gpu
    def test_get_value(self, target_device_idx, xp):
        pixel_pupil = 20
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx)

        # Create a new amplitude mask
        new_mask = xp.ones((pixel_pupil, pixel_pupil), dtype=pupilstop.dtype)
        pupilstop.set_value(new_mask)

        # Get the value and check it matches
        retrieved_mask = pupilstop.get_value()
        np.testing.assert_array_equal(cpuArray(retrieved_mask), cpuArray(new_mask))

    @cpu_and_gpu
    def test_float(self, target_device_idx, xp):
        '''Test that precision=1 results in a single-precision pupilstop'''
        pixel_pupil = 10
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx, precision=1)
        assert pupilstop.A.dtype == np.float32

    @cpu_and_gpu
    def test_double(self, target_device_idx, xp):
        '''Test that precision=0 results in a double-precision ef'''
        pixel_pupil = 10
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx, precision=0)
        assert pupilstop.A.dtype == np.float64

    @cpu_and_gpu
    def test_float_from_other_types(self, target_device_idx, xp):
        '''Test that precision=1 results in a single-precision pupilstop,
        even when a mask of a different dtype is set'''
        pixel_pupil = 10
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx, precision=1)
        for dtype in [xp.float64, int, bool]:
            new_mask = xp.ones((pixel_pupil, pixel_pupil), dtype=dtype)
            pupilstop.set_value(new_mask)
            assert pupilstop.A.dtype == np.float32


    @cpu_and_gpu
    def test_double_from_other_types(self, target_device_idx, xp):
        '''Test that precision=1 results in a double-precision pupilstop,
        even when a mask of a different dtype is set'''
        pixel_pupil = 10
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx, precision=0)
        for dtype in [xp.float32, int, bool]:
            new_mask = xp.ones((pixel_pupil, pixel_pupil), dtype=dtype)
            pupilstop.set_value(new_mask)
            assert pupilstop.A.dtype == np.float64
