import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
import shutil

from specula.base_value import BaseValue
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.slopes import Slopes
from specula.data_objects.source import Source
from specula.data_objects.intmat import Intmat
from specula.data_objects.subap_data import SubapData
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.dm import DM
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.im_calibrator import ImCalibrator
from specula.processing_objects.rec_calibrator import RecCalibrator

from test.specula_testlib import cpu_and_gpu

class TestImRecCalibrator(unittest.TestCase):

    def setUp(self):
        """Create unique temporary directory for each test"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_existing_im_file_is_detected(self):
        """Test that ImCalibrator detects existing IM files"""
        im_tag = 'test_im'
        im_filename = f'{im_tag}.fits'
        im_path = os.path.join(self.test_dir, im_filename)

        # Create empty file
        with open(im_path, 'w') as f:
            f.write('')

        with self.assertRaises(FileExistsError):
            _ = ImCalibrator(nmodes=10, data_dir=self.test_dir, im_tag=im_tag)
 
    def test_existing_im_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing files"""
        im_tag = 'test_im_overwrite'
        im_filename = f'{im_tag}.fits'
        im_path = os.path.join(self.test_dir, im_filename)

        # Create empty file
        with open(im_path, 'w') as f:
            f.write('')

        # Should not raise
        _ = ImCalibrator(nmodes=10, data_dir=self.test_dir, im_tag=im_tag, overwrite=True)

    def test_existing_rec_file_is_detected(self):
        """Test that RecCalibrator detects existing REC files"""
        rec_tag = 'test_rec'
        rec_filename = f'{rec_tag}.fits'
        rec_path = os.path.join(self.test_dir, rec_filename)

        # Create empty file
        with open(rec_path, 'w') as f:
            f.write('')

        with self.assertRaises(FileExistsError):
            _ = RecCalibrator(nmodes=2, data_dir=self.test_dir, rec_tag=rec_tag)
  
    @cpu_and_gpu
    def test_triggered_by_slopes_only(self, target_device_idx, xp):
        """Test that calibrator only triggers when slopes are updated"""
        im_tag = 'test_im_trigger'

        slopes = Slopes(2, target_device_idx=target_device_idx)
        cmd = BaseValue(value=xp.zeros(2), target_device_idx=target_device_idx)
        calibrator = ImCalibrator(nmodes=10, data_dir=self.test_dir, im_tag=im_tag, overwrite=True)
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)
        calibrator.setup()

        slopes.generation_time = 1
        cmd.generation_time = 1

        calibrator.check_ready(t=1)
        calibrator.trigger()
        calibrator.post_trigger()

        # Check that output was created and updated
        self.assertEqual(calibrator.outputs['out_intmat'].generation_time, 1)

        # Do not advance slopes.generation_time
        slopes.generation_time = 1
        cmd.generation_time = 2

        calibrator.check_ready(t=2)
        calibrator.trigger()
        calibrator.post_trigger()

        # Check that trigger was not executed (output time unchanged)
        self.assertEqual(calibrator.outputs['out_intmat'].generation_time, 1)

        # Advance both
        slopes.generation_time = 3
        cmd.generation_time = 3

        calibrator.check_ready(t=3)
        calibrator.trigger()
        calibrator.post_trigger()

        # Check that trigger was executed
        self.assertEqual(calibrator.outputs['out_intmat'].generation_time, 3)

    def test_existing_rec_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing files"""
        rec_tag = 'test_rec_overwrite'
        rec_filename = f'{rec_tag}.fits'
        rec_path = os.path.join(self.test_dir, rec_filename)

        # Create empty file
        with open(rec_path, 'w') as f:
            f.write('')

        intmat = Intmat(intmat=specula.np.array([[1, 2], [3, 4]]))
        calibrator = RecCalibrator(nmodes=2, data_dir=self.test_dir, rec_tag=rec_tag, overwrite=True)
        calibrator.inputs['in_intmat'].set(intmat)

        # Should not raise
        calibrator.setup()

    @cpu_and_gpu
    def test_reconstructor_generation(self, target_device_idx, xp):
        """Test that reconstructor is generated from interaction matrix"""
        rec_tag = 'test_rec_gen'

        # Create mock interaction matrix (slopes x modes)
        n_slopes = 6
        n_modes = 3
        mock_im = xp.random.random((n_slopes, n_modes)).astype(xp.float32)
        intmat = Intmat(intmat=mock_im, target_device_idx=target_device_idx)

        # Set generation time BEFORE setup
        intmat.generation_time = 1

        calibrator = RecCalibrator(nmodes=n_modes, data_dir=self.test_dir, rec_tag=rec_tag, overwrite=True)
        calibrator.inputs['in_intmat'].set(intmat)
        calibrator.setup()

        # Check ready and trigger
        calibrator.check_ready(t=1)
        calibrator.trigger()
        calibrator.finalize()

        # Check that file was created
        rec_path = os.path.join(self.test_dir, f'{rec_tag}.fits')
        self.assertTrue(os.path.exists(rec_path))

    @cpu_and_gpu
    def test_automatic_im_tag_generation(self, target_device_idx, xp):
        """Test that ImCalibrator generates automatic im_tag when not specified"""

        # Create SimulParams (needed for DM and SH)
        simul_params = SimulParams(
            pixel_pupil=64,
            pixel_pitch=0.1
        )

        # create a Pupilstop
        pupilstop = Pupilstop(simul_params,
                              mask_diam=0.9,
                              obs_diam=0.1,
                              target_device_idx=target_device_idx)

        # Create Source
        source = Source(
            polar_coordinates=[10.0, 0.0],
            magnitude=5,
            wavelengthInNm=600,
            target_device_idx=target_device_idx
        )

        # Create DM with proper initialization
        dm = DM(
            simul_params=simul_params,
            type_str='zernike',
            nmodes=40,
            obsratio=0.1,
            height=0,
            target_device_idx=target_device_idx
        )

        # Create SH sensor with proper initialization
        sensor = SH(
            subap_wanted_fov=4.0,
            sensor_pxscale=0.5,
            subap_npx=8,
            subap_on_diameter=8,
            wavelengthInNm=600,
            target_device_idx=target_device_idx
        )

        # ------------------------------------------------------------------------------
        # Set up inputs for ShSlopec
        subap_on_diameter = 2
        subap_npx = 100
        idxs = {}
        map = {}
        mask_subap = xp.ones((subap_on_diameter*subap_npx, subap_on_diameter*subap_npx))

        count = 0
        for i in range(subap_on_diameter):
            for j in range(subap_on_diameter):
                mask_subap *= 0
                mask_subap[i*subap_npx:(i+1)*subap_npx,j*subap_npx:(j+1)*subap_npx] = 1
                idxs[count] = xp.where(mask_subap == 1)
                map[count] = j * subap_on_diameter + i
                count += 1

        v = xp.zeros((len(idxs), subap_npx*subap_npx), dtype=int)
        m = xp.zeros(len(idxs), dtype=int)
        for k, idx in idxs.items():
            v[k] = xp.ravel_multi_index(idx, mask_subap.shape)
            m[k] = map[k]

        subapdata = SubapData(idxs=v, display_map = m, nx=subap_on_diameter, ny=subap_on_diameter, target_device_idx=target_device_idx)

        # Create SH slope computer
        slopec = ShSlopec(
            subapdata=subapdata,  # Use a test tag
            weightedPixRad=4.0,
            target_device_idx=target_device_idx
        )

        slopes = Slopes(2, target_device_idx=target_device_idx)
        cmd = BaseValue(value=xp.zeros(2), target_device_idx=target_device_idx)

        # Create calibrator with im_tag='auto' to trigger automatic generation
        calibrator = ImCalibrator(
            nmodes=10, 
            data_dir=self.test_dir,
            im_tag='auto',
            pupilstop=pupilstop,
            source=source,
            dm=dm,
            sensor=sensor,
            slopec=slopec,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)
        calibrator.setup()

        print(f"Generated IM tag: {calibrator.im_tag}")

        # Check that im_tag was automatically generated
        self.assertIsNotNone(calibrator.im_tag)
        self.assertIsInstance(calibrator.im_tag, str)
        self.assertTrue(len(calibrator.im_tag) > 0)
        self.assertNotEqual(calibrator.im_tag, 'auto')

        # Check that the generated tag contains expected components
        self.assertIn('_sh', calibrator.im_tag)  # Should contain sensor type
        self.assertIn('pup', calibrator.im_tag)  # Should contain pupil info
        self.assertIn('coor', calibrator.im_tag)  # Should contain coordinates info
        self.assertIn('mds', calibrator.im_tag)  # Should contain modes info
        self.assertIn('stop', calibrator.im_tag)  # Should contain pupilstop info