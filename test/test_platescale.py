import unittest
import os
import yaml
import specula
specula.init(-1,precision=1)  # Default target device

import numpy as np

from specula.simul import Simul
from specula.lib.platescale_coeff import platescale_coeff
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.dm import DM
from specula.processing_objects.linear_combination import LinearCombination

class TestPlateScale(unittest.TestCase):
    """Test"""

    def test_platescale(self):
        """"""
        verbose = False  # Set to True to print debug information
        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        yml_files = ['params_platescale_test.yml']
        simul = Simul(*yml_files)

        with open(yml_files[0], 'r') as stream:
            params = yaml.safe_load(stream)

        # Build the DMs using the parameters
        simul.build_objects(params)

        # Sort DMs by their names
        dm_keys = sorted([k for k in simul.objs.keys() if k.startswith('dm')],
                        key=lambda x: int(''.join(filter(str.isdigit, x))))
        dm_list = [simul.objs[k] for k in dm_keys]
        coeff = platescale_coeff(dm_list, params['main']['pixel_pupil'])

        if verbose:
            print("coeff from platescale_coeff", coeff)

        # verify that the coeffiecients are not None
        self.assertIsNotNone(coeff, "coeff is None")

    def _zern_dm(self, npixels, nmodes, **kwargs):
        simul_params = SimulParams(time_step=1, pixel_pupil=64, pixel_pitch=0.015625)
        return DM(simul_params, height=0, type_str='zernike', nmodes=nmodes,
                  npixels=npixels, **kwargs)

    def test_platescale_ignores_dm_mode_selection(self):
        '''The plate scale modes are taken from the full DM basis:
        the DM start_mode does not change the coefficients'''
        dm1 = self._zern_dm(64, 10)
        coeff = platescale_coeff([dm1, self._zern_dm(128, 10)], 64)
        coeff_start = platescale_coeff([dm1, self._zern_dm(128, 10, start_mode=2)], 64)
        np.testing.assert_allclose(coeff_start, coeff)
        self.assertTrue(np.all(np.abs(coeff) > 0))

    def test_platescale_needs_five_modes(self):
        with self.assertRaises(ValueError):
            platescale_coeff([self._zern_dm(64, 10), self._zern_dm(128, 4)], 64)

    def test_platescale_modes_must_be_driven(self):
        with self.assertRaises(ValueError):
            platescale_coeff([self._zern_dm(64, 10), self._zern_dm(128, 10, start_mode=3)], 64)

    def test_linear_combination_start_modes_deprecated(self):
        simul_params = SimulParams(time_step=1, pixel_pupil=64, pixel_pitch=0.015625)
        dm1 = self._zern_dm(64, 10)
        dm3 = self._zern_dm(128, 10)
        with self.assertWarns(FutureWarning):
            LinearCombination(simul_params, dm1=dm1, dm3=dm3, start_modes=[0, 0])
        with self.assertRaises(ValueError):
            LinearCombination(simul_params, dm1=dm1, dm3=dm3, start_modes=[0, 2])
