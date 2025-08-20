import unittest
import os
import yaml
import specula
specula.init(-1,precision=1)  # Default target device

from specula.simul import Simul
from specula.lib.platescale_coeff import platescale_coeff

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

        # start_modes values are extracted from the params dictionary
        start_modes = []
        dm_params = {k: v for k, v in params.items() if k.startswith('dm')}
        for dm_param in dm_params.values():
            if 'start_mode' in dm_param:
                start_modes.append(dm_param['start_mode'])
            else:
                start_modes.append(0)

        # Build the DMs using the parameters
        simul.build_objects(params)

        # Sort DMs by their names
        dm_keys = sorted([k for k in simul.objs.keys() if k.startswith('dm')],
                        key=lambda x: int(''.join(filter(str.isdigit, x))))
        dm_list = [simul.objs[k] for k in dm_keys]
        # Call platescale_coeff with the DMs and their starting modes
        coeff = platescale_coeff(dm_list, start_modes, params['main']['pixel_pupil'])

        if verbose:
            print("coeff from platescale_coeff", coeff)

        # verify that the coeffiecients are not None
        self.assertIsNotNone(coeff, "coeff is None")