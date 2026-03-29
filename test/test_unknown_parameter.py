import os
import unittest

import specula
specula.init(-1,precision=0)  # Default target device

from specula.simul import Simul

   
class TestUnknownParameter(unittest.TestCase):

    def test_simulation_with_unknown_parameter(self):
        """Run a simulation with an unknown paramter in the YAML file, and check that it raises an error."""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        print("Running simulation with an unknown parameter...")
        yml_files = ['params_unknown_parameter_test.yml']
        simul = Simul(*yml_files)

        with self.assertRaises(ValueError):
            simul.run()

