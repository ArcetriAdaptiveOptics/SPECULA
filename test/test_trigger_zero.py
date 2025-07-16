import unittest
import os
import specula
specula.init(-1,precision=1)  # Default target device

from specula.simul import Simul

class TestTriggerZero(unittest.TestCase):
    """Test that the simulation runs even when a trigger step is devoid of any processing object"""

    def setUp(self):
        self.cwd = os.getcwd()

    def tearDown(self):
        # Change back to original directory
        os.chdir(self.cwd)

    def test_trigger_zero(self):
        """Run the simulation and check the results"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        yml_files = ['params_trigger_zero.yml']
        simul = Simul(*yml_files)
        simul.run()


