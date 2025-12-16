

import specula
specula.init(0)  # Default target device

import unittest
import tempfile

from specula.simul import Simul


class TestDisplayServer(unittest.TestCase):

    def test_display_spawn(self):
        '''
        Test that a DisplayServer can be started

        Expected to fail on Windows and MacOS
        '''
        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy
          total_time: 0.001
          time_step: 0.001
          display_server: true
          
        test:
          class: 'Source'
          polar_coordinates: [1, 2]
          magnitude: null
          wavelengthInNm: null
        '''
        with tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False) as tmp:
            tmp.write(yml)
            yml_path = tmp.name

        simul = Simul(yml_path)
        simul.run()

