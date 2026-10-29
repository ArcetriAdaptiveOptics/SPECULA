import specula
specula.init(0)  # Default target device

import unittest

from specula.data_objects.simul_params import SimulParams
from specula.simul import Simul
from iteras.data_objects.sim_params import SimParams


class TestSimulParams(unittest.TestCase):

    def test_default_initialization(self):
        """Test default values of the simulation parameters."""
        params = SimulParams()

        self.assertIsNone(params.pixel_pupil)
        self.assertIsNone(params.pixel_pitch)
        self.assertEqual(params.root_dir, '.')
        self.assertEqual(params.total_time, 0.1)
        self.assertEqual(params.time_step, 0.001)
        self.assertEqual(params.zenithAngleInDeg, 0)
        self.assertFalse(params.display_server)
        self.assertEqual(params.add_modules, [])

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        params = SimulParams(
            pixel_pupil=128,
            pixel_pitch=0.0005,
            root_dir='/tmp/sim',
            total_time=2.0,
            time_step=0.01,
            zenithAngleInDeg=30,
            add_modules=['foo', 'bar'],
            display_server=True
        )

        self.assertEqual(params.pixel_pupil, 128)
        self.assertAlmostEqual(params.pixel_pitch, 0.0005)
        self.assertEqual(params.root_dir, '/tmp/sim')
        self.assertEqual(params.total_time, 2.0)
        self.assertEqual(params.time_step, 0.01)
        self.assertEqual(params.zenithAngleInDeg, 30)
        self.assertTrue(params.display_server)
        self.assertEqual(params.add_modules, ['foo', 'bar'])


    def test_wrong_add_modules(self):
        """Test initialization with wrong values for add_modules"""
        with self.assertRaises(ValueError):
            _ = SimulParams(add_modules=42)

        with self.assertRaises(ValueError):
            _ = SimulParams(add_modules=[3.1415, 42])

    def test_positional_arguments(self):
        """pixel_pupil and pixel_pitch are still the first two arguments"""
        params = SimulParams(160, 0.05)
        self.assertEqual(params.pixel_pupil, 160)
        self.assertEqual(params.pixel_pitch, 0.05)

    def test_is_iteras_sim_params(self):
        """SimulParams adds the AO parameters to the generic iteras SimParams"""
        self.assertTrue(issubclass(SimulParams, SimParams))
        params = SimParams(total_time=1.0, time_step=0.01)
        self.assertEqual(params.total_time, 1.0)
        self.assertEqual(params.time_step, 0.01)
        self.assertFalse(hasattr(params, 'pixel_pupil'))

    def test_simul_finds_main_params_by_class(self):
        """Simul uses the section whose class is SimParams or a subclass as main parameters"""
        for classname in ['SimulParams', 'SimParams']:
            params = {'other': {'class': 'BaseValue'},
                      'main': {'class': classname, 'root_dir': 'dummy'},
                      'broken': {'not_a_class': 1}}
            simul = Simul('dummy.yml')
            self.assertTrue(simul.is_main_params(params, 'main'))
            self.assertFalse(simul.is_main_params(params, 'other'))
            self.assertFalse(simul.is_main_params(params, 'broken'))
            simul.setSimulParams(params)
            self.assertIs(simul.mainParams, params['main'])
