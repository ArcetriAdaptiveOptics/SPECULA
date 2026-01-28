import specula
specula.init(0)  # Default target device

import unittest

from specula.data_objects.simul_params import SimulParams


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
