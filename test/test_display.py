import os

import specula
specula.init(0)  # Default target device

import pytest
import unittest

import numpy as np
import matplotlib

from specula import cpuArray
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.psf import PSF
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.display.phase_display import PhaseDisplay
from specula.display.pixels_display import PixelsDisplay
from specula.display.slopec_display import SlopecDisplay
from specula.display.psf_display import PsfDisplay
from test.specula_testlib import cpu_and_gpu


matplotlib.use('Agg')  # Use non-interactive backend for GitHub CI


class TestDisplays(unittest.TestCase):
    """Test display classes for proper initialization and basic functionality"""

    def setUp(self):
        """Set up common test data"""
        self.pixel_pupil = 64
        self.pixel_pitch = 0.1
        self.S0 = 1.0

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @cpu_and_gpu
    def test_phase_display_init_and_trigger(self, target_device_idx, xp):
        """Test PhaseDisplay initialization and trigger"""
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                          S0=self.S0, target_device_idx=target_device_idx)
        ef.generation_time = 1
        
        display = PhaseDisplay(title='Test Phase Display')
        display.inputs['phase'].set(ef)

        # Test trigger creates figure
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        self.assertEqual(display._title, 'Test Phase Display')
        self.assertIsNotNone(display.inputs['phase'])
        self.assertTrue(display._opened)
        self.assertIsNotNone(display.fig)
        self.assertIsNotNone(display.ax)

        display.close()

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @cpu_and_gpu
    def test_pixels_display_init_and_trigger(self, target_device_idx, xp):
        """Test PixelsDisplay initialization and trigger"""
        pixels_data = xp.arange(9).reshape((3,3))
        pixels = Pixels(3, 3, bits=16, signed=0, target_device_idx=target_device_idx)
        pixels.set_value(pixels_data)
        pixels.generation_time = 1

        display = PixelsDisplay(title='Test Pixels Display')
        display.inputs['pixels'].set(pixels)

        # Test trigger creates figure and displays content
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        self.assertEqual(display._title, 'Test Pixels Display')
        self.assertIsNotNone(display.inputs['pixels'])
        self.assertTrue(display._opened)
        self.assertIsNotNone(display.img)

        display.close()

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @cpu_and_gpu
    def test_slopec_display_init_and_trigger(self, target_device_idx, xp):
        """Test SlopecDisplay initialization and trigger"""
        slopes_data = xp.random.random(100)
        slopes = Slopes(slopes=slopes_data, target_device_idx=target_device_idx)
        slopes.generation_time = 1

        display = SlopecDisplay(title='Test Slopes Display')
        display.inputs['slopes'].set(slopes)

        # Test trigger creates figure and displays content
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        self.assertEqual(display._title, 'Test Slopes Display')
        self.assertIsNotNone(display.inputs['slopes'])
        self.assertTrue(display._opened)
        self.assertIsNotNone(display.fig)
        self.assertIsNotNone(display.ax)

        display.close()

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @cpu_and_gpu
    def test_psf_display_init_and_trigger(self, target_device_idx, xp):
        """Test PsfDisplay initialization and trigger"""
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                          S0=self.S0, target_device_idx=target_device_idx)
        ef.generation_time = 1

        simulParams = SimulParams(time_step=0.001, pixel_pupil=self.pixel_pupil, pixel_pitch=self.pixel_pitch)

        psf = PSF(simulParams, wavelengthInNm=500, target_device_idx=target_device_idx)
        psf.inputs['in_ef'].set(ef)
        psf.setup()
        psf.check_ready(1)
        psf.trigger()
        psf.post_trigger()

        display = PsfDisplay(title='Test PSF Display')
        display.inputs['psf'].set(psf.outputs['out_psf'])

        # Test trigger creates figure and displays content
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        self.assertEqual(display._title, 'Test PSF Display')
        self.assertIsNotNone(display.inputs['psf'])
        self.assertTrue(display._opened)
        self.assertIsNotNone(display.img)

        display.close()

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @cpu_and_gpu
    def test_modes_display_trigger(self, target_device_idx, xp):
        """Test ModesDisplay trigger functionality"""
        from specula.display.modes_display import ModesDisplay
        from specula.base_value import BaseValue

        # Create test modes data
        modes_data = xp.random.random(20) * 100 - 50  # Random values between -50 and 50
        modes = BaseValue(modes_data, target_device_idx=target_device_idx)
        modes.generation_time = 1

        display = ModesDisplay(title='Test Modes Display', yrange=(-100, 100))
        display.inputs['modes'].set(modes)

        # Test trigger creates figure and displays content
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        self.assertTrue(display._opened)
        self.assertIsNotNone(display.fig)
        self.assertIsNotNone(display.ax)

        display.close()

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @cpu_and_gpu
    def test_plot_display_trigger(self, target_device_idx, xp):
        """Test PlotDisplay trigger functionality"""
        from specula.display.plot_display import PlotDisplay
        from specula.base_value import BaseValue

        # Create test scalar value for history plotting
        value = BaseValue([42.5], target_device_idx=target_device_idx)

        display = PlotDisplay(title='Test Plot Display', histlen=50)
        display.inputs['value'].set(value)

        display.setup()

        # Test multiple triggers to build history
        for i in range(5):
            value.generation_time = i+1
            value.set_value([10 * i + xp.random.random()])
            display.check_ready(i+1)
            display.trigger_code()

        self.assertTrue(display._opened)
        self.assertIsNotNone(display.lines)
        self.assertEqual(display._count, 5)

        display.close()

    def test_display_figsize_parameter(self):
        """Test that figsize parameter is properly handled"""
        figsize = (8, 6)
        display = PhaseDisplay(figsize=figsize)
        self.assertEqual(display._figsize, figsize)

    def test_display_log_scale_parameter(self):
        """Test log_scale parameter for PixelsDisplay"""
        display = PixelsDisplay(log_scale=True)
        self.assertTrue(display._log_scale)

        display = PixelsDisplay(log_scale=False)
        self.assertFalse(display._log_scale)

    @cpu_and_gpu
    def test_display_data_consistency(self, target_device_idx, xp):
        """Test that display maintains data consistency"""
        ef = ElectricField(32, 32, 0.1, S0=2.5, target_device_idx=target_device_idx)

        display = PhaseDisplay()
        display.inputs['phase'].set(ef)

        # Check that the input data matches what we set
        retrieved_ef = display.inputs['phase'].get(target_device_idx)
        np.testing.assert_array_equal(cpuArray(ef.phaseInNm), cpuArray(retrieved_ef.phaseInNm))

    @cpu_and_gpu
    def test_multiple_displays_same_data(self, target_device_idx, xp):
        """Test that multiple displays can use the same data source"""
        ef = ElectricField(32, 32, 0.1, S0=1.0, target_device_idx=target_device_idx)

        display1 = PhaseDisplay(title='Display 1')
        display2 = PhaseDisplay(title='Display 2')

        display1.inputs['phase'].set(ef)
        display2.inputs['phase'].set(ef)

        # Both displays should have access to the same data
        ef1 = display1.inputs['phase'].get(target_device_idx)
        ef2 = display2.inputs['phase'].get(target_device_idx)

        np.testing.assert_array_equal(cpuArray(ef1.phaseInNm), cpuArray(ef2.phaseInNm))

    def test_display_title_customization(self):
        """Test custom titles for displays"""
        custom_titles = [
            'Custom Phase Display',
            'My Pixels View',
            'Slopes Monitor',
            'PSF Viewer'
        ]

        displays = [
            PhaseDisplay(title=custom_titles[0]),
            PixelsDisplay(title=custom_titles[1]),
            SlopecDisplay(title=custom_titles[2]),
            PsfDisplay(title=custom_titles[3])
        ]

        for display, expected_title in zip(displays, custom_titles):
            self.assertEqual(display._title, expected_title)