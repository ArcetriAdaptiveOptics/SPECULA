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
from specula.display.plot_vector_display import PlotVectorDisplay
from specula.base_value import BaseValue
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
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
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

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
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

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
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

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_psf_display_init_and_trigger(self, target_device_idx, xp):
        """Test PsfDisplay initialization and trigger"""
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                          S0=self.S0, target_device_idx=target_device_idx)
        ef.generation_time = 1

        simulParams = SimulParams(
            time_step=0.001, pixel_pupil=self.pixel_pupil, pixel_pitch=self.pixel_pitch
        )

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

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_modes_display_trigger(self, target_device_idx, xp):
        """Test ModesDisplay trigger functionality"""
        from specula.display.modes_display import ModesDisplay
        from specula.base_value import BaseValue

        # Create test modes data
        modes_data = xp.random.random(20) * 100 - 50  # Random values between -50 and 50
        modes = BaseValue(value=modes_data, target_device_idx=target_device_idx)
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

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_plot_display_trigger(self, target_device_idx, xp):
        """Test PlotDisplay trigger functionality"""
        from specula.display.plot_display import PlotDisplay
        from specula.base_value import BaseValue

        # Create test scalar value for history plotting
        value = BaseValue(value=xp.array([42.5]), target_device_idx=target_device_idx)

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

        matplotlib.pyplot.close(display.fig)

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

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_basic_vector_plot(self, target_device_idx, xp):
        """Test basic vector plotting with 3 elements"""
        display = PlotVectorDisplay(title='Test Vector', histlen=50)

        # Create 3D vector
        vec = BaseValue(value=xp.array([1.0, 2.0, 3.0]), target_device_idx=target_device_idx)
        vec.generation_time = 1

        display.inputs['vector'].set(vec)
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        self.assertTrue(display._opened)
        self.assertIsNotNone(display.lines)
        self.assertEqual(len(display.lines), 3)  # 3 elements
        self.assertEqual(display._count, 1)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_vector_with_selected_indices(self, target_device_idx, xp):
        """Test plotting only selected elements"""
        # Plot only indices 0 and 2 from a 5-element vector
        display = PlotVectorDisplay(
            title='Selected elements',
            indices=[0, 2],
            legend_labels=['X', 'Z']
        )

        vec = BaseValue(value=xp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                       target_device_idx=target_device_idx)
        vec.generation_time = 1

        display.inputs['vector'].set(vec)
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        # Should only have 2 lines (indices 0 and 2)
        self.assertEqual(len(display.lines), 2)

        # Check legend labels
        labels = [line.get_label() for line in display.lines]
        self.assertIn('X', labels)
        self.assertIn('Z', labels)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_vector_history_accumulation(self, target_device_idx, xp):
        """Test that vector history accumulates correctly"""
        display = PlotVectorDisplay(histlen=10)

        vec = BaseValue(value=xp.array([0.0, 0.0]), target_device_idx=target_device_idx)
        display.inputs['vector'].set(vec)
        display.setup()

        # Add 5 samples
        for i in range(5):
            vec.set_value(xp.array([float(i), float(i * 2)]))
            vec.generation_time = i + 1
            display.check_ready(i + 1)
            display.trigger_code()

        self.assertEqual(display._count, 5)
        self.assertEqual(len(display._time_history), 5)

        # Check history values
        np.testing.assert_array_equal(display._history[0, :], [0.0, 0.0])
        np.testing.assert_array_equal(display._history[4, :], [4.0, 8.0])

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_vector_history_scrolling(self, target_device_idx, xp):
        """Test that history scrolls when buffer is full"""
        histlen = 5
        display = PlotVectorDisplay(histlen=histlen)

        vec = BaseValue(value=xp.array([0.0, 0.0]), target_device_idx=target_device_idx)
        display.inputs['vector'].set(vec)
        display.setup()

        # Add more samples than buffer size
        for i in range(10):
            vec.generation_time = i + 1
            vec.set_value(xp.array([float(i), float(i * 2)]))
            display.check_ready(i + 1)
            display.trigger_code()

        # Should have scrolled, keeping last 5
        self.assertEqual(display._count, histlen)
        self.assertEqual(len(display._time_history), histlen)

        # Last value should be [9, 18]
        np.testing.assert_array_equal(display._history[histlen-1, :], [9.0, 18.0])

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_single_element_vector(self, target_device_idx, xp):
        """Test with 1-element vector (scalar-like)"""
        display = PlotVectorDisplay(title='Scalar Vector')

        vec = BaseValue(value=xp.array([42.0]), target_device_idx=target_device_idx)
        vec.generation_time = 1

        display.inputs['vector'].set(vec)
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        self.assertEqual(len(display.lines), 1)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_vector_from_list(self, target_device_idx, xp):
        """Test vector created from Python list"""
        display = PlotVectorDisplay()

        vec = BaseValue(value=[1.5, 2.5, 3.5], target_device_idx=target_device_idx)
        vec.generation_time = 1

        display.inputs['vector'].set(vec)
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        self.assertEqual(len(display.lines), 3)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_custom_legend_labels(self, target_device_idx, xp):
        """Test custom legend labels"""
        labels = ['Tip', 'Tilt', 'Focus']
        display = PlotVectorDisplay(legend_labels=labels)

        vec = BaseValue(value=xp.array([0.1, 0.2, 0.3]), target_device_idx=target_device_idx)
        vec.generation_time = 1

        display.inputs['vector'].set(vec)
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        line_labels = [line.get_label() for line in display.lines]
        self.assertEqual(line_labels, labels)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_fixed_y_range(self, target_device_idx, xp):
        """Test fixed Y axis range"""
        yrange = (-10, 10)
        display = PlotVectorDisplay(yrange=yrange)

        vec = BaseValue(value=xp.array([5.0, -5.0]), target_device_idx=target_device_idx)
        vec.generation_time = 1

        display.inputs['vector'].set(vec)
        display.setup()
        display.check_ready(1)
        display.trigger_code()

        ylim = display.ax.get_ylim()
        self.assertEqual(ylim, yrange)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_iteration_x_axis(self, target_device_idx, xp):
        """Test iteration mode for x-axis instead of time"""
        display = PlotVectorDisplay(x_axis='iteration')

        vec = BaseValue(value=xp.array([1.0, 2.0]), target_device_idx=target_device_idx)
        display.inputs['vector'].set(vec)
        display.setup()

        for i in range(3):
            vec.generation_time = i + 1
            vec.set_value(xp.array([float(i), float(i * 2)]))
            display.check_ready(i + 1)
            display.trigger_code()

        # Check x values are iterations [1, 2, 3]
        expected_x = [1, 2, 3]
        np.testing.assert_array_equal(display._time_history, expected_x)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_pixels_display_crop_slice(self, target_device_idx, xp):
        """Test crop with slice mode"""
        display = PixelsDisplay(crop=(10, 40, 20, 50), crop_mode='slice')

        # Create 100x100 image
        size_pixels = 100
        image_data = xp.random.random((size_pixels, size_pixels))
        pix = Pixels(size_pixels, size_pixels, target_device_idx=target_device_idx)
        pix.set_value(image_data)
        pix.generation_time = 1

        # Set input BEFORE setup
        display.inputs['pixels'].set(pix)
        display.setup()

        self.assertTrue(display.check_ready(1))
        display.trigger_code()

        # Check displayed image has correct shape
        displayed = display.img.get_array()
        self.assertEqual(displayed.shape, (30, 30))  # 50-20=30 rows, 40-10=30 cols

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_pixels_display_crop_center(self, target_device_idx, xp):
        """Test crop with center mode"""
        display = PixelsDisplay(crop=(50, 50, 15, 15), crop_mode='center')

        # Create 100x100 image
        size_pixels = 100
        image_data = xp.random.random((size_pixels, size_pixels))
        pix = Pixels(size_pixels, size_pixels, target_device_idx=target_device_idx)
        pix.set_value(image_data)
        pix.generation_time = 1

        # Set input BEFORE setup
        display.inputs['pixels'].set(pix)
        display.setup()

        self.assertTrue(display.check_ready(1))
        display.trigger_code()

        # Check displayed image has correct shape
        displayed = display.img.get_array()
        self.assertEqual(displayed.shape, (30, 30))  # 2*15=30 in both dimensions

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_pixels_display_crop_dynamic(self, target_device_idx, xp):
        """Test dynamic crop change"""
        display = PixelsDisplay()

        # Create 100x100 image
        size_pixels = 100
        image_data = xp.random.random((size_pixels, size_pixels))
        pix = Pixels(size_pixels, size_pixels, target_device_idx=target_device_idx)
        pix.set_value(image_data)
        pix.generation_time = 1

        display.inputs['pixels'].set(pix)
        display.setup()

        # First display without crop
        self.assertTrue(display.check_ready(1))
        display.trigger_code()

        displayed = display.img.get_array()
        self.assertEqual(displayed.shape, (100, 100))

        # Now set crop dynamically
        display.set_crop((25, 75, 25, 75), crop_mode='slice')

        pix.generation_time = 2
        self.assertTrue(display.check_ready(2))
        display.trigger_code()

        displayed = display.img.get_array()
        self.assertEqual(displayed.shape, (50, 50))  # 75-25=50

        # Clear crop
        display.clear_crop()

        pix.generation_time = 3
        self.assertTrue(display.check_ready(3))
        display.trigger_code()

        displayed = display.img.get_array()
        self.assertEqual(displayed.shape, (100, 100))

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_pixels_display_crop_with_log_scale(self, target_device_idx, xp):
        """Test that crop works with log scale"""
        display = PixelsDisplay(
            crop=(25, 75, 25, 75),
            crop_mode='slice',
            log_scale=True
        )

        # Create 100x100 image
        size_pixels = 100
        image_data = xp.random.random((size_pixels, size_pixels)) + 1.0  # Avoid log(0)
        pix = Pixels(size_pixels, size_pixels, target_device_idx=target_device_idx)
        pix.set_value(image_data)
        pix.generation_time = 1

        display.inputs['pixels'].set(pix)
        display.setup()

        self.assertTrue(display.check_ready(1))
        display.trigger_code()

        displayed = display.img.get_array()
        self.assertEqual(displayed.shape, (50, 50))

        # Verify log scale is handled by LogNorm, not by modifying data
        import matplotlib.colors
        self.assertIsInstance(display.img.norm, matplotlib.colors.LogNorm)
        self.assertTrue(np.all(displayed >= 1.0))  # Data is unchanged

        matplotlib.pyplot.close(display.fig)
