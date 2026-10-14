import specula
specula.init(0)  # Default target device

import pytest
import unittest
import inspect
import time
import queue
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import matplotlib

from specula import cpuArray
from specula.loop_control import LoopControl
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.psf import PSF
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.display import display_process
from specula.display.base_display import BaseDisplay
from specula.display.phase_display import PhaseDisplay
from specula.display.pixels_display import PixelsDisplay
from specula.display.pixels_pup_display import PixelsPupDisplay
from specula.display.slopec_display import SlopecDisplay
from specula.display.psf_display import PsfDisplay
from specula.display.plot_display import PlotDisplay
from specula.display.modes_display import ModesDisplay
from specula.display.plot_vector_display import PlotVectorDisplay
from specula.display.double_phase_display import DoublePhaseDisplay
from specula.display.display_recorder import DisplayRecorder
from specula.base_value import BaseValue
from test.specula_testlib import cpu_and_gpu

# Classes that accept a window_xy constructor argument (all BaseDisplay
# subclasses except DoublePhaseDisplay)
DISPLAY_CLASSES_WITH_WINDOW_XY = [
    PixelsDisplay, PixelsPupDisplay, PlotDisplay, SlopecDisplay,
    PhaseDisplay, PsfDisplay, ModesDisplay, PlotVectorDisplay,
]


matplotlib.use('Agg')  # Use non-interactive backend for GitHub CI


class _RecordingPhaseDisplay(PhaseDisplay):
    '''PhaseDisplay that records the calls made by the display loop'''
    calls = []

    def setup(self):
        super().setup()
        self.calls.append('setup')

    def _update_display(self, phase):
        super()._update_display(phase)
        self.calls.append('update')

    def finalize(self):
        self.calls.append('finalize')


class TestDisplays(unittest.TestCase):
    """Test display classes for proper initialization and basic functionality"""

    def setUp(self):
        """Set up common test data"""
        self.pixel_pupil = 64
        self.pixel_pitch = 0.1
        self.S0 = 1.0

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_default_window_numbers_are_progressive(self):
        saved_plot_completed = dict(BaseDisplay._BaseDisplay__plot_completed)
        BaseDisplay._BaseDisplay__plot_completed = {}

        try:
            d1 = PhaseDisplay(title='Window 1')
            d2 = PhaseDisplay(title='Window 2')
            d3 = PhaseDisplay(title='Window 3', window=10)
            d4 = PhaseDisplay(title='Window 4')
            d5 = PhaseDisplay(title='Window 5', window=np.int64(20))
            d6 = PhaseDisplay(title='Window 6')
            with self.assertRaises(ValueError):
                PhaseDisplay(title='Invalid Window 0', window=0)
            with self.assertRaises(ValueError):
                PhaseDisplay(title='Invalid Window True', window=True)
            with self.assertRaises(ValueError):
                PhaseDisplay(title='Invalid Window String', window='abc')

            self.assertEqual(d1.window, 1)
            self.assertEqual(d2.window, 2)
            self.assertEqual(d3.window, 10)
            self.assertEqual(d4.window, 11)
            self.assertEqual(d5.window, 20)
            self.assertEqual(d6.window, 21)
        finally:
            BaseDisplay._BaseDisplay__plot_completed = saved_plot_completed
            matplotlib.pyplot.close('all')

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_existing_window_number_raises(self):
        saved_plot_completed = dict(BaseDisplay._BaseDisplay__plot_completed)
        BaseDisplay._BaseDisplay__plot_completed = {}

        try:
            d1 = PhaseDisplay(title='Window 1', window=7)
            with self.assertRaises(ValueError):
                PhaseDisplay(title='Window 2', window=7)
            self.assertEqual(d1.window, 7)
        finally:
            BaseDisplay._BaseDisplay__plot_completed = saved_plot_completed
            matplotlib.pyplot.close('all')

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_displays_grouped_in_one_window(self):
        saved_plot_completed = dict(BaseDisplay._BaseDisplay__plot_completed)
        BaseDisplay._BaseDisplay__plot_completed = {}

        try:
            d1 = PhaseDisplay(title='Left', window=3, subplot=121)
            d2 = PlotDisplay(title='Right', window=3, subplot=122)
            with self.assertRaises(ValueError):
                PhaseDisplay(title='Same subplot', window=3, subplot=122)
            d4 = PhaseDisplay(title='Next free window')

            self.assertIs(d1.fig, d2.fig)
            self.assertIsNot(d1.ax, d2.ax)
            self.assertEqual(len(d1.fig.axes), 2)
            self.assertEqual(d4.window, 4)
        finally:
            BaseDisplay._BaseDisplay__plot_completed = saved_plot_completed
            matplotlib.pyplot.close('all')

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_reset_windows(self):
        saved_plot_completed = dict(BaseDisplay._BaseDisplay__plot_completed)
        BaseDisplay._BaseDisplay__plot_completed = {}

        try:
            d1 = PhaseDisplay(title='First simulation', window=3)
            BaseDisplay.reset_windows()
            self.assertFalse(matplotlib.pyplot.fignum_exists(d1.fig.number))

            # A new simulation can use the same window
            d2 = PhaseDisplay(title='Second simulation', window=3)
            self.assertEqual(d2.window, 3)
            self.assertEqual(len(d2.fig.axes), 1)
        finally:
            BaseDisplay._BaseDisplay__plot_completed = saved_plot_completed
            matplotlib.pyplot.close('all')

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_double_phase_display_trigger(self, target_device_idx, xp):
        """DoublePhaseDisplay draws both phases and accumulates the PSD"""
        ef1 = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                            S0=self.S0, target_device_idx=target_device_idx)
        ef2 = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                            S0=self.S0, target_device_idx=target_device_idx)
        ef1.phaseInNm[:] = xp.arange(self.pixel_pupil)[None, :]
        ef2.phaseInNm[:] = xp.arange(self.pixel_pupil)[:, None]
        ef1.generation_time = ef1.seconds_to_t(1)
        ef2.generation_time = ef2.seconds_to_t(1)

        display = DoublePhaseDisplay()
        display.inputs['phase1'].set(ef1)
        display.inputs['phase2'].set(ef2)

        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

        self.assertEqual(display.nframes, 1)
        self.assertIsNotNone(display.img1)
        self.assertIsNotNone(display.img2)
        self.assertEqual(display.ax4.get_title(), 'PSD Average (n=1)')

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_phase_display_init_and_trigger(self, target_device_idx, xp):
        """Test PhaseDisplay initialization and trigger"""
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                          S0=self.S0, target_device_idx=target_device_idx)
        ef.generation_time = ef.seconds_to_t(1)

        display = PhaseDisplay(title='Test Phase Display')
        display.inputs['phase'].set(ef)

        # Test trigger creates figure
        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

        self.assertEqual(display.ax.get_title(), 'Test Phase Display')
        self.assertIsNotNone(display.inputs['phase'])
        self.assertIsNotNone(display.ax)
        self.assertIsNotNone(display.fig)

        matplotlib.pyplot.close(display.fig)

    @cpu_and_gpu
    def test_phase_display_in_display_process(self, target_device_idx, xp):
        """With async displays, drawing happens in the display process"""
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                          S0=self.S0, target_device_idx=target_device_idx)
        ef.generation_time = ef.seconds_to_t(1)

        value = BaseValue(value=xp.array(1.0), target_device_idx=target_device_idx)
        value.generation_time = value.seconds_to_t(1)

        display_process.init(True)
        try:
            display = PhaseDisplay(title='Async Phase Display')
            display.inputs['phase'].set(ef)
            self.assertIsNone(display.fig)
            plot = PlotDisplay(title='Async Plot Display')
            plot.inputs['value_list'].set([value])   # 'value' input left unset

            # The display process inherits the environment: no GUI windows
            with mock.patch.dict('os.environ', {'MPLBACKEND': 'Agg'}):
                display_process.start(precision=specula.global_precision, log_level='INFO')
            process = display_process._process

            loop = LoopControl()
            loop.add(display, idx=0)
            loop.add(plot, idx=0)
            loop.run(run_time=1, dt=1)
            self.assertEqual(display.outputs['out_window_id'].value, display.window)

            # More updates than the queue can hold: phase updates are skipped,
            # history plot updates never
            for _ in range(10):
                display_process.send(display)
                display_process.send(plot)
            self.assertGreater(display_process._dropped, 0)
            self.assertLessEqual(display_process._dropped, 10)
        finally:
            display_process.stop(display.logger)
            display_process.init(False)

        self.assertEqual(process.exitcode, 0)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_display_worker_loop(self):
        """Display loop of the display process, run here to check the display calls"""
        saved_plot_completed = dict(BaseDisplay._BaseDisplay__plot_completed)
        BaseDisplay._BaseDisplay__plot_completed = {}
        _RecordingPhaseDisplay.calls = []

        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                           S0=self.S0, target_device_idx=-1)
        value = BaseValue(value=np.array(1.0), target_device_idx=-1)
        specs = display_process._dumps([('phase_disp', _RecordingPhaseDisplay, (), {'title': 'Phase'}),
                                        ('plot_disp', PlotDisplay, (), {'title': 'Plot'})])
        q = queue.Queue()
        slots = threading.Semaphore(0)
        for t in [1, 2]:
            q.put((True, display_process._dumps(('phase_disp', t, {'phase': ef}))))
            q.put((False, display_process._dumps(('plot_disp', t, {'value': value, 'value_list': None}))))

        # The terminator arrives later: the loop also waits on an empty queue
        timer = threading.Timer(0.1, lambda: q.put(None))
        timer.start()
        try:
            display_process._worker_loop(q, slots, specs, log_level='INFO')
        finally:
            timer.join()
            BaseDisplay._BaseDisplay__plot_completed = saved_plot_completed
            matplotlib.pyplot.close('all')

        self.assertEqual(_RecordingPhaseDisplay.calls, ['setup', 'update', 'update', 'finalize'])
        # A queue slot is freed for each phase update, not for the history plot updates
        self.assertTrue(slots.acquire(blocking=False))
        self.assertTrue(slots.acquire(blocking=False))
        self.assertFalse(slots.acquire(blocking=False))

    def test_display_process_not_started_without_displays(self):
        display_process.init(True)
        try:
            display_process.start(precision=specula.global_precision, log_level='INFO')
            self.assertIsNone(display_process._process)
            self.assertFalse(display_process.enabled)
        finally:
            display_process.stop(None)
            display_process.init(False)

    def test_display_recorder_with_async_displays(self):
        display_process.init(True)
        try:
            with self.assertRaises(ValueError):
                DisplayRecorder(filename='unused.mp4')
        finally:
            display_process.init(False)

    def test_display_process_dead(self):
        """If the display process dies, the simulation goes on and stop() does not wait"""
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                          S0=self.S0, target_device_idx=-1)
        ef.generation_time = ef.seconds_to_t(1)

        display_process.init(True)
        try:
            display = PhaseDisplay(title='Async Phase Display')
            display.inputs['phase'].set(ef)
            with mock.patch.dict('os.environ', {'MPLBACKEND': 'Agg'}):
                display_process.start(precision=specula.global_precision, log_level='INFO')
            display_process._process.kill()
            display_process._process.join()

            loop = LoopControl()
            loop.add(display, idx=0)
            loop.run(run_time=3, dt=1)
        finally:
            t0 = time.time()
            display_process.stop(display.logger, timeout=30)
            display_process.init(False)

        self.assertLess(time.time() - t0, 5)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_pixels_display_init_and_trigger(self, target_device_idx, xp):
        """Test PixelsDisplay initialization and trigger"""
        pixels_data = xp.arange(9).reshape((3,3))
        pixels = Pixels(3, 3, bits=16, signed=0, target_device_idx=target_device_idx)
        pixels.set_value(pixels_data)
        pixels.generation_time = pixels.seconds_to_t(1)

        display = PixelsDisplay(title='Test Pixels Display')
        display.inputs['pixels'].set(pixels)

        # Test trigger creates figure and displays content
        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

        self.assertEqual(display.ax.get_title(), 'Test Pixels Display')
        self.assertIsNotNone(display.inputs['pixels'])
        self.assertIsNotNone(display.ax)
        self.assertIsNotNone(display.img)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_slopec_display_init_and_trigger(self, target_device_idx, xp):
        """Test SlopecDisplay initialization and trigger"""
        slopes_data = xp.random.random(100)
        slopes = Slopes(slopes=slopes_data, target_device_idx=target_device_idx)
        slopes.generation_time = slopes.seconds_to_t(1)

        display = SlopecDisplay(title='Test Slopes Display')
        display.inputs['slopes'].set(slopes)

        # Test trigger creates figure and displays content
        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

    #    self.assertEqual(display.ax.get_title(), 'Test Slopes Display')
        self.assertIsNotNone(display.inputs['slopes'])
        self.assertIsNotNone(display.ax)
        self.assertIsNotNone(display.fig)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_psf_display_init_and_trigger(self, target_device_idx, xp):
        """Test PsfDisplay initialization and trigger"""
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                          S0=self.S0, target_device_idx=target_device_idx)
        ef.generation_time = ef.seconds_to_t(1)

        simulParams = SimulParams(
            time_step=0.001, pixel_pupil=self.pixel_pupil, pixel_pitch=self.pixel_pitch
        )

        psf = PSF(simulParams, wavelengthInNm=500, target_device_idx=target_device_idx)
        psf.inputs['in_ef'].set(ef)

        display = PsfDisplay(title='Test PSF Display')
        display.inputs['psf'].set(psf.outputs['out_psf'])

        loop = LoopControl()
        loop.add(psf, idx=0)
        loop.add(display, idx=1)
        loop.run(run_time=1, dt=1)

        self.assertEqual(display.ax.get_title(), 'Test PSF Display')
        self.assertIsNotNone(display.inputs['psf'])
        self.assertIsNotNone(display.ax)
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
        modes.generation_time = modes.seconds_to_t(1)

        display = ModesDisplay(title='Test Modes Display', yrange=(-100, 100))
        display.inputs['modes'].set(modes)

        # Test trigger creates figure and displays content
        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

        self.assertIsNotNone(display.ax)
        self.assertIsNotNone(display.fig)

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

        self.assertIsNotNone(display.ax)
        self.assertIsNotNone(display.lines)
        self.assertEqual(display._count, 5)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_display_figsize_parameter(self):
        """Test that figsize parameter is properly handled"""
        figsize = (8, 6)
        display = PhaseDisplay(figsize=figsize)
        self.assertEqual(display.figsize, figsize)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_display_log_scale_parameter(self):
        """Test log_scale parameter for PixelsDisplay"""
        display = PixelsDisplay(log_scale=True)
        self.assertTrue(display._log_scale)

        display = PixelsDisplay(log_scale=False)
        self.assertFalse(display._log_scale)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_display_data_consistency(self, target_device_idx, xp):
        """Test that display maintains data consistency"""
        ef = ElectricField(32, 32, 0.1, S0=2.5, target_device_idx=target_device_idx)

        display = PhaseDisplay()
        display.inputs['phase'].set(ef)

        # Check that the input data matches what we set
        retrieved_ef = display.inputs['phase'].get(target_device_idx)
        np.testing.assert_array_equal(cpuArray(ef.phaseInNm), cpuArray(retrieved_ef.phaseInNm))

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
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

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
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
            self.assertEqual(display.ax.get_title(), expected_title)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_basic_vector_plot(self, target_device_idx, xp):
        """Test basic vector plotting with 3 elements"""
        display = PlotVectorDisplay(title='Test Vector', histlen=50)

        # Create 3D vector
        vec = BaseValue(value=xp.array([1.0, 2.0, 3.0]), target_device_idx=target_device_idx)
        vec.generation_time = vec.seconds_to_t(1)

        display.inputs['vector'].set(vec)

        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

        self.assertIsNotNone(display.ax)
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
        vec.generation_time = vec.seconds_to_t(1)

        display.inputs['vector'].set(vec)
        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

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
    def test_plot_with_nonvector(self, target_device_idx, xp):
        """Test basic vector plotting with a list instead of a numpy array"""
        display = PlotVectorDisplay(title='Test Vector', histlen=50)

        # Create 3D vector
        vec = BaseValue(value=[1, 2, 3], target_device_idx=target_device_idx)
        vec.generation_time = vec.seconds_to_t(1)

        display.inputs['vector'].set(vec)

        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

        self.assertIsNotNone(display.ax)
        self.assertIsNotNone(display.lines)
        self.assertEqual(len(display.lines), 3)  # 3 elements
        self.assertEqual(display._count, 1)

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
        vec.generation_time = vec.seconds_to_t(1)

        display.inputs['vector'].set(vec)
        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

        self.assertEqual(len(display.lines), 1)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_vector_from_list(self, target_device_idx, xp):
        """Test vector created from Python list"""
        display = PlotVectorDisplay()

        vec = BaseValue(value=[1.5, 2.5, 3.5], target_device_idx=target_device_idx)
        vec.generation_time = vec.seconds_to_t(1)

        display.inputs['vector'].set(vec)
        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

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
        vec.generation_time = vec.seconds_to_t(1)

        display.inputs['vector'].set(vec)
        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

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
        vec.generation_time = vec.seconds_to_t(1)

        display.inputs['vector'].set(vec)
        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

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
    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu

    def test_plot_display_with_labels(self, target_device_idx, xp):
        """Test PlotDisplay with custom labels for multiple inputs"""
        from specula.display.plot_display import PlotDisplay
        from specula.base_value import BaseValue

        # Create three test values
        value1 = BaseValue(value=xp.array([1.0]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([2.0]), target_device_idx=target_device_idx)
        value3 = BaseValue(value=xp.array([3.0]), target_device_idx=target_device_idx)

        # Set up display with custom labels
        custom_labels = ['Signal A', 'Signal B', 'Signal C']
        display = PlotDisplay(
            title='Test Multi-Input with Labels',
            histlen=50,
            labels=custom_labels
        )

        # Set up input list
        display.inputs['value_list'].set(value1)
        display.inputs['value_list'].append(value2)
        display.inputs['value_list'].append(value3)

        display.setup()

        # Trigger a few times to build history
        for i in range(3):
            value1.generation_time = i + 1
            value2.generation_time = i + 1
            value3.generation_time = i + 1
            value1.set_value(xp.array([1.0 * (i + 1)]))
            value2.set_value(xp.array([2.0 * (i + 1)]))
            value3.set_value(xp.array([3.0 * (i + 1)]))
            display.check_ready(i + 1)
            display.trigger_code()

        # Check that we have 3 lines
        self.assertEqual(len(display.lines), 3)

        # Check that labels were applied correctly
        line_labels = [line.get_label() for line in display.lines]
        self.assertEqual(line_labels, custom_labels)

        # Check that legend was added
        self.assertTrue(display._legend_added)
        legend = display.ax.get_legend()
        self.assertIsNotNone(legend)

        # Verify legend text matches our labels
        legend_texts = [text.get_text() for text in legend.get_texts()]
        self.assertEqual(legend_texts, custom_labels)

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_plot_display_partial_labels(self, target_device_idx, xp):
        """Test PlotDisplay with fewer labels than inputs"""
        from specula.display.plot_display import PlotDisplay
        from specula.base_value import BaseValue

        # Create three values but only provide two labels
        value1 = BaseValue(value=xp.array([1.0]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([2.0]), target_device_idx=target_device_idx)
        value3 = BaseValue(value=xp.array([3.0]), target_device_idx=target_device_idx)

        display = PlotDisplay(
            title='Test Partial Labels',
            labels=['First', 'Second']  # Only 2 labels for 3 inputs
        )

        display.inputs['value_list'].set(value1)
        display.inputs['value_list'].append(value2)
        display.inputs['value_list'].append(value3)

        display.setup()

        value1.generation_time = 1
        value2.generation_time = 1
        value3.generation_time = 1
        display.check_ready(1)
        display.trigger_code()

        # Check labels: first two custom, third default
        line_labels = [line.get_label() for line in display.lines]
        self.assertEqual(line_labels[0], 'First')
        self.assertEqual(line_labels[1], 'Second')
        self.assertEqual(line_labels[2], 'Input 2')  # Default for missing label

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_plot_display_no_labels(self, target_device_idx, xp):
        """Test PlotDisplay without labels uses defaults"""
        from specula.display.plot_display import PlotDisplay
        from specula.base_value import BaseValue

        value1 = BaseValue(value=xp.array([1.0]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([2.0]), target_device_idx=target_device_idx)

        display = PlotDisplay(title='Test Default Labels')  # No labels parameter

        display.inputs['value_list'].set(value1)
        display.inputs['value_list'].append(value2)

        display.setup()

        value1.generation_time = 1
        value2.generation_time = 1
        display.check_ready(1)
        display.trigger_code()

        # Check default labels
        line_labels = [line.get_label() for line in display.lines]
        self.assertEqual(line_labels[0], 'Input 0')
        self.assertEqual(line_labels[1], 'Input 1')

        matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_plot_display_single_input_no_legend(self, target_device_idx, xp):
        """Test that single input doesn't create legend"""
        from specula.display.plot_display import PlotDisplay
        from specula.base_value import BaseValue

        value = BaseValue(value=xp.array([42.5]), target_device_idx=target_device_idx)
        value.generation_time = value.seconds_to_t(1)

        display = PlotDisplay(
            title='Single Input',
            labels=['Only Signal']
        )
        display.inputs['value'].set(value)

        loop = LoopControl()
        loop.add(display, idx=0)
        loop.run(run_time=1, dt=1)

        # Single input should not add legend
        self.assertFalse(display._legend_added)
        legend = display.ax.get_legend()
        self.assertIsNone(legend)

        matplotlib.pyplot.close(display.fig)

    def test_window_xy_signature_present(self):
        """All BaseDisplay subclasses except DoublePhaseDisplay accept window_xy"""
        for cls in DISPLAY_CLASSES_WITH_WINDOW_XY:
            with self.subTest(cls=cls.__name__):
                sig = inspect.signature(cls.__init__)
                self.assertIn('window_xy', sig.parameters)

        # DoublePhaseDisplay is explicitly excluded from this feature
        sig = inspect.signature(DoublePhaseDisplay.__init__)
        self.assertNotIn('window_xy', sig.parameters)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_construct_with_valid_window_xy_does_not_raise(self):
        """Constructing with a valid window_xy must not raise, even though the
        Agg backend used in tests has no real window to move"""
        figs_to_close = []
        try:
            for cls in DISPLAY_CLASSES_WITH_WINDOW_XY:
                with self.subTest(cls=cls.__name__):
                    display = cls(window_xy=(100, 200))
                    figs_to_close.append(display.fig)
        finally:
            for fig in figs_to_close:
                matplotlib.pyplot.close(fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_invalid_window_xy_logs_warning_and_does_not_raise(self):
        """Invalid window_xy values must be rejected with a warning, not an exception"""
        invalid_values = ['abc', [1], (1, 'x'), 5]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertLogs('specula.PhaseDisplay', level='WARNING') as log:
                    display = PhaseDisplay(window_xy=value)
                self.assertTrue(any('window_xy' in msg for msg in log.output))
                matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_set_window_position_tk_backend(self):
        """_set_window_position uses wm_geometry() when available (Tk-like manager)"""
        display = PhaseDisplay()
        try:
            fake_window = SimpleNamespace(wm_geometry=mock.MagicMock())
            display.fig.canvas.manager.window = fake_window

            display._set_window_position([10, 20])

            fake_window.wm_geometry.assert_called_once_with('+10+20')
        finally:
            matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_set_window_position_qt_backend(self):
        """_set_window_position uses move() when wm_geometry is not available (Qt/GTK-like manager)"""
        display = PhaseDisplay()
        try:
            fake_window = SimpleNamespace(move=mock.MagicMock())
            display.fig.canvas.manager.window = fake_window

            display._set_window_position((10, 20))

            fake_window.move.assert_called_once_with(10, 20)
        finally:
            matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_set_window_position_swallows_window_manager_exceptions(self):
        """An exception raised while moving the window must not propagate"""
        display = PhaseDisplay()
        try:
            class RaisingWindow:
                def wm_geometry(self, *args, **kwargs):
                    raise RuntimeError('boom')

            display.fig.canvas.manager.window = RaisingWindow()

            # Should not raise
            display._set_window_position([10, 20])
        finally:
            matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_pixels_pup_display_img_is_none_after_init(self):
        """PixelsPupDisplay.img must be initialized to None (it used to be read
        before being assigned)"""
        display = PixelsPupDisplay()
        try:
            self.assertIsNone(display.img)
        finally:
            matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_safe_draw_skipped_with_non_interactive_backend(self):
        """With a non-interactive backend (Agg) nobody sees the figure,
        so it must not be rendered"""
        display = PhaseDisplay()
        try:
            with mock.patch.object(display.fig.canvas, 'draw_idle') as draw_idle:
                display._safe_draw()
            draw_idle.assert_not_called()
        finally:
            matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_safe_draw_done_with_open_interactive_window(self):
        """An open GUI window must be redrawn"""
        display = PhaseDisplay()
        try:
            display.fig.canvas.required_interactive_framework = 'tk'
            with mock.patch.object(display.fig.canvas, 'draw_idle') as draw_idle:
                display._safe_draw()
            draw_idle.assert_called_once()
        finally:
            matplotlib.pyplot.close(display.fig)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_safe_draw_skipped_with_closed_window(self):
        """A GUI window closed by the user must not be redrawn"""
        display = PhaseDisplay()
        display.fig.canvas.required_interactive_framework = 'tk'
        matplotlib.pyplot.close(display.fig)
        with mock.patch.object(display.fig.canvas, 'draw_idle') as draw_idle:
            display._safe_draw()
        draw_idle.assert_not_called()

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    def test_safe_draw_done_on_notebook(self):
        """On notebooks the figure is drawn even with a non-interactive backend"""
        display = PhaseDisplay()
        try:
            display.onNotebook = True
            with mock.patch.object(display.fig.canvas, 'draw_idle') as draw_idle:
                display._safe_draw()
            draw_idle.assert_called_once()
        finally:
            matplotlib.pyplot.close(display.fig)
