import os
import sys
import subprocess
import logging
import unittest
from unittest.mock import MagicMock, patch

import specula
specula.init(0)  # Default target device

from specula.base_time_obj import BaseTimeObj

from test.specula_testlib import cpu_and_gpu

class TestBaseValue(unittest.TestCase):

    # ---------- INITIALIZATION TESTS ----------

    @cpu_and_gpu
    def test_init_defaults_cpu_and_gpu(self, target_device_idx, xp):
        """Check that BaseTimeObj initializes properly for CPU and GPU."""
        obj = BaseTimeObj(target_device_idx=target_device_idx)

        # Device index is set correctly
        self.assertEqual(obj.target_device_idx, target_device_idx)

        # xp is set to numpy for CPU, cupy for GPU
        if target_device_idx >= 0:
            self.assertEqual(obj.xp_str, "cp")
        else:
            self.assertEqual(obj.xp_str, "np")

        # Precision sets dtype correctly
        self.assertEqual(obj.dtype, obj.xp.float64 if obj.precision == 0 else obj.xp.float32)

    @cpu_and_gpu
    def test_init_uses_global_precision_if_none(self, target_device_idx, xp):
        """If precision is None, global_precision should be used."""
        with patch("iteras.global_precision", 1):
            obj = BaseTimeObj(target_device_idx=target_device_idx)
            self.assertEqual(obj.precision, 1)

    @cpu_and_gpu
    def test_gpu_initialization_calls_device_use(self, target_device_idx, xp):
        """For GPU devices, device.use() must be called and memory usage recorded."""
        if target_device_idx < 0:
            self.skipTest("CPU mode — not applicable")

        with patch("iteras.cp.cuda.Device") as mock_device:
            mock_device.return_value = MagicMock()
            obj = BaseTimeObj(target_device_idx=target_device_idx)
            mock_device.assert_called_once_with(target_device_idx)
            obj._target_device.use.assert_called_once()

    # ---------- TIME CONVERSION TESTS ----------

    @cpu_and_gpu
    def test_time_conversion_round_trip(self, target_device_idx, xp):
        """Ensure seconds -> t -> seconds round-trips correctly."""
        obj = BaseTimeObj(target_device_idx=target_device_idx)
        seconds = 3.141592
        t = obj.seconds_to_t(seconds)
        back_to_seconds = obj.t_to_seconds(t)
        self.assertAlmostEqual(back_to_seconds, seconds, places=9)

    # ---------- MEMORY MONITORING TESTS ----------

    @cpu_and_gpu
    def test_start_and_stop_mem_usage_count_gpu(self, target_device_idx, xp):
        """Ensure GPU memory counters update correctly."""
        if target_device_idx < 0:
            self.skipTest("CPU mode — not applicable")

        obj = BaseTimeObj(target_device_idx=target_device_idx)

        with patch("iteras.cp.get_default_memory_pool") as mock_pool:
            mock_pool.return_value.used_bytes.side_effect = [1000, 2500]
            obj.startMemUsageCount()
            obj.stopMemUsageCount()
            self.assertEqual(obj.gpu_bytes_used, 1500)

    @cpu_and_gpu
    def test_print_mem_usage_calls_print_on_gpu(self, target_device_idx, xp):
        """Ensure printMemUsage prints only for GPU."""
        obj = BaseTimeObj(target_device_idx=target_device_idx)
        obj.gpu_bytes_used = 1048576  # 1MB
        if target_device_idx >= 0:
            with self.assertLogs(obj.logger.logger, logging.DEBUG):
                obj.printMemUsage()
        else:
            with self.assertNoLogs(obj.logger.logger, logging.INFO):
                obj.printMemUsage()

    # ---------- MONITORMEM DECORATOR TESTS ----------

    @cpu_and_gpu
    def test_monitor_mem_wraps_methods(self, target_device_idx, xp):
        """Ensure monitorMem decorator wraps __init__ and setup properly."""
        calls = []

        class Dummy(BaseTimeObj):
            def __init__(self, target_device_idx=None):
                calls.append("init")
                super().__init__(target_device_idx=target_device_idx)

            def setup(self):
                calls.append("setup")

        obj = Dummy(target_device_idx=target_device_idx)
        obj.setup()
        self.assertIn("init", calls)
        self.assertIn("setup", calls)

    # ---------- TO_XP TESTS ----------

    @cpu_and_gpu
    def test_to_xp_passes_through_to_global_function(self, target_device_idx, xp):
        """Ensure to_xp delegates to the global to_xp function correctly."""
        obj = BaseTimeObj(target_device_idx=target_device_idx)

        with patch("specula.base_time_obj.to_xp", return_value="converted") as mock_to_xp:
            result = obj.to_xp([1, 2, 3], dtype=obj.dtype, force_copy=True)
            mock_to_xp.assert_called_once_with(obj.xp, [1, 2, 3], obj.dtype, True)
            self.assertEqual(result, "converted")

    @cpu_and_gpu
    def test_center_of_mass_available(self, target_device_idx, xp):
        """Ensure center_of_mass helper is available and returns correct coordinates."""
        obj = BaseTimeObj(target_device_idx=target_device_idx)
        arr = obj.xp.zeros((5, 5), dtype=obj.dtype)
        arr[1, 3] = 1
        yc, xc = obj.ndimage_center_of_mass(arr)
        self.assertAlmostEqual(float(yc), 1.0)
        self.assertAlmostEqual(float(xc), 3.0)

    @cpu_and_gpu
    def test_scipy_helpers_follow_device(self, target_device_idx, xp):
        """The scipy helpers come from scipy on CPU and from cupyx on GPU"""
        obj = BaseTimeObj(target_device_idx=target_device_idx)
        if target_device_idx >= 0:  # pragma: gpu
            import cupyx.scipy.ndimage, cupyx.scipy.fft, cupyx.scipy.linalg
            expected = {'ndimage_rotate': cupyx.scipy.ndimage.rotate,
                        'ndimage_shift': cupyx.scipy.ndimage.shift,
                        'ndimage_center_of_mass': cupyx.scipy.ndimage.center_of_mass,
                        '_scipy_ifft2': cupyx.scipy.fft.ifft2,
                        'dct': cupyx.scipy.fft.dct, 'idct': cupyx.scipy.fft.idct,
                        '_lu_factor': cupyx.scipy.linalg.lu_factor,
                        '_lu_solve': cupyx.scipy.linalg.lu_solve}
        else:
            import scipy.ndimage, scipy.fft, scipy.fftpack, scipy.linalg
            expected = {'ndimage_rotate': scipy.ndimage.rotate,
                        'ndimage_shift': scipy.ndimage.shift,
                        'ndimage_center_of_mass': scipy.ndimage.center_of_mass,
                        '_scipy_ifft2': scipy.fft.ifft2,
                        'dct': scipy.fftpack.dct, 'idct': scipy.fftpack.idct,
                        '_lu_factor': scipy.linalg.lu_factor,
                        '_lu_solve': scipy.linalg.lu_solve}
        for name, func in expected.items():
            with self.subTest(name=name):
                self.assertIs(getattr(obj, name), func)
        # Not stored in the object (e.g. when it is pickled)
        self.assertFalse(set(expected) & set(vars(obj)))

    @unittest.skipIf(specula.cp is None, "needs a GPU to copy to")
    def test_scipy_helpers_after_copy_to(self):
        """A data object copied to another device gets the helpers of that device"""
        from specula.base_value import BaseValue
        import scipy.ndimage
        import cupyx.scipy.ndimage
        gpu_value = BaseValue(value=specula.cp.zeros(3), target_device_idx=0)
        cpu_value = gpu_value.copyTo(-1)
        self.assertIs(gpu_value.ndimage_shift, cupyx.scipy.ndimage.shift)
        self.assertIs(cpu_value.ndimage_shift, scipy.ndimage.shift)

    def test_scipy_helpers_not_imported_at_creation(self):
        """Creating objects does not import the scipy helper modules"""
        # Modules already loaded by "import cupy" are not counted (CuPy 12 loads scipy.linalg)
        r = self._run_python('import sys\n'
                             'try:\n    import cupy\nexcept Exception:\n    pass\n'
                             'helpers = ("scipy.ndimage", "scipy.linalg", "scipy.fftpack")\n'
                             'before = [m for m in helpers if m in sys.modules]\n'
                             'import iteras; iteras.init(-1); '
                             'from iteras.base_value import BaseValue; BaseValue(); '
                             'loaded = [m for m in helpers if m in sys.modules and m not in before]; '
                             'assert not loaded, loaded')
        self.assertEqual(r.returncode, 0, r.stderr)

    # ------ Time resolution tests

    @cpu_and_gpu
    def test_time_conversion_with_custom_time_resolution(self, target_device_idx, xp):
        """Verify t_to_seconds and seconds_to_t with a custom time_resolution."""
        obj = BaseTimeObj(target_device_idx=target_device_idx)
        obj._time_resolution = 1_000  # 1 kHz resolution

        # 2.5 seconds should correspond to 2500 "ticks"
        seconds = 2.5
        t = obj.seconds_to_t(seconds)
        self.assertEqual(t, 2500)

        # Convert back
        recovered_seconds = obj.t_to_seconds(t)
        self.assertAlmostEqual(recovered_seconds, seconds, places=6)


    @cpu_and_gpu
    def test_time_conversion_with_high_resolution(self, target_device_idx, xp):
        """Check correctness with very high time resolution (1e12)."""
        obj = BaseTimeObj(target_device_idx=target_device_idx)
        obj._time_resolution = int(1e12)

        seconds = 1.23456789
        t = obj.seconds_to_t(seconds)
        expected_t = int(round(seconds, ndigits=9) * obj._time_resolution)
        self.assertEqual(t, expected_t)

        back_to_seconds = obj.t_to_seconds(t)
        self.assertAlmostEqual(back_to_seconds, seconds, places=9)


    @cpu_and_gpu
    def test_time_conversion_with_low_resolution(self, target_device_idx, xp):
        """Ensure rounding is handled correctly for very low resolution."""
        obj = BaseTimeObj(target_device_idx=target_device_idx)
        obj._time_resolution = 10  # Very low resolution: 10 ticks per second

        seconds = 3.14159
        t = obj.seconds_to_t(seconds)

        # With 10 ticks per second, we expect round(seconds*10)
        expected_t = int(round(seconds, ndigits=9) * 10)
        self.assertEqual(t, expected_t)

        # Back conversion: small resolution leads to quantization errors
        recovered_seconds = obj.t_to_seconds(t)
        self.assertAlmostEqual(recovered_seconds, expected_t / 10, places=6)


    @cpu_and_gpu
    def test_time_conversion_symmetry_for_various_resolutions(self, target_device_idx, xp):
        """Check seconds -> t -> seconds symmetry for several custom resolutions."""
        obj = BaseTimeObj(target_device_idx=target_device_idx)

        for resolution in [1, 1000, int(1e6), int(1e9)]:
            obj._time_resolution = resolution
            seconds = 0.123456789
            t = obj.seconds_to_t(seconds)
            recovered = obj.t_to_seconds(t)
            # Within one "tick" of precision
            self.assertAlmostEqual(recovered, seconds, delta=1/resolution)

    @cpu_and_gpu
    def test_time_conversion_modulus(self, target_device_idx, xp):
        '''
        Test that modulus operation work on the result of seconds_to_t(),
        even when the original data would fail in floating point
        '''
        obj = BaseTimeObj(target_device_idx=target_device_idx)

        dt = 1.0
        time_step = 0.002   # 1.0 % 0.002 == 0.0019999999999999792

        dt = obj.seconds_to_t(dt)
        time_step = obj.seconds_to_t(time_step)
        assert dt % time_step == 0

    # ---------- SPECULA INITIALIZATION CHECKS ----------
    # These run in a subprocess, since specula.init() has already been
    # called in this process

    def _run_python(self, code):
        # Make sure the subprocess imports the same specula package as this process
        env = os.environ.copy()
        root = os.path.dirname(os.path.dirname(os.path.abspath(specula.__file__)))
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [root, env.get('PYTHONPATH')]))
        return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)

    def test_uninitialized_raises(self):
        r = self._run_python('from specula.base_time_obj import BaseTimeObj; BaseTimeObj()')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Not initialized', r.stderr, r.stderr)

    def test_init_after_import_works(self):
        # The framework reads the settings when objects are created,
        # so init() can be called after importing its modules
        r = self._run_python('import specula; from specula.base_time_obj import BaseTimeObj; '
                             'specula.init(-1, precision=0); o = BaseTimeObj(); '
                             'assert o.target_device_idx == -1 and o.precision == 0, vars(o)')
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_init_before_import_works(self):
        r = self._run_python('import specula; specula.init(-1); '
                             'from specula.base_time_obj import BaseTimeObj; BaseTimeObj()')
        self.assertEqual(r.returncode, 0, r.stderr)

    # Same checks in-process, simulating the uninitialized globals with patch

    def test_uninitialized_raises_inprocess(self):
        # The global state is kept in iteras, specula forwards it
        with patch('iteras.global_precision', None):
            with self.assertRaisesRegex(RuntimeError, 'Not initialized'):
                BaseTimeObj()

    def test_settings_read_at_creation(self):
        # Objects created after a new init() use its settings
        with patch('iteras.global_precision', 0), patch('iteras.default_target_device_idx', -1):
            obj = BaseTimeObj()
        self.assertEqual((obj.precision, obj.target_device_idx), (0, -1))
        with patch('iteras.global_precision', 1), patch('iteras.default_target_device_idx', -1):
            obj = BaseTimeObj()
        self.assertEqual((obj.precision, obj.target_device_idx), (1, -1))

    def test_gpu_without_cupy_raises(self):
        with patch('iteras.cp', None):
            with self.assertRaisesRegex(RuntimeError, 'cupy is not available'):
                BaseTimeObj(target_device_idx=0)
