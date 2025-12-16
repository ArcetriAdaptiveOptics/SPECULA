import unittest
from unittest.mock import MagicMock, patch
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
        with patch("specula.base_time_obj.global_precision", 1):
            obj = BaseTimeObj(target_device_idx=target_device_idx)
            self.assertEqual(obj.precision, 1)

    @cpu_and_gpu
    def test_gpu_initialization_calls_device_use(self, target_device_idx, xp):
        """For GPU devices, device.use() must be called and memory usage recorded."""
        if target_device_idx < 0:
            self.skipTest("CPU mode — not applicable")

        with patch("specula.base_time_obj.cp.cuda.Device") as mock_device:
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

        with patch("specula.base_time_obj.cp.get_default_memory_pool") as mock_pool:
            mock_pool.return_value.used_bytes.side_effect = [1000, 2500]
            obj.startMemUsageCount()
            obj.stopMemUsageCount()
            self.assertEqual(obj.gpu_bytes_used, 1500)

    @cpu_and_gpu
    def test_print_mem_usage_calls_print_on_gpu(self, target_device_idx, xp):
        """Ensure printMemUsage prints only for GPU."""
        obj = BaseTimeObj(target_device_idx=target_device_idx)
        with patch("builtins.print") as mock_print:
            obj.gpu_bytes_used = 1048576  # 1MB
            obj.printMemUsage()
            if target_device_idx >= 0:
                mock_print.assert_called_once()
            else:
                mock_print.assert_not_called()

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