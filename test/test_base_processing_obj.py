
import specula
specula.init(0)  # Default target device

import numpy as np
import unittest
from unittest.mock import MagicMock

from specula import cp, cpuArray
from specula.base_processing_obj import BaseProcessingObj

from test.specula_testlib import cpu_and_gpu


class TestBaseProcessingObj(unittest.TestCase):
   
    def test_to_xp_from_cpu_to_cpu(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        data_cpu = np.arange(3)
        assert id(data_cpu) == id(obj.to_xp(data_cpu))

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_to_xp_from_gpu_to_cpu(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        data_gpu = cp.arange(3)
        data_cpu = obj.to_xp(data_gpu)
        assert isinstance(data_cpu, np.ndarray)
        np.testing.assert_array_equal(data_cpu, cpuArray(data_gpu))

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_to_xp_from_cpu_to_gpu(self):
        obj = BaseProcessingObj(target_device_idx=0)
        data_cpu = np.arange(3)
        data_gpu = obj.to_xp(data_cpu)
        assert isinstance(data_gpu, cp.ndarray)
        np.testing.assert_array_equal(data_cpu, cpuArray(data_gpu))

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_to_xp_from_gpu_to_gpu(self):
        obj = BaseProcessingObj(target_device_idx=0)
        data_gpu = cp.arange(3)
        assert id(data_gpu) == id(obj.to_xp(data_gpu))

    def test_to_xp_from_cpu_to_cpu_with_copy(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        data_cpu = np.arange(3)
        assert id(data_cpu) != id(obj.to_xp(data_cpu, force_copy=True))

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_to_xp_from_gpu_to_gpu_with_copy(self):
        obj = BaseProcessingObj(target_device_idx=0)
        data_gpu = cp.arange(3)
        assert id(data_gpu) != id(obj.to_xp(data_gpu, force_copy=True))

    @cpu_and_gpu
    def test_initialization(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        self.assertEqual(obj.current_time, 0)
        self.assertEqual(obj.current_time_seconds, 0)
        self.assertEqual(obj.inputs, {})
        self.assertEqual(obj.local_inputs, {})
        self.assertEqual(obj.outputs, {})
        self.assertEqual(obj.remote_outputs, {})
        self.assertEqual(obj.sent_valid, {})
        self.assertEqual(obj.name, "BaseProcessingObj")
        self.assertEqual(obj.target_device_idx, target_device_idx)

    @cpu_and_gpu
    def test_prepare_trigger_updates_current_time_seconds(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.t_to_seconds = MagicMock(return_value=42.5)

        obj.prepare_trigger(t=10)
        self.assertEqual(obj.current_time_seconds, 42.5)

    @cpu_and_gpu
    def test_add_remote_output(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.addRemoteOutput("output1", ("rank", "tag", "delay"))
        self.assertIn("output1", obj.remote_outputs)
        self.assertEqual(obj.remote_outputs["output1"], [("rank", "tag", "delay")])

    @cpu_and_gpu
    def test_check_input_times_empty_returns_true(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        self.assertTrue(obj.checkInputTimes())

    @cpu_and_gpu
    def test_check_input_times_with_inputs(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)

        mock_input = MagicMock()
        mock_input.generation_time = 5
        obj.local_inputs = {"test": mock_input}
        obj.get_all_inputs = MagicMock()
        obj.inputs['test'] = MagicMock()   # Necessary to skip the first check in checkInputTimes()

        obj.current_time = 10
        self.assertFalse(obj.checkInputTimes())

        obj.current_time = 0
        self.assertTrue(obj.checkInputTimes())

    @cpu_and_gpu
    def test_post_trigger_resets_inputs_changed(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs_changed = True
        obj.stream = MagicMock()
        obj.cuda_graph = None
        obj._target_device = MagicMock()

        obj.post_trigger()
        self.assertFalse(obj.inputs_changed)

    @cpu_and_gpu
    def test_post_trigger_raises_if_inputs_not_changed(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs_changed = False
        with self.assertRaises(RuntimeError):
            obj.post_trigger()

    # TODO this patch does not seem to be reverted correctly
    #      and subsequent tests involving streams fail.
    # @cpu_and_gpu
    # def test_device_stream_creates_and_reuses_stream(self, target_device_idx, xp):
    #     with patch.object(cp.cuda, "Stream", return_value="fake_stream") as mock_stream:
    #         s1 = BaseProcessingObj.device_stream(target_device_idx)
    #         s2 = BaseProcessingObj.device_stream(target_device_idx)
    #         self.assertEqual(s1, s2)
    #         mock_stream.assert_called_once()

    @cpu_and_gpu
    def test_check_ready_sets_inputs_changed_true(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.checkInputTimes = MagicMock(return_value=True)
        obj.prepare_trigger = MagicMock()

        result = obj.check_ready(42)
        self.assertTrue(result)
        self.assertTrue(obj.inputs_changed)
        obj.prepare_trigger.assert_called_once_with(42)

    @cpu_and_gpu
    def test_check_ready_sets_inputs_changed_false(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.checkInputTimes = MagicMock(return_value=False)

        result = obj.check_ready(42)
        self.assertFalse(result)
        self.assertFalse(obj.inputs_changed)

    @cpu_and_gpu
    def test_trigger_raises_if_inputs_not_changed(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs_changed = False
        with self.assertRaises(RuntimeError):
            obj.trigger()

    @cpu_and_gpu
    def test_trigger_calls_trigger_code_when_no_cuda_graph(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs_changed = True
        obj.trigger_code = MagicMock()
        obj.cuda_graph = None

        obj.trigger()
        obj.trigger_code.assert_called_once()

# --- CUDA STREAM MANAGEMENT TESTS ---

    @cpu_and_gpu
    def test_device_stream_reuses_existing_stream(self, target_device_idx, xp):
        # Pre-populate stream cache
        test_device_idx = 42
        BaseProcessingObj._streams[test_device_idx] = "cached_stream"
        result = BaseProcessingObj.device_stream(test_device_idx)
        self.assertEqual(result, "cached_stream")
        del BaseProcessingObj._streams[test_device_idx]

    # --- CUDA GRAPH CAPTURE TESTS ---

    @cpu_and_gpu
    def test_capture_stream_records_and_creates_graph(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.trigger_code = MagicMock()

        mock_stream = MagicMock()
        obj.stream = mock_stream

        mock_graph = MagicMock()
        mock_stream.end_capture.return_value = mock_graph

        obj.capture_stream()

        # trigger_code should be called twice: before and during capture
        self.assertEqual(obj.trigger_code.call_count, 2)
        # end_capture must set cuda_graph
        self.assertEqual(obj.cuda_graph, mock_graph)

    @cpu_and_gpu
    def test_capture_stream_with_no_trigger_code_error_handled(self, target_device_idx, xp):
        """If trigger_code raises, ensure end_capture is not called."""
        obj = BaseProcessingObj(target_device_idx=target_device_idx)

        obj.trigger_code = MagicMock(side_effect=RuntimeError("GPU failure"))
        mock_stream = MagicMock()
        obj.stream = mock_stream

        with self.assertRaises(RuntimeError):
            obj.capture_stream()

        mock_stream.end_capture.assert_not_called()

    # --- CUDA SYNCHRONIZATION TESTS ---

    @cpu_and_gpu
    def test_post_trigger_synchronizes_cuda_graph(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs_changed = True
        obj._target_device = MagicMock()
        obj.target_device_idx = 0  # Force execution of GPU code path
        obj.stream = MagicMock()
        obj.cuda_graph = "fake_graph"

        obj.post_trigger()

        # Device must be selected again
        obj._target_device.use.assert_called_once()
        # Stream must be synchronized because cuda_graph is set
        obj.stream.synchronize.assert_called_once()

    @cpu_and_gpu
    def test_post_trigger_skips_synchronize_if_no_graph(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs_changed = True
        obj._target_device = MagicMock()
        obj.target_device_idx = 0  # Force execution of GPU code path
        obj.stream = MagicMock()
        obj.cuda_graph = None

        obj.post_trigger()

        # Device still selected
        obj._target_device.use.assert_called_once()
        # No synchronization since there's no graph
        obj.stream.synchronize.assert_not_called()
