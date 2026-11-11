import specula
specula.init(0)  # Default target device

import numpy as np
import unittest
from unittest.mock import MagicMock

from specula import cp, cpuArray
from specula.base_value import BaseValue
from specula.connections import InputList, InputValue
from specula.base_processing_obj import BaseProcessingObj
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.log import MPI_DBG_LEVEL
from iteras.accelerator import Accelerator
from iteras.data_objects.accelerator import Accelerator as AcceleratorOptions

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
    def test_check_input_times_returns_true_for_invalid_optional_inputs(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)

        obj.inputs['test1'] = InputValue(type=BaseValue, optional=True)
        obj.inputs['test2'] = InputValue(type=BaseValue, optional=True)
        self.assertTrue(obj.checkInputTimes())

    @cpu_and_gpu
    def test_check_input_times_returns_false_for_valid_optional_inputs(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)

        obj.inputs['test1'] = InputValue(type=BaseValue, optional=True)
        obj.inputs['test2'] = InputValue(type=BaseValue, optional=True)

        value = BaseValue()
        obj.inputs['test1'].set(value)
        value.generation_time = 1 # Simulate non-refreshed inputs

        obj.current_time = 2
        self.assertFalse(obj.checkInputTimes())

    @cpu_and_gpu
    def test_get_all_inputs_formats_inputs_only_when_mpi_debug_enabled(self, target_device_idx, xp):
        '''
        Inputs must only be converted to strings when MPI debug logging is
        enabled, since this is expensive (GPU arrays are copied to the host)
        '''
        n_str_calls = []

        class CountingValue(BaseValue):
            def __str__(self):
                n_str_calls.append(1)
                return 'CountingValue'

        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs['test'] = InputValue(type=BaseValue)
        obj.inputs['test'].set(CountingValue(target_device_idx=target_device_idx))

        # Default: the logger's own level is NOTSET, effective level is higher
        obj.get_all_inputs()
        self.assertEqual(len(n_str_calls), 0)

        orig_level = obj.logger.logger.level
        try:
            obj.logger.logger.setLevel(MPI_DBG_LEVEL)
            obj.get_all_inputs()
            self.assertGreater(len(n_str_calls), 0)
        finally:
            obj.logger.logger.setLevel(orig_level)

    @cpu_and_gpu
    def test_post_trigger_resets_inputs_changed(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs_changed = True
        obj._target_device = MagicMock()

        obj.post_trigger()
        self.assertFalse(obj.inputs_changed)

    @cpu_and_gpu
    def test_post_trigger_raises_if_inputs_not_changed(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs_changed = False
        with self.assertRaises(RuntimeError):
            obj.post_trigger()

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
    def test_trigger_calls_trigger_code_when_not_accelerated(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs_changed = True
        obj.trigger_code = MagicMock()

        obj.trigger()
        obj.trigger_code.assert_called_once()

# --- CUDA STREAM MANAGEMENT TESTS ---

    # --- CUDA GRAPH CAPTURE TESTS (through the accelerator) ---

    @staticmethod
    def _accelerate(obj):  # pragma: gpu
        '''Accelerate *obj* as Simul does, returning its GraphRunner'''
        obj.can_accelerate = lambda: True
        Accelerator(AcceleratorOptions()).apply([obj])
        return obj._accel

    @staticmethod
    def _trigger(obj):  # pragma: gpu
        obj.inputs_changed = True
        obj.trigger()
        obj.post_trigger()
        cp.cuda.Device(0).synchronize()

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_graph_temporaries_are_not_reused(self):
        '''
        Temporary arrays allocated by trigger_code() during the capture are
        released when it returns, but the CUDA graph keeps using them at each
        launch: they must not be given to other arrays, which the graph would
        overwrite, nor freed, which would make the launch fail.
        '''
        n = 1024 * 1024

        class TempObj(BaseProcessingObj):
            def __init__(self):
                super().__init__(target_device_idx=0)
                self.inp = cp.arange(n, dtype=cp.float32)
                self.out = cp.zeros(n, dtype=cp.float32)

            def trigger_code(self):
                self.out[:] = self.inp * 2 + 1  # allocates temporaries

        def run(obj):
            obj.out[:] = 0
            self._trigger(obj)
            np.testing.assert_array_equal(cpuArray(obj.out), np.arange(n) * 2 + 1)

        obj = TempObj()
        runner = self._accelerate(obj)
        run(obj)        # warm-up
        run(obj)        # capture
        self.assertIsNotNone(runner.graph)

        others = [cp.full(n, -1, dtype=cp.float32) for _ in range(4)]
        run(obj)
        for other in others:
            np.testing.assert_array_equal(cpuArray(other), -1)

        del others
        cp.get_default_memory_pool().free_all_blocks()
        run(obj)

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_graph_mempool_is_freed_with_the_object(self):
        '''
        A deleted cupy MemoryPool does not release its memory to the device:
        the memory of the CUDA graph pool must be freed when the object is
        garbage collected, otherwise it is lost until the process exits.
        '''
        import gc
        n = 1024 * 1024

        class TempObj(BaseProcessingObj):
            def __init__(self):
                super().__init__(target_device_idx=0)
                self.inp = cp.arange(n, dtype=cp.float32)
                self.out = cp.zeros(n, dtype=cp.float32)

            def trigger_code(self):
                self.out[:] = self.inp * 2 + 1  # allocates temporaries

        obj = TempObj()
        runner = self._accelerate(obj)
        self._trigger(obj)
        self._trigger(obj)
        pool = runner.pool
        self.assertGreater(pool.total_bytes(), 0)

        del obj, runner
        gc.collect()
        self.assertEqual(pool.total_bytes(), 0)

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_fft_plans_are_not_in_the_graph_mempool(self):
        '''
        The cuFFT plans created by the first execution of trigger_code() are
        cached by cupy and shared with other objects: their work areas must not
        come from the CUDA graph pool, which is freed with the object. The
        first triggers run as Python code, before the capture, so after the
        capture the pool holds no memory in use, since the temporaries of
        trigger_code() are released when it returns.
        '''
        # An FFT size used nowhere else, so that the plan is created here
        shape = (37, 41)

        class FftObj(BaseProcessingObj):
            def __init__(self):
                super().__init__(target_device_idx=0)
                self.inp = cp.asarray(np.random.default_rng(1).standard_normal(shape))
                self.out = cp.zeros(shape)

            def trigger_code(self):
                self.out[:] = cp.abs(cp.fft.fft2(self.inp))

        obj = FftObj()
        runner = self._accelerate(obj)
        self._trigger(obj)
        self._trigger(obj)
        self.assertIsNotNone(runner.graph)
        with cp.cuda.Device(0):
            self.assertEqual(runner.pool.used_bytes(), 0)

        obj.out[:] = 0
        self._trigger(obj)
        np.testing.assert_allclose(cpuArray(obj.out), np.abs(np.fft.fft2(cpuArray(obj.inp))))

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_trigger_code_runs_once_per_trigger_when_capturing(self):
        '''
        With a stateful trigger_code() (like an integrator), each trigger() must
        advance the state exactly once, also in the steps where the CUDA graph
        is captured: the first capture, and after invalidate_graph().
        '''
        class Integrator(BaseProcessingObj):
            def __init__(self):
                super().__init__(target_device_idx=0)
                self.state = cp.zeros(4, dtype=cp.float32)

            def trigger_code(self):
                self.state += 1

        def run(obj):
            self._trigger(obj)
            return float(obj.state[0])

        obj = Integrator()
        runner = self._accelerate(obj)
        self.assertEqual(run(obj), 1.0)     # Python code
        self.assertIsNone(runner.graph)
        self.assertEqual(run(obj), 2.0)     # captured here
        self.assertIsNotNone(runner.graph)
        self.assertEqual(run(obj), 3.0)     # graph launch
        obj.invalidate_graph()
        self.assertEqual(run(obj), 4.0)     # captured again
        self.assertEqual(run(obj), 5.0)     # graph launch

    @cpu_and_gpu
    def test_invalidate_graph_without_stream_does_nothing(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.trigger_code = MagicMock()
        obj.invalidate_graph()
        obj.inputs_changed = True
        obj.trigger()
        obj.trigger_code.assert_called_once()

    @cpu_and_gpu
    def test_not_accelerated_by_default(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        self.assertFalse(obj.can_accelerate())
        self.assertEqual(obj.internal_objects(), [])

    # --- CUDA SYNCHRONIZATION TESTS ---

    @cpu_and_gpu
    def test_post_trigger_does_not_synchronize(self, target_device_idx, xp):
        obj = BaseProcessingObj(target_device_idx=target_device_idx)
        obj.inputs_changed = True
        obj._target_device = MagicMock()
        obj.target_device_idx = 0  # Force execution of GPU code path

        obj.post_trigger()

        # Device still selected, and no synchronization
        obj._target_device.use.assert_called_once()
        obj._target_device.synchronize.assert_not_called()

    def test_check_input_names_raises_if_input_missing(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.input_names = MagicMock(return_value={"input1": InputDesc(BaseValue, 'Some description')})
        obj.inputs = {}
        with self.assertRaises(ValueError):
            obj.check_input_names()

    def test_check_input_names_raises_if_input_wrong_type(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.input_names = MagicMock(return_value={"input1": InputDesc(BaseValue, 'Some description')})
        obj.inputs = {"input1": "not an InputValue"}
        with self.assertRaises(TypeError):
            obj.check_input_names()
    
    def test_check_input_names_raises_if_input_wrong_type_in_list(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.input_names = MagicMock(return_value={"input1": InputDesc(BaseValue, 'Some description')})
        obj.inputs = {"input1": ["not an InputValue"]}
        with self.assertRaises(TypeError):
            obj.check_input_names()

    def test_check_input_names_passes_for_valid_inputs(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.input_names = MagicMock(return_value={"input1": (BaseValue, 'desc1'), "input2": (BaseValue, 'desc2')})
        obj.inputs = {"input1": InputValue(type=BaseValue), "input2": InputList(type=BaseValue)}
        try:
            obj.check_input_names()  # Should not raise
        except Exception as e:
            self.fail(f"check_input_names raised an exception unexpectedly: {e}")

    def test_check_output_names_raises_if_output_missing(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.output_names = MagicMock(return_value={"output1": BaseValue})
        obj.outputs = {}
        with self.assertRaises(ValueError):
            obj.check_output_names()
    
    def test_check_output_names_passes_for_valid_outputs(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.output_names = MagicMock(return_value={"output1": (BaseValue, 'desc1'), "output2": (BaseValue, 'desc2')})
        obj.outputs = {"output1": BaseValue(), "output2": BaseValue()}
        try:
            obj.check_output_names()  # Should not raise
        except Exception as e:
            self.fail(f"check_output_names raised an exception unexpectedly: {e}")

    def test_sanity_check_calls_input_and_output_checks(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.check_input_names = MagicMock()
        obj.check_output_names = MagicMock()

        try:
            obj.sanity_check()  # Should not raise
        except Exception as e:
            self.fail(f"sanity_check raised an exception unexpectedly: {e}")

        obj.check_input_names.assert_called_once()
        obj.check_output_names.assert_called_once()

    def test_check_output_names_supports_placeholder_pattern_multiple_outputs(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.output_names = MagicMock(return_value={
            "out_modes_{sensor_idx}": (BaseValue, "Dynamic outputs"),
        })
        obj.outputs = {
            "out_modes_0": BaseValue(),
            "out_modes_1": BaseValue(),
        }

        obj.check_output_names()  # Should not raise

    def test_check_output_names_supports_placeholder_pattern(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.output_names = MagicMock(return_value={
            "out_modes_{sensor_idx}": (BaseValue, "Dynamic outputs"),
        })
        obj.outputs = {
            "out_modes_0": BaseValue(),
            "out_modes_3": BaseValue(),
        }

        obj.check_output_names()  # Should not raise

    def test_check_input_names_supports_placeholder_pattern(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.input_names = MagicMock(return_value={
            "in_sensor_{idx}": (BaseValue, "Per-sensor input (optional)"),
        })
        obj.inputs = {
            "in_sensor_0": InputValue(type=BaseValue),
            "in_sensor_1": InputValue(type=BaseValue),
        }

        obj.check_input_names()  # Should not raise

    def test_check_output_names_pattern_missing_raises(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.output_names = MagicMock(return_value={
            "out_modes_{sensor_idx}": (BaseValue, "Dynamic outputs"),
        })
        obj.outputs = {
            "out_other": BaseValue(),
        }

        with self.assertRaises(ValueError):
            obj.check_output_names()

class TestSendAndHooks(unittest.TestCase):
    '''MPI sends of outputs (with a mocked communicator), and the default class hooks'''

    def test_send_remote_output_of_a_buffer(self):
        from unittest import mock
        import iteras
        from iteras import mpi_progress
        from iteras.log import MPI_SEND_DBG_LEVEL
        obj = BaseProcessingObj(target_device_idx=-1)
        obj.current_time = 5
        item = BaseValue(value=np.arange(3.0), target_device_idx=-1)
        item.generation_time = 5
        obj.sent_valid[10] = True       # not the first send: the data only, with a buffer
        obj.logger.setLevel(MPI_SEND_DBG_LEVEL)
        comm = MagicMock()
        with mock.patch.object(iteras, 'process_comm', comm), \
                mock.patch.object(mpi_progress, 'isend') as isend:
            obj.send_remote_output(item, 1, 10, first_mpi_send=False)
        buffer, dest, tag, step = isend.call_args.args
        np.testing.assert_array_equal(buffer, [0, 1, 2])
        self.assertIsNot(buffer, item.value)          # a copy: the object goes on
        self.assertEqual((dest, tag, step), (1, 10, 5))
        comm.ibsend.assert_called_once_with(5, dest=1, tag=11)

    def test_default_hooks(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        self.assertFalse(obj.can_accelerate())          # no own CUDA graph
        params = {'n_slices': 4}
        self.assertIs(BaseProcessingObj.single_device_params(params), params)
