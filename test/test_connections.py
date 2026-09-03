import specula
specula.init(0)  # Default target device

import unittest
from unittest.mock import MagicMock, patch

from specula import np, cp
from specula import cpuArray

from specula.base_value import BaseValue
from specula.connections import InputList, InputValue, _InputItem, split_output

from test.specula_testlib import cpu_and_gpu


class TestSplitOutput(unittest.TestCase):

    def test_plain(self):
        output = split_output('control.out_comm')
        self.assertEqual(output.obj_name, 'control')
        self.assertEqual(output.output_key, 'out_comm')
        self.assertIsNone(output.input_name)
        self.assertEqual(output.delay, 0)

    def test_alias(self):
        output = split_output('comm-control.out_comm')
        self.assertEqual(output.obj_name, 'control')
        self.assertEqual(output.output_key, 'out_comm')
        self.assertEqual(output.input_name, 'comm')

    def test_alias_with_dot_is_allowed(self):
        # A dot in the alias (the part used as filename by DataStore/DataBuffer)
        # does not conflict with the '-' and '.' used as syntax separators,
        # because the alias is split off (on '-') before the object.output
        # part is split (on '.').
        output = split_output('sr.5000-slopec.out_slopes')
        self.assertEqual(output.input_name, 'sr.5000')
        self.assertEqual(output.obj_name, 'slopec')
        self.assertEqual(output.output_key, 'out_slopes')

    def test_delay_suffix(self):
        output = split_output('control.out_comm:1')
        self.assertEqual(output.delay, 1)

    def test_object_name_with_dot_raises(self):
        with self.assertRaises(ValueError):
            split_output('my.control.out_comm')

    def test_object_name_with_extra_dash_raises(self):
        with self.assertRaises(ValueError):
            split_output('comm-my-control.out_comm')

    def test_alias_with_dash_raises(self):
        with self.assertRaises(ValueError):
            split_output('my-alias-control.out_comm')

    def test_object_name_with_colon_raises(self):
        with self.assertRaises(ValueError):
            split_output('my:control.out_comm')

    def test_invalid_delay_suffix_raises(self):
        with self.assertRaises(ValueError):
            split_output('control.out_comm:notanumber')


class TestConnections(unittest.TestCase):
   
    @cpu_and_gpu
    def test_input_value_same_device(self, target_device_idx, xp):

        data = xp.arange(2)
        input_v = InputValue(type=BaseValue)

        output_v = BaseValue(value=data, target_device_idx=target_device_idx)
        input_v.set(output_v)

        result = input_v.get(target_device_idx=target_device_idx)
        assert(result.target_device_idx == target_device_idx)
        np.testing.assert_array_equal(cpuArray(data), cpuArray(result.value))

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "cupy not installed")
    def test_input_value_other_device(self, target_device_idx, xp):
            
        data = xp.arange(2)
        input_v = InputValue(type=BaseValue)

        output_v = BaseValue(value=data, target_device_idx=target_device_idx)
        input_v.set(output_v)

        if target_device_idx == 0:
            my_target = -1
        else:
            my_target = 0

        result = input_v.get(target_device_idx=my_target)
        assert(result.target_device_idx == my_target)
        np.testing.assert_array_equal(cpuArray(data), cpuArray(result.value))

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "cupy not installed")
    def test_input_value_transfer_does_not_allocate_a_new_object(self, target_device_idx, xp):

        data = xp.arange(2)
        input_v = InputValue(type=BaseValue)

        output_v = BaseValue(value=data, target_device_idx=target_device_idx)
        input_v.set(output_v)

        if target_device_idx == 0:
            my_target = -1
        else:
            my_target = 0

        result1 = input_v.get(target_device_idx=my_target)
        result2 = input_v.get(target_device_idx=my_target)
        assert(id(result1) == id(result2))

    @cpu_and_gpu
    def test_input_list_same_device(self, target_device_idx, xp):

        data1 = xp.arange(2)
        data2 = xp.arange(2)+2
        input_v = InputList(type=BaseValue)

        output1 = BaseValue(value=data1, target_device_idx=target_device_idx)
        output2 = BaseValue(value=data2, target_device_idx=target_device_idx)
        input_v.set([output1, output2])

        result = input_v.get(target_device_idx=target_device_idx)
        assert(result[0].target_device_idx == target_device_idx)
        assert(result[1].target_device_idx == target_device_idx)
        np.testing.assert_array_equal(cpuArray(data1), cpuArray(result[0].value))
        np.testing.assert_array_equal(cpuArray(data2), cpuArray(result[1].value))

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "cupy not installed")
    def test_input_list_other_device(self, target_device_idx, xp):

        data1 = xp.arange(2)
        data2 = xp.arange(2)+2
        input_v = InputList(type=BaseValue)

        output1 = BaseValue(value=data1, target_device_idx=target_device_idx)
        output2 = BaseValue(value=data2, target_device_idx=target_device_idx)
        input_v.set([output1, output2])

        if target_device_idx == 0:
            my_target = -1
        else:
            my_target = 0

        result = input_v.get(target_device_idx=my_target)

        assert(result[0].target_device_idx == my_target)
        assert(result[1].target_device_idx == my_target)
        np.testing.assert_array_equal(cpuArray(data1), cpuArray(result[0].value))
        np.testing.assert_array_equal(cpuArray(data2), cpuArray(result[1].value))

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "cupy not installed")
    def test_input_list_transfer_does_not_allocate_a_new_object(self, target_device_idx, xp):

        data1 = xp.arange(2)
        data2 = xp.arange(2)+2
        input_v = InputList(type=BaseValue)

        output1 = BaseValue(value=data1, target_device_idx=target_device_idx)
        output2 = BaseValue(value=data2, target_device_idx=target_device_idx)
        input_v.set([output1, output2])

        if target_device_idx == 0:
            my_target = -1
        else:
            my_target = 0

        result1 = input_v.get(target_device_idx=my_target)
        result2 = input_v.get(target_device_idx=my_target)
        assert(id(result1[0]) == id(result2[0]))
        assert(id(result1[1]) == id(result2[1]))

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "cupy not installed")
    def test_input_list_can_append(self, target_device_idx, xp):

        data1 = xp.arange(2)
        data2 = xp.arange(2)+2
        input_v = InputList(type=BaseValue)

        output1 = BaseValue(value=data1, target_device_idx=target_device_idx)
        output2 = BaseValue(value=data2, target_device_idx=target_device_idx)
        input_v.set([output1])
        input_v.append(output2)

        result = input_v.get(target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(data1), cpuArray(result[0].value))
        np.testing.assert_array_equal(cpuArray(data2), cpuArray(result[1].value))




class DummyValue:
    def __init__(self, val=None, xp_str="np"):
        self._value = val
        self.xp_str = xp_str
        self.xp = None
        self.generation_time = None

    def get_value(self):
        return self._value

    def set_value(self, val):
        self._value = val


class TestReceiveNewValue(unittest.TestCase):

    def setUp(self):
        self.item = _InputItem(
            type_=DummyValue,
            value=None,
            remote_rank=1,
            tag=10
        )
        self.item.logger = MagicMock()

    # ----------------------------------------
    # First receive (pickle path)
    # ----------------------------------------

    @patch("specula.connections.process_comm")
    @patch("specula.connections.np")
    @patch("specula.connections.cp")
    def test_first_receive_numpy(self, mock_cp, mock_np, mock_comm):
        """
        Should use recv() and assign np when xp_str != 'cp'
        """
        new_val = DummyValue(val=123, xp_str="np")
        mock_comm.recv.return_value = new_val

        result = self.item.receive_new_value(first_mpi_receive=True)

        self.assertEqual(result, new_val)
        self.assertEqual(result.xp, mock_np)
        mock_comm.recv.assert_called_once_with(source=1, tag=10)

    @patch("specula.connections.process_comm")
    @patch("specula.connections.np")
    @patch("specula.connections.cp")
    def test_first_receive_cupy(self, mock_cp, mock_np, mock_comm):
        """
        Should assign cp when xp_str == 'cp'
        """
        new_val = DummyValue(val=123, xp_str="cp")
        mock_comm.recv.return_value = new_val

        result = self.item.receive_new_value(first_mpi_receive=True)

        self.assertEqual(result.xp, mock_cp)

    # ----------------------------------------
    # Force pickle path when cloned_value is empty
    # ----------------------------------------

    @patch("specula.connections.process_comm")
    def test_buffer_path_falls_back_if_no_value(self, mock_comm):
        """
        If cloned_value.get_value() is None → use pickle path
        """
        self.item.cloned_value = DummyValue(val=None)

        new_val = DummyValue(val=42, xp_str="np")
        mock_comm.recv.return_value = new_val

        result = self.item.receive_new_value(first_mpi_receive=False)

        self.assertEqual(result, new_val)
        mock_comm.recv.assert_called_once_with(source=1, tag=10)

    # ----------------------------------------
    # Logging calls sanity check
    # ----------------------------------------

    @patch("specula.connections.process_comm")
    def test_logging_called(self, mock_comm):
        new_val = DummyValue(val=1, xp_str="np")
        mock_comm.recv.return_value = new_val

        self.item.receive_new_value(first_mpi_receive=True)

        self.assertTrue(self.item.logger.mpi_send_debug.called)

    # ----------------------------------------
    # Exception propagation
    # ----------------------------------------

    @patch("specula.connections.process_comm")
    def test_recv_exception_propagates(self, mock_comm):
        """
        No try/except → exception should propagate
        """
        mock_comm.recv.side_effect = RuntimeError("MPI failure")

        with self.assertRaises(RuntimeError):
            self.item.receive_new_value(first_mpi_receive=True)

