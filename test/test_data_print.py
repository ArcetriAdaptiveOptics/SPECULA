import unittest
from io import StringIO
from unittest.mock import patch

import specula
specula.init(0)  # Default target device

import numpy as np
from specula.base_value import BaseValue
from specula.processing_objects.data_print import DataPrint
from test.specula_testlib import cpu_and_gpu


class TestDataPrint(unittest.TestCase):
    """Test DataPrint class for printing array values"""

    @cpu_and_gpu
    def test_data_print_basic(self, target_device_idx, xp):
        """Test basic DataPrint functionality with 1D array"""
        print_dt = 0.1
        printer = DataPrint(print_dt=print_dt, target_device_idx=target_device_idx)

        # Create test data
        data = BaseValue(target_device_idx=target_device_idx)
        data.value = xp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        printer.inputs['in_value'].set(data)
        printer.setup()

        # Capture print output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # First trigger - should print
            time = printer.seconds_to_t(0.1)
            data.generation_time = time
            printer.check_ready(time)
            printer.trigger()

            output = mock_stdout.getvalue()
            self.assertIn('t=', output)
            self.assertIn('[', output)

    @cpu_and_gpu
    def test_data_print_with_range_slice(self, target_device_idx, xp):
        """Test DataPrint with range_slice parameter"""
        printer = DataPrint(
            print_dt=0.1,
            range_slice=(0, 3),
            target_device_idx=target_device_idx
        )

        data = BaseValue(target_device_idx=target_device_idx)
        data.value = xp.arange(10)
        printer.inputs['in_value'].set(data)
        printer.setup()

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            time = printer.seconds_to_t(1)
            data.generation_time = time
            printer.check_ready(time)
            printer.trigger()

            output = mock_stdout.getvalue()
            # Should only print first 3 elements
            self.assertIn('0.0000', output)
            self.assertIn('1.0000', output)
            self.assertIn('2.0000', output)
            self.assertNotIn('3.0000', output)

    @cpu_and_gpu
    def test_data_print_with_prefix(self, target_device_idx, xp):
        """Test DataPrint with custom prefix"""
        prefix = "Test Data"
        printer = DataPrint(
            print_dt=0.1,
            prefix=prefix,
            target_device_idx=target_device_idx
        )

        data = BaseValue(target_device_idx=target_device_idx)
        data.value = xp.array([1.0])
        printer.inputs['in_value'].set(data)
        printer.setup()

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            time = printer.seconds_to_t(1)
            data.generation_time = time
            printer.check_ready(time)
            printer.trigger()

            output = mock_stdout.getvalue()
            self.assertIn(prefix, output)

    @cpu_and_gpu
    def test_data_print_format_string(self, target_device_idx, xp):
        """Test DataPrint with custom format string"""
        printer = DataPrint(
            print_dt=0.1,
            format_str='.2f',
            target_device_idx=target_device_idx
        )

        data = BaseValue(target_device_idx=target_device_idx)
        data.value = xp.array([1.23456])
        printer.inputs['in_value'].set(data)
        printer.setup()

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            time = printer.seconds_to_t(1)
            data.generation_time = time
            printer.check_ready(time)
            printer.trigger()

            output = mock_stdout.getvalue()
            self.assertIn('1.23', output)
            self.assertNotIn('1.23456', output)

    @cpu_and_gpu
    def test_data_print_timing(self, target_device_idx, xp):
        """Test that DataPrint respects print_dt interval"""
        print_dt = 0.5
        printer = DataPrint(print_dt=print_dt, target_device_idx=target_device_idx)

        data = BaseValue(target_device_idx=target_device_idx)
        data.value = xp.array([1.0])
        printer.inputs['in_value'].set(data)
        printer.setup()

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # First trigger at t=0.1 - should print (first trigger always prints)
            time = printer.seconds_to_t(0.1)
            data.generation_time = time
            printer.check_ready(time)
            printer.current_time = time
            printer.trigger()
            output1 = mock_stdout.getvalue()
            self.assertNotEqual(output1, '')

            # Second trigger at t=0.2 - should NOT print (too soon)
            mock_stdout.truncate(0)
            mock_stdout.seek(0)
            time = printer.seconds_to_t(0.2)
            data.generation_time = time
            printer.check_ready(time)
            printer.trigger()
            output2 = mock_stdout.getvalue()
            self.assertEqual(output2, '')

            # Third trigger at t=1.0 - should print (>= print_dt elapsed)
            time = printer.seconds_to_t(1.0)
            data.generation_time = time
            printer.check_ready(time)
            printer.trigger()
            output3 = mock_stdout.getvalue()
            self.assertNotEqual(output3, '')

    @cpu_and_gpu
    def test_data_print_2d_array(self, target_device_idx, xp):
        """Test DataPrint with 2D array"""
        printer = DataPrint(print_dt=0.1, target_device_idx=target_device_idx)

        data = BaseValue(target_device_idx=target_device_idx)
        data.value = xp.ones((5, 5))
        printer.inputs['in_value'].set(data)
        printer.setup()

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            time = printer.seconds_to_t(1)
            data.generation_time = time
            printer.check_ready(time)
            printer.trigger()

            output = mock_stdout.getvalue()
            self.assertIn('shape=', output)
            self.assertIn('(5, 5)', output)

    @cpu_and_gpu
    def test_data_print_scalar(self, target_device_idx, xp):
        """Test DataPrint with scalar value"""
        printer = DataPrint(print_dt=0.1, target_device_idx=target_device_idx)

        data = BaseValue(target_device_idx=target_device_idx)
        data.value = xp.array(42.5)
        printer.inputs['in_value'].set(data)
        printer.setup()

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            time = printer.seconds_to_t(1)
            data.generation_time = time
            printer.check_ready(time)
            printer.trigger()

            output = mock_stdout.getvalue()
            self.assertIn('42.5', output)
            self.assertNotIn('[', output)  # Scalars shouldn't have brackets

    @cpu_and_gpu
    def test_data_print_every_other_element(self, target_device_idx, xp):
        """Test DataPrint with slice selecting every other element"""
        printer = DataPrint(
            print_dt=15,
            range_slice=(None, None, 2),
            target_device_idx=target_device_idx
        )

        data = BaseValue(target_device_idx=target_device_idx)
        data.value = xp.arange(10)
        printer.inputs['in_value'].set(data)
        printer.setup()

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            time = printer.seconds_to_t(15)
            data.generation_time = time
            printer.check_ready(time)
            printer.trigger()

            output = mock_stdout.getvalue()
            # Should print 0, 2, 4, 6, 8
            self.assertIn('0.0000', output)
            self.assertIn('2.0000', output)
            self.assertIn('4.0000', output)
            # But not 1, 3, 5, 7, 9
            self.assertNotIn('1.0000', output)
            self.assertNotIn('3.0000', output)