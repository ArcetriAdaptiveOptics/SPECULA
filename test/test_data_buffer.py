import os
import shutil
import yaml
import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.simul import Simul
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.data_buffer import DataBuffer
from specula.base_data_obj import BaseDataObj

import numpy as np
import unittest
from test.specula_testlib import cpu_and_gpu


class TestDataBuffer(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_data_buffer')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @cpu_and_gpu
    def test_data_buffer_basic(self, target_device_idx, xp):
        """Test basic DataBuffer functionality"""
        buffer_size = 3
        total_steps = 7  # Should emit 2 times (at step 3 and 6) + final

        params = {
            'main': {
                'class': 'SimulParams', 
                'root_dir': self.tmp_dir,
                'time_step': 0.1, 
                'total_time': total_steps * 0.1
            },
            'generator': {
                'class': 'WaveGenerator',
                'target_device_idx': target_device_idx, 
                'amp': 1, 
                'freq': 1  # Generate constant values for simplicity
            },
            'buffer': {
                'class': 'DataBuffer',
                'buffer_size': buffer_size,
                'inputs': {'input_list': ['gen-generator.output']}
            }
        }

        filename = os.path.join(self.tmp_dir, 'test_data_buffer.yaml')
        with open(filename, 'w') as outfile:
            yaml.dump(params, outfile)

        simul = Simul(filename)
        simul.run()

        # Access DataBuffer through simul
        buffer_obj = simul.objs['buffer']

        # Verify buffer exists and has correct structure
        self.assertIsNotNone(buffer_obj)
        self.assertEqual(buffer_obj.buffer_size, buffer_size)

        # Verify outputs were created
        self.assertIn('gen_buffered', buffer_obj.outputs)

        # Verify final output was emitted
        output_obj = buffer_obj.outputs['gen_buffered']
        self.assertIsNotNone(output_obj.value)

        # Last emission should contain remaining data (step 7)
        # Since we emit at step 3 and 6, last buffer should have 1 element
        final_data = output_obj.value
        self.assertEqual(len(final_data), 1)  # Only the last step

    @cpu_and_gpu
    def test_data_buffer_vs_datastore(self, target_device_idx, xp):
        """Compare DataBuffer with DataStore to verify consistency"""
        buffer_size = 10  # Large buffer to capture all data
        total_steps = 5

        params = {
            'main': {
                'class': 'SimulParams',
                'root_dir': self.tmp_dir,
                'time_step': 0.1, 
                'total_time': total_steps * 0.1
            },
            'generator': {
                'class': 'WaveGenerator',
                'target_device_idx': target_device_idx,
                'amp': 2,
                'freq': 0
            },
            'buffer': {
                'class': 'DataBuffer',
                'buffer_size': buffer_size,
                'inputs': {'input_list': ['gen-generator.output']}
            },
            'store': {
                'class': 'DataStore',
                'store_dir': self.tmp_dir,
                'inputs': {'input_list': ['gen-generator.output']}
            }
        }

        filename = os.path.join(self.tmp_dir, 'test_buffer_vs_store.yaml')
        with open(filename, 'w') as outfile:
            yaml.dump(params, outfile)

        simul = Simul(filename)
        simul.run()

        # Compare buffer data with store data
        buffer_obj = simul.objs['buffer']
        store_obj = simul.objs['store']

        # Data from buffer (after final emission)
        buffer_data = buffer_obj.outputs['gen_buffered'].value

        # Data from store
        store_data = np.array(list(store_obj.storage['gen'].values()))

        # Should be identical (buffer captured all data)
        np.testing.assert_array_equal(cpuArray(buffer_data), cpuArray(store_data))

    @cpu_and_gpu
    def test_data_buffer_multiple_emissions(self, target_device_idx, xp):
        """Test multiple buffer emissions by manually stepping through simulation"""
        buffer_size = 2
        total_steps = 5

        # Create generator
        generator = WaveGenerator(target_device_idx=target_device_idx, amp=1, freq=0)
        generator.setup()

        # Create buffer with manual input setup
        buffer = DataBuffer(buffer_size=buffer_size)
        buffer.target_device_idx = target_device_idx

        # Manually create input for buffer (simulate what simul.py does)
        from specula.connections import InputValue
        from specula.base_value import BaseValue
        buffer.inputs = {}
        buffer.inputs['gen'] = InputValue(type=BaseValue)
        # Connect generator output to buffer input
        buffer.inputs['gen'].set(generator.outputs['output'])
        buffer.setOutputs()
        buffer.setup()

        # Track emissions manually
        emissions = []
        original_emit = buffer.emit_buffered_data

        def track_emit():
            emissions.append({
                'time': buffer.current_time,
                'step_counter': buffer.step_counter,
                'data_count': len(buffer.storage['gen'])
            })
            original_emit()

        buffer.emit_buffered_data = track_emit


        # Simulate steps manually
        for step in range(total_steps):
            current_time = step * 0.1

            # Update generator
            generator.current_time = current_time
            generator.check_ready(current_time)
            generator.trigger()
            generator.post_trigger()

            # Update buffer
            buffer.current_time = current_time
            buffer.check_ready(current_time)
            buffer.trigger()
            buffer.post_trigger()

        # Finalize buffer to emit remaining data
        buffer.finalize()

        # Verify emissions
        # With buffer_size=2 and total_steps=5:
        # - Emission at step 2 (steps 0,1)
        # - Emission at step 4 (steps 2,3)
        # - Final emission (step 4 remaining)
        expected_emission_count = 3
        self.assertEqual(len(emissions), expected_emission_count)

        # Check emission data counts
        expected_counts = [2, 2, 1]  # First two full, last one partial
        actual_counts = [e['data_count'] for e in emissions]
        self.assertEqual(actual_counts, expected_counts)

        # Check final buffer state
        self.assertEqual(len(buffer.storage['gen']), 0)  # Should be reset
        self.assertEqual(buffer.step_counter, 0)  # Should be reset

        # Verify final output exists
        self.assertIn('gen_buffered', buffer.outputs)
        final_output = buffer.outputs['gen_buffered'].value
        self.assertEqual(len(final_output), 1)  # Last emission


    @cpu_and_gpu
    def test_data_buffer_fails_early(self, target_device_idx, xp):
        """Test that DataBuffer fails during Setup() if a class without get_value() is set as an input"""
        buffer_size = 2


        # Create buffer with manual input setup
        buffer = DataBuffer(buffer_size=buffer_size)
        buffer.target_device_idx = target_device_idx

        data = BaseDataObj()

        # Manually create input for buffer (simulate what simul.py does)
        from specula.connections import InputValue
        buffer.inputs['gen'] = InputValue(type=BaseDataObj)
        buffer.inputs['gen'].set(data)

        with self.assertRaises(TypeError):
            buffer.setup()

