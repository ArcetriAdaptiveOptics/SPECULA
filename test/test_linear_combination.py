import specula
specula.init(0)

import unittest
from specula import cpuArray, np
from specula.base_value import BaseValue
from specula.processing_objects.linear_combination import LinearCombination
from test.specula_testlib import cpu_and_gpu
from specula.data_objects.simul_params import SimulParams

class TestLinearCombination(unittest.TestCase):

    def setUp(self):
        self.simul_params = SimulParams(pixel_pupil=10, pixel_pitch=1.0, time_step=1)

    @cpu_and_gpu
    def test_basic_combination_no_focus_no_lift(self, target_device_idx, xp):
        '''Test basic combination without focus and lift.'''
        # LGS and NGS only
        lgs = BaseValue(value=xp.array([10., 20., 30., 40., 50.]),
                        target_device_idx=target_device_idx)
        ngs = BaseValue(value=xp.array([1., 2., 3., 4., 5.]),
                        target_device_idx=target_device_idx)
        vectors = [lgs, ngs]
        lc = LinearCombination(self.simul_params,
                               no_focus=True,
                               no_lift=True,
                               target_device_idx=target_device_idx)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()
        lc.trigger_code()
        out = cpuArray(lc.outputs['out_vector'].value)
        # TIP/TILT copied from NGS
        assert out[0] == 1.0
        assert out[1] == 2.0
        # Rest unchanged
        assert out[2] == 30.0

    @cpu_and_gpu
    def test_combination_with_focus(self, target_device_idx, xp):
        '''Test combination with focus.'''
        lgs = BaseValue(value=xp.array([10., 20., 30., 40., 50.]),
                        target_device_idx=target_device_idx)
        focus = BaseValue(value=xp.array([99.]),
                          target_device_idx=target_device_idx)
        ngs = BaseValue(value=xp.array([1., 2., 3., 4., 5.]),
                        target_device_idx=target_device_idx)
        vectors = [lgs, focus, ngs]
        lc = LinearCombination(self.simul_params,
                               no_focus=False,
                               no_lift=True,
                               target_device_idx=target_device_idx)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()
        lc.trigger_code()
        out = cpuArray(lc.outputs['out_vector'].value)
        # TIP/TILT copied from NGS
        assert out[0] == 1.0
        assert out[1] == 2.0
        # Focus copied from focus
        assert out[2] == 99.0

    @cpu_and_gpu
    def test_combination_with_lift(self, target_device_idx, xp):
        '''Test combination with lift.'''
        lgs = BaseValue(value=xp.array([10., 20., 30., 40., 50.]),
                        target_device_idx=target_device_idx)
        lift = BaseValue(value=xp.array([77.]),
                         target_device_idx=target_device_idx)
        ngs = BaseValue(value=xp.array([1., 2., 3., 4., 5.]),
                        target_device_idx=target_device_idx)
        vectors = [lgs, lift, ngs]
        lc = LinearCombination(self.simul_params,
                               no_focus=True,
                               no_lift=False,
                               target_device_idx=target_device_idx)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()
        lc.trigger_code()
        out = cpuArray(lc.outputs['out_vector'].value)
        # TIP/TILT copied from NGS
        assert out[0] == 1.0
        assert out[1] == 2.0
        # Lift is appended at the end
        assert out[-1] == 77.0

    @cpu_and_gpu
    def test_combination_with_focus_and_lift(self, target_device_idx, xp):
        '''Test combination with focus and lift.'''
        lgs = BaseValue(value=xp.array([10., 20., 30., 40., 50.]),
                        target_device_idx=target_device_idx)
        focus = BaseValue(value=xp.array([99.]),
                          target_device_idx=target_device_idx)
        lift = BaseValue(value=xp.array([77.]),
                         target_device_idx=target_device_idx)
        ngs = BaseValue(value=xp.array([1., 2., 3., 4., 5.]),
                        target_device_idx=target_device_idx)
        vectors = [lgs, focus, lift, ngs]
        lc = LinearCombination(self.simul_params,
                               no_focus=False,
                               no_lift=False,
                               target_device_idx=target_device_idx)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()
        lc.trigger_code()
        out = cpuArray(lc.outputs['out_vector'].value)
        # TIP/TILT copied from NGS
        assert out[0] == 1.0
        assert out[1] == 2.0
        # Focus copied from focus
        assert out[2] == 99.0
        # Lift is appended at the end
        assert out[-1] == 77.0

    @cpu_and_gpu
    def test_plate_scale_idx(self, target_device_idx, xp):
        '''Test that plate_scale_idx works correctly.'''
        lgs = BaseValue(value=xp.array([10., 20., 30., 40., 50., 60., 70.]),
                        target_device_idx=target_device_idx)
        focus = BaseValue(value=xp.array([99.]),
                          target_device_idx=target_device_idx)
        ngs = BaseValue(value=xp.array([1., 2., 3., 4., 5.]),
                        target_device_idx=target_device_idx)
        vectors = [lgs, focus, ngs]
        lc = LinearCombination(self.simul_params,
                               no_focus=False,
                               no_lift=True,
                               plate_scale_idx=3,
                               target_device_idx=target_device_idx)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()
        lc.trigger_code()
        out = cpuArray(lc.outputs['out_vector'].value)
        # Check that the plate_scale_idx block is overwritten by ngs[2:]
        np.testing.assert_array_equal(out[3:6], cpuArray(ngs.value[2:5]))

    @cpu_and_gpu
    def test_invalid_input_vector_combinations(self, target_device_idx, xp):
        '''Test that invalid input vector combinations raise errors.'''
        lgs = BaseValue(value=xp.array([1., 2., 3.]), target_device_idx=target_device_idx)
        focus = BaseValue(value=xp.array([4.]), target_device_idx=target_device_idx)
        lift = BaseValue(value=xp.array([5.]), target_device_idx=target_device_idx)
        ngs = BaseValue(value=xp.array([6., 7., 8.]), target_device_idx=target_device_idx)

        # Case 1: 4 inputs but one flag True (not valid)
        vectors = [lgs, focus, lift, ngs]
        lc = LinearCombination(self.simul_params, no_focus=True, no_lift=False, target_device_idx=target_device_idx)
        lc.inputs['in_vectors_list'].set(vectors)
        with self.assertRaises(ValueError):
            lc.setup()

        # Case 2: 3 inputs but both flags True (not valid)
        vectors = [lgs, focus, ngs]
        lc = LinearCombination(self.simul_params, no_focus=True, no_lift=True, target_device_idx=target_device_idx)
        lc.inputs['in_vectors_list'].set(vectors)
        with self.assertRaises(ValueError):
            lc.setup()

        # Case 3: 2 inputs but one flag False (not valid)
        vectors = [lgs, ngs]
        lc = LinearCombination(self.simul_params, no_focus=False, no_lift=True, target_device_idx=target_device_idx)
        lc.inputs['in_vectors_list'].set(vectors)
        with self.assertRaises(ValueError):
            lc.setup()

    @cpu_and_gpu
    def test_input_vectors_not_modified(self, target_device_idx, xp):
        '''Test that input vectors are not modified by linear combination.'''
        # Create input vectors with specific values
        lgs_original = xp.array([10., 20., 30., 40., 50.])
        focus_original = xp.array([99.])
        lift_original = xp.array([77.])
        ngs_original = xp.array([1., 2., 3., 4., 5.])

        # Create BaseValue objects
        lgs = BaseValue(value=lgs_original.copy(), target_device_idx=target_device_idx)
        focus = BaseValue(value=focus_original.copy(), target_device_idx=target_device_idx)
        lift = BaseValue(value=lift_original.copy(), target_device_idx=target_device_idx)
        ngs = BaseValue(value=ngs_original.copy(), target_device_idx=target_device_idx)

        vectors = [lgs, focus, lift, ngs]

        # Create LinearCombination with all features enabled
        lc = LinearCombination(self.simul_params,
                            no_focus=False,
                            no_lift=False,
                            target_device_idx=target_device_idx)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()

        # Store original values for comparison
        lgs_before = cpuArray(lgs.value.copy())
        focus_before = cpuArray(focus.value.copy())
        lift_before = cpuArray(lift.value.copy())
        ngs_before = cpuArray(ngs.value.copy())

        # Execute trigger_code (this should not modify input vectors)
        lc.trigger_code()

        # Verify input vectors are unchanged
        np.testing.assert_array_equal(cpuArray(lgs.value), lgs_before,
                                    err_msg="LGS input vector was modified")
        np.testing.assert_array_equal(cpuArray(focus.value), focus_before,
                                    err_msg="Focus input vector was modified")
        np.testing.assert_array_equal(cpuArray(lift.value), lift_before,
                                    err_msg="Lift input vector was modified")
        np.testing.assert_array_equal(cpuArray(ngs.value), ngs_before,
                                    err_msg="NGS input vector was modified")

        # Verify the output is correct (different from inputs)
        out = cpuArray(lc.outputs['out_vector'].value)

        # TIP/TILT should come from NGS
        self.assertEqual(out[0], 1.0)
        self.assertEqual(out[1], 2.0)
        # Focus should come from focus vector
        self.assertEqual(out[2], 99.0)
        # Lift should be appended
        self.assertEqual(out[-1], 77.0)

        # Verify that output is different from original lgs (proving it was modified)
        self.assertNotEqual(out[0], lgs_before[0])  # TIP changed from 10 to 1
        self.assertNotEqual(out[1], lgs_before[1])  # TILT changed from 20 to 2
        self.assertNotEqual(out[2], lgs_before[2])  # Focus changed from 30 to 99

    @cpu_and_gpu
    def test_copy_efficiency_multiple_calls(self, target_device_idx, xp):
        '''Test that multiple calls don't accumulate modifications to inputs.'''
        # Create input vectors
        lgs = BaseValue(value=xp.array([10., 20., 30., 40., 50.]),
                        target_device_idx=target_device_idx)
        ngs = BaseValue(value=xp.array([1., 2., 3., 4., 5.]),
                        target_device_idx=target_device_idx)
        vectors = [lgs, ngs]

        lc = LinearCombination(self.simul_params,
                            no_focus=True,
                            no_lift=True,
                            target_device_idx=target_device_idx)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()

        # Store original values
        lgs_original = cpuArray(lgs.value.copy())
        ngs_original = cpuArray(ngs.value.copy())

        # Call trigger_code multiple times
        for i in range(5):
            lc.trigger_code()

            # Verify inputs remain unchanged after each call
            np.testing.assert_array_equal(cpuArray(lgs.value), lgs_original,
                                        err_msg=f"LGS modified after call {i+1}")
            np.testing.assert_array_equal(cpuArray(ngs.value), ngs_original,
                                        err_msg=f"NGS modified after call {i+1}")

            # Verify output is consistent
            out = cpuArray(lc.outputs['out_vector'].value)
            self.assertEqual(out[0], 1.0, f"TIP inconsistent at call {i+1}")
            self.assertEqual(out[1], 2.0, f"TILT inconsistent at call {i+1}")
            self.assertEqual(out[2], 30.0, f"Mode 2 inconsistent at call {i+1}")

    @cpu_and_gpu
    def test_reference_vs_copy_behavior(self, target_device_idx, xp):
        '''Test that demonstrates the difference between reference and copy behavior.'''
        # Create a simple test case
        original_array = xp.array([100., 200., 300.])

        # Test 1: Reference behavior (what we want to avoid)
        reference = original_array  # No copy
        reference[0] = 999.  # This modifies original_array!
        self.assertEqual(cpuArray(original_array)[0], 999.,
                        "Reference behavior: original array was modified")

        # Reset for next test
        original_array[0] = 100.

        # Test 2: Copy behavior (what we want)
        copy_array = original_array.copy()  # Copy
        copy_array[0] = 999.  # This doesn't modify original_array
        self.assertEqual(cpuArray(original_array)[0], 100.,
                        "Copy behavior: original array was NOT modified")
        self.assertEqual(cpuArray(copy_array)[0], 999.,
                        "Copy behavior: copy array was modified")