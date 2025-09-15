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