import specula
specula.init(0)

import unittest
from specula import cpuArray, np
from specula.base_value import BaseValue
from specula.processing_objects.linear_combination import LinearCombination
from specula.data_objects.simul_params import SimulParams

class TestLinearCombination(unittest.TestCase):

    def setUp(self):
        self.simul_params = SimulParams(pixel_pupil=10, pixel_pitch=1.0, time_step=1)

    def test_basic_combination_no_focus_no_lift(self):
        # LGS and NGS only
        lgs = BaseValue(value=np.array([10., 20., 30., 40., 50.]))
        ngs = BaseValue(value=np.array([1., 2., 3., 4., 5.]))
        vectors = [lgs, ngs]
        lc = LinearCombination(self.simul_params, no_focus=True, no_lift=True)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()
        lc.trigger_code()
        out = cpuArray(lc.outputs['out_vector'].value)
        # TIP/TILT copied from NGS
        assert out[0] == 1.0
        assert out[1] == 2.0
        # Rest unchanged
        assert out[2] == 30.0

    def test_combination_with_focus(self):
        lgs = BaseValue(value=np.array([10., 20., 30., 40., 50.]))
        focus = BaseValue(value=np.array([99.]))
        ngs = BaseValue(value=np.array([1., 2., 3., 4., 5.]))
        vectors = [lgs, focus, ngs]
        lc = LinearCombination(self.simul_params, no_focus=False, no_lift=True)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()
        lc.trigger_code()
        out = cpuArray(lc.outputs['out_vector'].value)
        # TIP/TILT copied from NGS
        assert out[0] == 1.0
        assert out[1] == 2.0
        # Focus copied from focus
        assert out[2] == 99.0

    def test_combination_with_lift(self):
        lgs = BaseValue(value=np.array([10., 20., 30., 40., 50.]))
        lift = BaseValue(value=np.array([77.]))
        ngs = BaseValue(value=np.array([1., 2., 3., 4., 5.]))
        vectors = [lgs, lift, ngs]
        lc = LinearCombination(self.simul_params, no_focus=True, no_lift=False)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()
        lc.trigger_code()
        out = cpuArray(lc.outputs['out_vector'].value)
        # TIP/TILT copied from NGS
        assert out[0] == 1.0
        assert out[1] == 2.0
        # Lift is appended at the end
        assert out[-1] == 77.0

    def test_combination_with_focus_and_lift(self):
        lgs = BaseValue(value=np.array([10., 20., 30., 40., 50.]))
        focus = BaseValue(value=np.array([99.]))
        lift = BaseValue(value=np.array([77.]))
        ngs = BaseValue(value=np.array([1., 2., 3., 4., 5.]))
        vectors = [lgs, focus, lift, ngs]
        lc = LinearCombination(self.simul_params, no_focus=False, no_lift=False)
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

    def test_plate_scale_idx(self):
        lgs = BaseValue(value=np.array([10., 20., 30., 40., 50., 60., 70.]))
        focus = BaseValue(value=np.array([99.]))
        ngs = BaseValue(value=np.array([1., 2., 3., 4., 5.]))
        vectors = [lgs, focus, ngs]
        lc = LinearCombination(self.simul_params, no_focus=False, no_lift=True, plate_scale_idx=3)
        lc.inputs['in_vectors_list'].set(vectors)
        lc.setup()
        lc.trigger_code()
        out = cpuArray(lc.outputs['out_vector'].value)
        # Check that the plate_scale_idx block is overwritten by ngs[2:]
        np.testing.assert_array_equal(out[3:6], ngs.value[2:5])