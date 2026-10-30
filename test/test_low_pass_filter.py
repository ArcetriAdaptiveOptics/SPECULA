import specula
specula.init(-1)  # CPU

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.simul_params import SimulParams
from specula.data_objects.iir_filter_data import IirFilterData
from specula.processing_objects.low_pass_filter import LowPassFilter


class TestLowPassFilter(unittest.TestCase):

    def test_default_order(self):
        """Without n_ord and amplif_fact, the filter is the second-order Butterworth
        filter of lpf_from_fc()"""
        simul_params = SimulParams(time_step=0.001)
        lpf = LowPassFilter(simul_params, cutoff_freq=[10.0], target_device_idx=-1)
        ref = IirFilterData.lpf_from_fc(10.0, 1000.0, n_ord=2, target_device_idx=-1)
        np.testing.assert_allclose(cpuArray(lpf.iir_filter_data.num), cpuArray(ref.num))
        np.testing.assert_allclose(cpuArray(lpf.iir_filter_data.den), cpuArray(ref.den))

    def test_order_and_amplification_are_exclusive(self):
        simul_params = SimulParams(time_step=0.001)
        with self.assertRaises(ValueError):
            LowPassFilter(simul_params, cutoff_freq=[10.0], amplif_fact=[2.0], n_ord=2,
                          target_device_idx=-1)


if __name__ == '__main__':
    unittest.main()
