'''
Methods of IirFilterData not covered by test_iirfilter.py: filter construction
and editing, the analysis of the closed loop, and the control library interface.
'''
import specula
specula.init(-1, precision=0)

import os
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from iteras import cpuArray
from iteras.data_objects.iir_filter_data import IirFilterData, CONTROL_AVAILABLE
from iteras.lib import storage


def _integrator(gains=(0.5,), ff=None):
    return IirFilterData.from_gain_and_ff(list(gains), ff=ff, target_device_idx=-1)


class TestIirFilterDataConstruction(unittest.TestCase):

    def test_n_modes_repeats_the_filters(self):
        f = IirFilterData([2, 2], [2, 2], [[0.0, 0.5], [0.0, 0.4]], [[-1.0, 1.0], [-1.0, 1.0]],
                          n_modes=[2, 3], target_device_idx=-1)
        self.assertEqual(f.nfilter, 5)
        np.testing.assert_allclose(cpuArray(f.num)[:, 1], [0.5, 0.5, 0.4, 0.4, 0.4])
        for n_modes in ([2], [2, 3, 1]):
            with self.subTest(n_modes=n_modes), self.assertRaisesRegex(ValueError, 'n_modes'):
                IirFilterData([2, 2], [2, 2], [[0.0, 0.5], [0.0, 0.4]], [[-1.0, 1.0], [-1.0, 1.0]],
                              n_modes=n_modes, target_device_idx=-1)
        with self.assertRaisesRegex(ValueError, 'num and den'):
            IirFilterData([2, 2], [2, 2], [[0.0, 0.5]], [[-1.0, 1.0]], n_modes=[1, 1],
                          target_device_idx=-1)

    def test_zeros_and_poles(self):
        # (z - 0.5) / ((z - 0.2) (z - 0.9)), coefficients with the lowest order first
        f = IirFilterData([2], [3], [[0.0, -0.5, 1.0]], [[0.18, -1.1, 1.0]], target_device_idx=-1)
        np.testing.assert_allclose(cpuArray(f.get_zeros())[0, :1], [0.5])
        np.testing.assert_allclose(sorted(cpuArray(f.get_poles())[0]), [0.2, 0.9])
        f.set_zeros(np.array([[0.3, 0.0]]))
        np.testing.assert_allclose(cpuArray(f.zeros)[0, 0], 0.3)
        f.set_poles(np.array([[0.1, 0.4]]))
        np.testing.assert_allclose(sorted(cpuArray(f.poles)[0]), [0.1, 0.4])

    def test_set_gain(self):
        f = _integrator([0.5, 0.3])
        f.set_gain(0.2)
        np.testing.assert_allclose(cpuArray(f.gain), [0.2, 0.2])
        # Fewer gains than filters: only the first ones change; NaN keeps the gain
        f.set_gain([0.1])
        np.testing.assert_allclose(cpuArray(f.gain), [0.1, 0.2])
        f.set_gain([np.nan, 0.4])
        np.testing.assert_allclose(cpuArray(f.gain), [0.1, 0.4])
        # A filter without gain gets one
        g = IirFilterData([2], [2], [[0.0, 0.5]], [[-1.0, 1.0]], target_device_idx=-1)
        g.gain = None
        g.set_gain([0.7])
        np.testing.assert_allclose(cpuArray(g.gain), [0.7])

    def test_from_gain_and_ff(self):
        f = _integrator([0.5, 0.3], ff=[0.9])     # one forgetting factor for all
        np.testing.assert_allclose(cpuArray(f.den)[:, 0], [-0.9, -0.9])
        f = _integrator([0.5, 0.3], ff=[0.9, 0.8])
        np.testing.assert_allclose(cpuArray(f.den)[:, 0], [-0.9, -0.8])

    def test_low_pass_filters(self):
        with self.assertRaisesRegex(ValueError, 'order'):
            IirFilterData.lpf_from_fc(100.0, 1000.0, n_ord=3)
        with self.assertRaisesRegex(ValueError, 'half the sampling'):
            IirFilterData.lpf_from_fc(600.0, 1000.0)
        f = IirFilterData.lpf_from_fc_and_ampl([100.0, 200.0], 2.0, 1000.0, target_device_idx=-1)
        self.assertEqual(f.nfilter, 2)
        with self.assertRaisesRegex(ValueError, 'half the sampling'):
            IirFilterData.lpf_from_fc_and_ampl(600.0, 2.0, 1000.0)

    def test_save_restore_and_unsupported_methods(self):
        f = _integrator([0.5])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.npz')
            storage.write_object(path, {'VERSION': 2}, {'ordnum': cpuArray(f.ordnum)})
            with self.assertRaisesRegex(ValueError, 'version'):
                IirFilterData.restore(path)
        for call in (f.get_fits_header, lambda: IirFilterData.from_header({}), f.get_value,
                     lambda: f.set_value(0)):
            with self.assertRaises(NotImplementedError):
                call()


class TestIirFilterDataAnalysis(unittest.TestCase):

    def setUp(self):
        self.f = _integrator([0.5, 0.3])
        # Plant: one frame of delay, as dm, nw, dw coefficients
        self.dm, self.nw, self.dw = np.array([0.0, 1.0]), np.array([1.0]), np.array([1.0])

    def tearDown(self):
        plt.close('all')

    def test_transfer_functions_with_plant_and_plots(self):
        freq = np.array([1.0, 10.0, 100.0])
        for method in (self.f.RTF, self.f.NTF):
            with self.subTest(method=method.__name__):
                mag = method(0, 1000.0, freq=freq, dm=self.dm, nw=self.nw, dw=self.dw, plot=False)
                self.assertEqual(mag.shape, (3,))
                self.assertEqual(method(0, 1000.0, plot=True, unity_plant=True).shape, (1000,))
                method(0, 1000.0, freq=freq, plot=True, overplot=True, unity_plant=True)

    def test_frequency_response_edge_cases(self):
        # Leading zeros are removed, and empty coefficients become constants
        r = self.f.frequency_response(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), 1000.0)
        self.assertEqual(r.shape, (1000,))
        r = self.f.frequency_response(np.array([]), np.array([]), 1000.0, freq=np.array([10.0]))
        np.testing.assert_allclose(np.abs(r), [0.0])

    def test_stability(self):
        self.assertTrue(self.f.is_stable(0, dm=self.dm, nw=self.nw, dw=self.dw))
        # Constant closed-loop denominator
        static = IirFilterData([1], [1], [[1.0]], [[1.0]], target_device_idx=-1)
        self.assertFalse(static.is_stable(0, unity_plant=True))
        with self.assertRaises(IndexError):
            self.f.is_stable(5, unity_plant=True)

    def test_max_stable_gain_and_resonance(self):
        g = self.f.max_stable_gain(mode=0, dm=self.dm, nw=self.nw, dw=self.dw, n_gain=200)
        self.assertGreater(g, 0.5)
        gains = self.f.max_stable_gain(dm=self.dm, nw=self.nw, dw=self.dw, n_gain=200)
        self.assertEqual(gains.shape, (2,))
        static = IirFilterData([1], [1], [[1.0]], [[1.0]], target_device_idx=-1)
        self.assertEqual(static.max_stable_gain(mode=0, n_gain=10, unity_plant=True), 0.0)
        freq, amp = self.f.resonance_frequency(0, gain_factor=0.5, dm=self.dm, nw=self.nw, dw=self.dw)
        self.assertGreater(amp, 0)
        self.f.resonance_frequency(0, unity_plant=True)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.f.resonance_frequency(5)

    def test_stability_analysis(self):
        result = self.f.stability_analysis(mode=0, delay=1.0, n_gain=200)
        self.assertTrue(result['is_stable_at_max'])
        results = self.f.stability_analysis(delay=1.0, n_gain=200)
        self.assertEqual(len(results), 2)
        # No stable gain: no resonance
        static = IirFilterData([1], [1], [[1.0]], [[1.0]], target_device_idx=-1)
        self.assertEqual(static.stability_analysis(mode=0, n_gain=10, unity_plant=True)['resonance_frequency'], 0.0)
        self.assertEqual(static.stability_analysis(n_gain=10, unity_plant=True)[0]['resonance_frequency'], 0.0)


@unittest.skipUnless(CONTROL_AVAILABLE, 'The control package is not installed')
class TestIirFilterDataControl(unittest.TestCase):

    def setUp(self):
        self.f = _integrator([0.5])

    def tearDown(self):
        plt.close('all')

    def test_transfer_function(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.f.to_control_tf(mode=3)
        self.assertEqual(self.f.to_control_tf(dt=0.001).dt, 0.001)

    def test_plots_and_responses(self):
        for dt in (None, 0.001):
            with self.subTest(dt=dt):
                self.assertIsNotNone(self.f.bode_plot(dt=dt, plot=False))
                self.assertIsNotNone(self.f.nyquist_plot(dt=dt, plot=False))
                t, y = self.f.step_response(dt=dt)
                self.assertEqual(len(t), len(np.atleast_1d(y).ravel()) if np.ndim(y) == 1 else len(t))
                t, y = self.f.impulse_response(dt=dt)
                self.assertGreater(len(t), 0)
        self.assertEqual(len(self.f.stability_margins(dt=0.001, unity_plant=True)), 4)
        poles, zeros = self.f.pole_zero_map(dt=0.001, plot=False)
        self.assertEqual(len(poles), 1)


if __name__ == '__main__':
    unittest.main()
