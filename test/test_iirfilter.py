import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
import unittest.mock
import numpy as np
from specula import cpuArray
from specula.data_objects.iir_filter_data import IirFilterData
from test.specula_testlib import cpu_and_gpu

try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    control = None

requires_control = unittest.skipIf(not CONTROL_AVAILABLE, "Control library not available")


def nonzero(coeffs, default):
    """Non-zero coefficients, or [default] if they are all zero"""
    coeffs = cpuArray(coeffs)
    return coeffs[coeffs != 0] if np.any(coeffs != 0) else np.array([default])


class TestIirFilterData(unittest.TestCase):

    def n_modes_filter(self, target_device_idx, xp):
        """Two filters, expanded to 3 + 2 modes"""
        num = xp.array([[0.0, 0.5], [0.0, 0.3]])
        den = xp.array([[-1.0, 1.0], [-0.9, 1.0]])
        filt = IirFilterData([2, 2], [2, 2], num, den, n_modes=[3, 2],
                             target_device_idx=target_device_idx)
        return filt, cpuArray(num), cpuArray(den)

    def assert_same_coeffs(self, a, b, mode_a=0, mode_b=0):
        """Same non-zero numerator and denominator coefficients"""
        for name, default in (('num', 0), ('den', 1)):
            np.testing.assert_allclose(nonzero(getattr(a, name)[mode_a], default),
                                       nonzero(getattr(b, name)[mode_b], default),
                                       rtol=1e-10, atol=1e-12, err_msg=f'{name} of filter {mode_a}')

    @cpu_and_gpu
    def test_init_with_n_modes_expansion(self, target_device_idx, xp):
        filt, num, den = self.n_modes_filter(target_device_idx, xp)
        self.assertEqual(filt.num.shape, (5, 2))
        self.assertEqual(filt.den.shape, (5, 2))
        np.testing.assert_allclose(cpuArray(filt.num), [num[0]] * 3 + [num[1]] * 2)
        np.testing.assert_allclose(cpuArray(filt.den), [den[0]] * 3 + [den[1]] * 2)

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):
        filt, _, _ = self.n_modes_filter(target_device_idx, xp)
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'iir_filter_data.fits')
            filt.save(filename)
            restored = IirFilterData.restore(filename, target_device_idx=target_device_idx)
        for attr in ('num', 'den', 'ordnum', 'ordden'):
            np.testing.assert_array_equal(cpuArray(getattr(filt, attr)),
                                          cpuArray(getattr(restored, attr)))

    @cpu_and_gpu
    def test_from_gain_and_ff(self, target_device_idx, xp):
        """Default ff = 1: integrators 0.2 z / (z - 1)"""
        f = IirFilterData.from_gain_and_ff([0.2] * 10, target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(f.num), [[0, 0.2]] * 10)
        np.testing.assert_array_equal(cpuArray(f.den), [[-1, 1]] * 10)

    @requires_control
    @cpu_and_gpu
    def test_control_conversion_roundtrip_single_filter(self, target_device_idx, xp):
        original = IirFilterData.lpf_from_fc(50, 1000, n_ord=2, target_device_idx=target_device_idx)
        tf = original.to_control_tf(mode=0, dt=1.0 / 1000)
        self.assert_same_coeffs(original, IirFilterData.from_control_tf(
            tf, target_device_idx=target_device_idx))

    @requires_control
    @cpu_and_gpu
    def test_control_conversion_roundtrip_multiple_filters(self, target_device_idx, xp):
        fc_list = [10, 50, 100]
        original = IirFilterData.lpf_from_fc(fc_list, 1000, n_ord=2,
                                             target_device_idx=target_device_idx)
        tf_list = original.to_control_tf_list(dt=1.0 / 1000)
        self.assertEqual(len(tf_list), len(fc_list))
        restored = IirFilterData.from_control_tf(tf_list, target_device_idx=target_device_idx)
        for mode in range(len(fc_list)):
            self.assert_same_coeffs(original, restored, mode, mode)

    @requires_control
    @cpu_and_gpu
    def test_control_conversion_with_gain_and_ff(self, target_device_idx, xp):
        """Same frequency responses after the roundtrip (the coefficients may be rearranged)"""
        original = IirFilterData.from_gain_and_ff([0.1, 0.5, 1.0], [0.9, 0.95, 0.99],
                                                  target_device_idx=target_device_idx)
        restored = IirFilterData.from_control_tf(original.to_control_tf_list(dt=1.0),
                                                 target_device_idx=target_device_idx)
        freq = np.logspace(-2, 0, 100)
        for mode in range(3):
            a, b = [f.frequency_response(f.num[mode], f.den[mode], fs=1.0, freq=freq)
                    for f in (original, restored)]
            np.testing.assert_allclose(np.abs(a), np.abs(b), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(np.angle(a), np.angle(b), rtol=1e-12, atol=1e-14)

    @requires_control
    def test_control_conversion_error_handling(self):
        """Without control, the control-dependent methods raise ImportError"""
        import specula.data_objects.iir_filter_data as iir_module
        with unittest.mock.patch.object(iir_module, 'CONTROL_AVAILABLE', False):
            filter_data = IirFilterData.from_gain_and_ff([0.5], target_device_idx=None)
            for call in (filter_data.to_control_tf, filter_data.bode_plot,
                         filter_data.nyquist_plot, lambda: IirFilterData.from_control_tf([])):
                with self.assertRaises(ImportError):
                    call()

    @requires_control
    @cpu_and_gpu
    def test_frequency_response_consistency(self, target_device_idx, xp):
        """frequency_response() equals the response computed by control"""
        fs = 1000
        filter_data = IirFilterData.lpf_from_fc(100, fs, n_ord=2, target_device_idx=target_device_idx)
        freq = np.logspace(0, np.log10(fs / 2 - 10), 100)   # avoid the Nyquist frequency
        response_control = filter_data.to_control_tf(mode=0, dt=1.0 / fs).frequency_response(
            2 * np.pi * freq)
        response_manual = filter_data.frequency_response(filter_data.num[0], filter_data.den[0],
                                                         fs, freq=freq)
        np.testing.assert_allclose(np.abs(np.squeeze(response_control.magnitude)),
                                   np.abs(response_manual), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.squeeze(response_control.phase), np.angle(response_manual),
                                   rtol=1e-10, atol=1e-12)

    def test_has_control_support_property(self):
        self.assertEqual(IirFilterData.from_gain_and_ff([0.5]).has_control_support,
                         CONTROL_AVAILABLE)

    @cpu_and_gpu
    def test_discrete_delay_tf(self, target_device_idx, xp):
        """Integer delay d: num = [1, 0, ...], den = [0, ..., 0, 1] (length d + 1).
        Delay 2.3: same den as delay 3, num = [0.3, 0.7, 0, 0]"""
        filter_data = IirFilterData.from_gain_and_ff([0.5], target_device_idx=target_device_idx)
        for delay in (1, 2, 3, 4):
            num, den = filter_data.discrete_delay_tf(delay)
            np.testing.assert_array_equal(num, np.eye(delay + 1)[0])
            np.testing.assert_array_equal(den, np.eye(delay + 1)[-1])
        num, den = filter_data.discrete_delay_tf(2.3)
        np.testing.assert_allclose(num, [0.3, 0.7, 0.0, 0.0], rtol=1e-12)
        np.testing.assert_array_equal(den, [0.0, 0.0, 0.0, 1.0])

    @cpu_and_gpu
    def test_set_gain(self, target_device_idx, xp):
        """set_gain() with a list or a scalar scales the numerators, the denominators do not change"""
        gains, ff = [0.1, 0.5, 1.0], [0.9, 0.95, 0.99]
        for new_gain, expected in (([0.2, 1.0, 0.5], [0.2, 1.0, 0.5]), (0.2, [0.2] * 3)):
            with self.subTest(new_gain=new_gain):
                filter_data = IirFilterData.from_gain_and_ff(gains, ff,
                                                             target_device_idx=target_device_idx)
                np.testing.assert_allclose(cpuArray(filter_data.gain), gains, rtol=1e-12)
                filter_data.set_gain(new_gain)
                np.testing.assert_allclose(cpuArray(filter_data.gain), expected, rtol=1e-12)
                np.testing.assert_allclose(cpuArray(filter_data.num),
                                           [[0, g] for g in expected], rtol=1e-7)
                np.testing.assert_allclose(cpuArray(filter_data.den),
                                           [[-f, 1.0] for f in ff], rtol=1e-7)

    @cpu_and_gpu
    def test_set_gain_zero_preserves_numerator_structure(self, target_device_idx, xp):
        """Setting gain to zero must be reversible without losing numerator shape."""
        filter_data = IirFilterData([3], [2], xp.array([[0.2, 0.4, 0.8]], dtype=xp.float32),
                                    xp.array([[-0.9, 1.0, 0.0]], dtype=xp.float32),
                                    target_device_idx=target_device_idx)
        original_num = cpuArray(filter_data.num.copy())
        filter_data.set_gain([0.0])
        np.testing.assert_allclose(cpuArray(filter_data.num), np.zeros_like(original_num),
                                   rtol=1e-12, atol=1e-12)
        for gain, factor in ((0.8, 1.0), (1.6, 2.0)):
            filter_data.set_gain([gain])
            np.testing.assert_allclose(cpuArray(filter_data.num), original_num * factor,
                                       rtol=1e-7, atol=1e-9)

    @cpu_and_gpu
    def test_poles_zeros_gains_roundtrip_simple(self, target_device_idx, xp):
        """gain/ff filters: poles at ff, zeros at the origin"""
        gains, ff = [0.1, 0.5], [0.9, 0.95]
        f = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)
        np.testing.assert_allclose(cpuArray(f.get_poles()), [[ff[0]], [ff[1]]], rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(cpuArray(f.get_zeros()), [[0.0], [0.0]], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(cpuArray(f.gain), gains, rtol=1e-12, atol=1e-14)

    @cpu_and_gpu
    def test_rtf_ntf_output_shape(self, target_device_idx, xp):
        filter_data = IirFilterData.from_gain_and_ff([0.5], [1.0], target_device_idx=target_device_idx)
        fs = 1000
        freq = np.linspace(1/10, fs/2, 20*fs)
        nw, dw = filter_data.discrete_delay_tf(1)
        plant = dict(freq=freq, nw=nw, dw=dw, dm=xp.array([0.0, 1.0], dtype=xp.float32), plot=False)
        for tf in (filter_data.RTF(0, fs, **plant), filter_data.NTF(0, fs, **plant)):
            self.assertEqual(tf.shape, freq.shape)
            self.assertTrue(np.all(np.isfinite(tf)))

    @cpu_and_gpu
    def test_is_stable(self, target_device_idx, xp):
        """Integrator with a 3-frame delay: stable with gain 0.3, unstable with 0.7"""
        dm = xp.array([0.0, 1.0], dtype=xp.float32)
        for gain, stable in ((0.3, True), (0.7, False)):
            f = IirFilterData.from_gain_and_ff([gain], [1.0], target_device_idx=target_device_idx)
            nw, dw = f.discrete_delay_tf(2)
            self.assertEqual(f.is_stable(0, nw=nw, dw=dw, dm=dm), stable)

    @requires_control
    @cpu_and_gpu
    def test_stability_margins(self, target_device_idx, xp):
        filter_data = IirFilterData.from_gain_and_ff([0.1], [0.5], target_device_idx=target_device_idx)
        gm, pm, wg, wp = filter_data.stability_margins(0, dt=1.0, unity_plant=True)
        self.assertTrue(gm > 0 or np.isinf(gm))
        self.assertTrue(pm > 0)

    @cpu_and_gpu
    def test_max_stable_gain_integrator_known_values(self, target_device_idx, xp):
        """Integrator: max gain 1.0 with 2 frames of delay, 0.618 with 3, less with 4"""
        integrator = IirFilterData.from_gain_and_ff([1.0], [1.0], target_device_idx=target_device_idx)
        g2, g3, g4 = [integrator.max_stable_gain(mode=0, delay=d, n_gain=50000) for d in (2, 3, 4)]
        self.assertAlmostEqual(g2, 1.0, places=2)
        self.assertAlmostEqual(g3, 0.618, places=2)
        self.assertLess(g4, g3)

    @cpu_and_gpu
    def test_max_stable_gain_all_modes(self, target_device_idx, xp):
        filters = IirFilterData.from_gain_and_ff([1.0] * 3, [1.0] * 3,
                                                 target_device_idx=target_device_idx)
        max_gains = filters.max_stable_gain(delay=3, n_gain=20000)
        for max_gain in max_gains:
            self.assertAlmostEqual(max_gain, 0.618, places=1)
        np.testing.assert_allclose(max_gains, max_gains[0], rtol=1e-10)

    @cpu_and_gpu
    def test_max_stable_gain_caching(self, target_device_idx, xp):
        """Two pairs of identical filters, computed twice"""
        filters = IirFilterData.from_gain_and_ff([0.5, 1.0, 0.5, 1.0], [0.9, 1.0, 0.9, 1.0],
                                                 target_device_idx=target_device_idx)
        first = filters.max_stable_gain(delay=3, n_gain=10000)
        second = filters.max_stable_gain(delay=3, n_gain=10000)
        np.testing.assert_allclose(first, second, rtol=1e-12)
        self.assertAlmostEqual(first[0], first[2], places=10)
        self.assertAlmostEqual(second[1], second[3], places=10)

    @cpu_and_gpu
    def test_resonance_frequency_delay_relationship(self, target_device_idx, xp):
        """At 80% of the max stable gain, the resonance frequency decreases with the delay"""
        integrator = IirFilterData.from_gain_and_ff([1.0], [1.0], target_device_idx=target_device_idx)
        freqs = []
        for delay in (2, 3, 4, 5):
            gain_factor = integrator.max_stable_gain(mode=0, delay=delay, n_gain=20000) * 0.8
            self.assertGreater(gain_factor, 0)
            freqs.append(integrator.resonance_frequency(mode=0, gain_factor=gain_factor,
                                                        delay=delay, fs=1000.0)[0])
        self.assertGreater(min(freqs), 0)
        self.assertEqual(freqs, sorted(freqs, reverse=True))
        self.assertEqual(len(set(freqs)), len(freqs))

    @cpu_and_gpu
    def test_stability_analysis_comprehensive(self, target_device_idx, xp):
        integrator = IirFilterData.from_gain_and_ff([1.0], [1.0], target_device_idx=target_device_idx)
        result = integrator.stability_analysis(mode=0, delay=3, fs=1000.0, n_gain=20000)
        self.assertLessEqual({'mode', 'max_stable_gain', 'resonance_frequency',
                              'resonance_amplitude', 'is_stable_at_max'}, set(result))
        self.assertEqual(result['mode'], 0)
        self.assertAlmostEqual(result['max_stable_gain'], 0.618, places=1)
        self.assertGreater(result['resonance_frequency'], 0)
        self.assertTrue(result['is_stable_at_max'])

    @cpu_and_gpu
    def test_stability_analysis_all_modes(self, target_device_idx, xp):
        filters = IirFilterData.from_gain_and_ff([0.5, 1.0], [0.9, 1.0],
                                                 target_device_idx=target_device_idx)
        results = filters.stability_analysis(delay=3, fs=1000.0, n_gain=10000)
        self.assertEqual(len(results), 2)
        for i, result in enumerate(results):
            self.assertEqual(result['mode'], i)
            self.assertGreater(result['max_stable_gain'], 0)
            self.assertGreater(result['resonance_frequency'], 0)
            self.assertTrue(result['is_stable_at_max'])

    @cpu_and_gpu
    def test_max_stable_gain_with_plant_dynamics(self, target_device_idx, xp):
        """A 1-frame dm plus a 2-frame nw/dw plant is the same as delay=3"""
        integrator = IirFilterData.from_gain_and_ff([1.0], [1.0], target_device_idx=target_device_idx)
        nw, dw = integrator.discrete_delay_tf(2)
        self.assertAlmostEqual(
            integrator.max_stable_gain(mode=0, dm=xp.array([0.0, 1.0]), nw=nw, dw=dw, n_gain=20000),
            integrator.max_stable_gain(mode=0, delay=3, n_gain=20000), places=1)

    @cpu_and_gpu
    def test_resonance_frequency_gain_scaling(self, target_device_idx, xp):
        integrator = IirFilterData.from_gain_and_ff([1.0], [1.0], target_device_idx=target_device_idx)
        for gain_factor in (0.1, 0.3, 0.5):
            res_freq, _ = integrator.resonance_frequency(mode=0, gain_factor=gain_factor,
                                                         delay=3, fs=1000.0)
            self.assertGreater(res_freq, 0)

    def test_stability_methods_consistency(self):
        """is_stable() at low gain and at 95% of max_stable_gain(), not at 110%"""
        filter_data = IirFilterData.from_gain_and_ff([0.1], [1.0])
        plant = dict(dm=np.array([1.0]))
        plant['nw'], plant['dw'] = filter_data.discrete_delay_tf(3)
        self.assertTrue(filter_data.is_stable(0, **plant))
        max_gain = filter_data.max_stable_gain(mode=0, delay=3, n_gain=10000)
        self.assertLess(max_gain, 10.0)
        for factor, stable in ((0.95, True), (1.1, False)):
            filter_data.set_gain([max_gain * factor])
            self.assertEqual(filter_data.is_stable(0, **plant), stable)


class TestIirFilterDataZerosPoles(unittest.TestCase):

    def test_zeros_and_poles_of_known_filter(self):
        """H(z) = (z - 0.5) / (z - 0.8): coefficients lowest order first"""
        data = IirFilterData(ordnum=[2], ordden=[2], num=[[-0.5, 1.0]], den=[[-0.8, 1.0]],
                             target_device_idx=-1)
        np.testing.assert_allclose(cpuArray(data.get_zeros()), [[0.5]], atol=1e-12)
        np.testing.assert_allclose(cpuArray(data.get_poles()), [[0.8]], atol=1e-12)

    def test_set_zeros_and_poles(self):
        """set_zeros()/set_poles() give the coefficients of (z - 0.3) / (z - 0.9)"""
        data = IirFilterData(ordnum=[2], ordden=[2], num=[[0.0, 1.0]], den=[[-1.0, 1.0]],
                             target_device_idx=-1)
        data.set_zeros(np.array([[0.3]]))
        data.set_poles(np.array([[0.9]]))
        np.testing.assert_allclose(cpuArray(data.num), [[-0.3, 1.0]], atol=1e-12)
        np.testing.assert_allclose(cpuArray(data.den), [[-0.9, 1.0]], atol=1e-12)

    def test_complex_poles_of_second_order_filter(self):
        """Poles 0.9 exp(+-i pi/4): den = z^2 - 2 r cos(theta) z + r^2"""
        r, theta = 0.9, np.pi / 4
        data = IirFilterData(ordnum=[3], ordden=[3], num=[[0.0, 0.0, 1.0]],
                             den=[[r ** 2, -2 * r * np.cos(theta), 1.0]], target_device_idx=-1)
        poles = cpuArray(data.get_poles())[0]
        self.assertTrue(np.iscomplexobj(poles))
        np.testing.assert_allclose(sorted(poles, key=lambda p: p.imag),
                                   [r * np.exp(-1j * theta), r * np.exp(1j * theta)], atol=1e-12)

    def test_set_complex_poles(self):
        """set_poles() with a conjugate pair gives the real coefficients, and the poles back"""
        r, theta = 0.9, np.pi / 4
        pair = np.array([[r * np.exp(1j * theta), r * np.exp(-1j * theta)]])
        data = IirFilterData(ordnum=[3], ordden=[3], num=[[0.0, 0.0, 1.0]],
                             den=[[0.0, -1.0, 1.0]], target_device_idx=-1)
        data.set_poles(pair)
        np.testing.assert_allclose(cpuArray(data.den), [[r ** 2, -2 * r * np.cos(theta), 1.0]],
                                   atol=1e-12)
        self.assertFalse(np.iscomplexobj(cpuArray(data.den)))
        data.poles = None   # recompute them from the coefficients
        np.testing.assert_allclose(sorted(cpuArray(data.get_poles())[0], key=lambda p: p.imag),
                                   sorted(pair[0], key=lambda p: p.imag), atol=1e-12)

    def test_set_poles_rejects_unpaired_complex_roots(self):
        data = IirFilterData(ordnum=[3], ordden=[3], num=[[0.0, 0.0, 1.0]],
                             den=[[0.0, -1.0, 1.0]], target_device_idx=-1)
        with self.assertRaises(ValueError):
            data.set_poles(np.array([[0.5 + 0.5j, 0.2]]))


class TestIirFilterDataWidths(unittest.TestCase):

    def test_num_shorter_than_den_is_padded(self):
        """num = [0.5], den = [-1, 1]: IirFilter runs y[n] = y[n-1] + 0.5 x[n], i.e.
        H(z) = 0.5 z / (z - 1). num is padded to [0, 0.5], so that the analysis
        methods give the same transfer function as the simulation."""
        data = IirFilterData(ordnum=[1], ordden=[2], num=[[0.5]], den=[[-1.0, 1.0]],
                             target_device_idx=-1)
        np.testing.assert_array_equal(cpuArray(data.num), [[0.0, 0.5]])
        np.testing.assert_array_equal(cpuArray(data.gain), [0.5])
        fs, f = 1000.0, 300.0
        z = np.exp(2j * np.pi * f / fs)
        h = data.frequency_response(data.num[0], data.den[0], fs, freq=np.array([f]))[0]
        np.testing.assert_allclose(h, 0.5 * z / (z - 1), rtol=1e-12)

    def test_padding_does_not_change_the_simulation(self):
        """Impulse response of IirFilter with num = [0.5] and with num = [0, 0.5]"""
        from specula.base_value import BaseValue
        from specula.processing_objects.iir_filter import IirFilter

        def impulse_response(num):
            data = IirFilterData(ordnum=[len(num[0])], ordden=[2], num=num, den=[[-1.0, 1.0]],
                                 target_device_idx=-1)
            flt = IirFilter(data, target_device_idx=-1)
            inp = BaseValue(value=np.zeros(1), target_device_idx=-1)
            flt.inputs['delta_comm'].set(inp)
            flt.setup()
            out = []
            for t in range(1, 6):
                inp.value[:] = 1.0 if t == 1 else 0.0
                inp.generation_time = t
                flt.check_ready(t)
                flt.trigger()
                flt.post_trigger()
                out.append(float(cpuArray(flt.outputs['out_comm'].value)[0]))
            return out

        np.testing.assert_allclose(impulse_response([[0.5]]), [0.5] * 5)
        np.testing.assert_allclose(impulse_response([[0.0, 0.5]]), [0.5] * 5)

    def test_den_shorter_than_num_is_padded(self):
        data = IirFilterData(ordnum=[3], ordden=[2], num=[[0.1, 0.2, 0.3]], den=[[-0.5, 1.0]],
                             target_device_idx=-1)
        np.testing.assert_array_equal(cpuArray(data.den), [[0.0, -0.5, 1.0]])


class TestIirFilterDataLowPass(unittest.TestCase):

    def test_butterworth_cutoff(self):
        """Butterworth low-pass filters (bilinear transform with prewarping): unit DC gain,
        |H| = 1/sqrt(2) exactly at fc, zero at the Nyquist frequency, for orders 1, 2, 4"""
        fs, fc = 1000.0, 50.0
        for n_ord in (1, 2, 4):
            with self.subTest(n_ord=n_ord):
                data = IirFilterData.lpf_from_fc(fc, fs, n_ord=n_ord, target_device_idx=-1)
                h = np.abs(data.frequency_response(data.num[0], data.den[0], fs,
                                                   freq=np.array([1e-6, fc, fs / 2])))
                np.testing.assert_allclose(h, [1.0, 1 / np.sqrt(2), 0.0], atol=1e-9)


class TestIirFilterDataControlTimebase(unittest.TestCase):

    @requires_control
    def test_default_is_discrete_time(self):
        """The coefficients are in z: without dt, to_control_tf() gives a discrete-time
        transfer function, equal to 0.5 z / (z - 1) for the integrator"""
        tf = IirFilterData.from_gain_and_ff([0.5], target_device_idx=-1).to_control_tf()
        self.assertTrue(control.isdtime(tf, strict=True))
        z = np.exp(0.3j)
        np.testing.assert_allclose(complex(tf(z)), 0.5 * z / (z - 1), rtol=1e-12)


class TestIirFilterDataExplicitPlant(unittest.TestCase):

    def setUp(self):
        self.data = IirFilterData.from_gain_and_ff([0.5], target_device_idx=-1)

    def test_plant_must_be_chosen(self):
        """Without a plant, or with more than one, the analysis methods raise ValueError"""
        calls = [lambda **p: self.data.RTF(0, 1000.0, plot=False, **p),
                 lambda **p: self.data.NTF(0, 1000.0, plot=False, **p),
                 lambda **p: self.data.is_stable(0, **p),
                 lambda **p: self.data.max_stable_gain(mode=0, n_gain=10, **p),
                 lambda **p: self.data.resonance_frequency(0, **p),
                 lambda **p: self.data.stability_analysis(mode=0, n_gain=10, **p)]
        for call in calls:
            with self.assertRaisesRegex(ValueError, 'explicitly'):
                call()
            with self.assertRaisesRegex(ValueError, 'explicitly'):
                call(delay=1, unity_plant=True)
            with self.assertRaisesRegex(ValueError, 'together'):
                call(dm=[1.0], nw=[1.0])

    def test_unity_plant_equals_explicit_unity_coefficients(self):
        """unity_plant=True is the same as P = 1 / (1 * 1)"""
        freq = np.array([10.0, 100.0, 300.0])
        a = self.data.RTF(0, 1000.0, freq=freq, plot=False, unity_plant=True)
        b = self.data.RTF(0, 1000.0, freq=freq, plot=False, dm=[1.0], nw=[1.0], dw=[1.0])
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_rtf_of_integrator_with_one_frame_delay(self):
        """C = 0.5 z / (z - 1), P = 1/z: RTF = 1 / (1 + 0.5 / (z - 1)) = (z - 1) / (z - 0.5)"""
        fs, f = 1000.0, np.array([50.0, 200.0])
        z = np.exp(2j * np.pi * f / fs)
        rtf = self.data.RTF(0, fs, freq=f, plot=False, delay=1)
        np.testing.assert_allclose(rtf, np.abs((z - 1) / (z - 0.5)), rtol=1e-12)

    @requires_control
    def test_stability_margins_of_the_loop(self):
        """Integrator 0.5 z / (z - 1) with P = 1/z^2: the loop L = 0.5 / (z (z - 1)) is at the
        stability limit for a gain of 1 (z^2 - z + g = 0 has |z| = 1 for g = 1), so the gain
        margin is 1 / 0.5 = 2 (6.02 dB)"""
        gm_db, pm, wg, wp = self.data.stability_margins(0, delay=2)
        np.testing.assert_allclose(gm_db, 20 * np.log10(2.0), atol=1e-6)


class TestIirFilterDataUnusedRoots(unittest.TestCase):

    def test_unused_entries_are_nan(self):
        """Second-order Butterworth filter in arrays of width 5 (n_coeff = 2 n_ord + 1): two
        poles (a complex pair inside the unit circle) and two zeros at z = -1, then NaN"""
        data = IirFilterData.lpf_from_fc(50.0, 1000.0, n_ord=2, target_device_idx=-1)
        poles = cpuArray(data.get_poles())[0]
        zeros = cpuArray(data.get_zeros())[0]
        self.assertEqual(poles.shape, (4,))
        self.assertTrue(np.all(np.isfinite(poles[:2])) and np.all(np.isnan(poles[2:])))
        self.assertTrue(np.all(np.abs(poles[:2]) < 1))
        np.testing.assert_allclose(zeros[:2], [-1.0, -1.0], atol=1e-6)
        self.assertTrue(np.all(np.isnan(zeros[2:])))

    def test_true_zero_at_origin_is_kept(self):
        """The integrator 0.5 z / (z - 1) has a true zero at z = 0, which is not NaN"""
        data = IirFilterData.from_gain_and_ff([0.5], target_device_idx=-1)
        np.testing.assert_allclose(cpuArray(data.get_zeros()), [[0.0]], atol=1e-12)
        np.testing.assert_allclose(cpuArray(data.get_poles()), [[1.0]], atol=1e-12)


class TestIirFilterDataExactStability(unittest.TestCase):

    def test_max_stable_gain_of_integrator_with_delay(self):
        """Integrator g z / (z - 1), plant z^-d: z^d - z^(d-1) + g = 0 is at the stability limit for
        g = 2 (d = 1), g = 1 (d = 2), g = (sqrt(5) - 1) / 2 (d = 3). The result does not depend on the
        current gain of the filter (normalized filter)"""
        expected = {1: 2.0, 2: 1.0, 3: (np.sqrt(5) - 1) / 2}
        for gain in (1.0, 0.3):
            data = IirFilterData.from_gain_and_ff([gain], target_device_idx=-1)
            for d, g_max in expected.items():
                with self.subTest(gain=gain, delay=d):
                    np.testing.assert_allclose(data.max_stable_gain(mode=0, delay=d), g_max,
                                               rtol=1e-9)

    def test_max_stable_gain_is_the_stability_limit(self):
        """Fractional delay and a low-pass filter in the plant: the loop is stable just below the
        result and unstable just above it (checked with is_stable)"""
        data = IirFilterData.from_gain_and_ff([1.0], [0.99], target_device_idx=-1)
        lpf = IirFilterData.lpf_from_fc(200.0, 1000.0, n_ord=2, target_device_idx=-1)
        nw, dw = data.discrete_delay_tf(2.4)
        plant = dict(nw=np.convolve(nw, cpuArray(lpf.num[0])), dm=cpuArray(lpf.den[0]), dw=dw)
        g_max = data.max_stable_gain(mode=0, **plant)
        self.assertTrue(0 < g_max < 20)
        for factor, stable in ((1 - 1e-6, True), (1 + 1e-6, False)):
            probe = IirFilterData.from_gain_and_ff([g_max * factor], [0.99], target_device_idx=-1)
            self.assertEqual(probe.is_stable(0, **plant), stable)

    def test_always_stable_loop_returns_max_gain(self):
        """Integrator with P = 1: z (1 + g) - 1 = 0 has its root 1 / (1 + g) inside the unit
        circle for all positive gains"""
        data = IirFilterData.from_gain_and_ff([1.0], target_device_idx=-1)
        self.assertEqual(data.max_stable_gain(mode=0, unity_plant=True), 20.0)
        self.assertEqual(data.max_stable_gain(mode=0, unity_plant=True, max_gain=5.0), 5.0)

    def test_zero_gain_filter_cannot_be_normalized(self):
        data = IirFilterData(ordnum=[2], ordden=[2], num=[[0.5, 0.0]], den=[[-1.0, 1.0]],
                             target_device_idx=-1)
        with self.assertRaisesRegex(ValueError, 'normalized'):
            data.max_stable_gain(mode=0, delay=1)

    def test_resonance_is_the_rtf_peak(self):
        """Integrator 0.5 z / (z - 1), plant z^-2: L = 0.5 / (z (z - 1)). The resonance is the
        maximum of |1 / (1 + L)|, here computed directly on the same grid"""
        data = IirFilterData.from_gain_and_ff([0.5], target_device_idx=-1)
        fs = 1000.0
        freq = np.linspace(0, fs / 2, 10001)[1:]
        z = np.exp(2j * np.pi * freq / fs)
        loop = 0.5 / (z * (z - 1))
        for kwargs, expected in (({}, np.abs(1 / (1 + loop))),   # default: RTF
                                 ({'transfer_function': 'ntf'}, np.abs(loop / (1 + loop)))):
            f_res, peak = data.resonance_frequency(0, delay=2, fs=fs, **kwargs)
            self.assertAlmostEqual(f_res, freq[np.argmax(expected)])
            np.testing.assert_allclose(peak, expected.max(), rtol=1e-10)

    def test_stability_analysis_resonance_near_the_limit(self):
        """Integrator with gain 0.5 and plant z^-2: g_max = 1, and at 0.99 g_max the closed-loop
        poles z^2 - z + 0.99 = 0 are close to the unit circle at the angle
        arccos(1 / (2 sqrt(0.99))): the RTF peaks at that frequency. The current gain (0.5) must
        not be applied twice."""
        data = IirFilterData.from_gain_and_ff([0.5], target_device_idx=-1)
        fs = 1000.0
        result = data.stability_analysis(mode=0, delay=2, fs=fs)
        np.testing.assert_allclose(result['max_stable_gain'], 1.0, rtol=1e-9)
        f_pole = np.arccos(1 / (2 * np.sqrt(0.99))) / (2 * np.pi) * fs
        self.assertAlmostEqual(result['resonance_frequency'], f_pole, delta=1.0)
        self.assertGreater(result['resonance_amplitude'], 10)
