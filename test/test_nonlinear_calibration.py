import os
import tempfile
import unittest

import specula
specula.init(0)

from specula import np, cpuArray
from specula.data_objects.nonlinear_calibration import NonlinearCalibration

from test.specula_testlib import cpu_and_gpu


AMPLITUDES = np.linspace(-10.0, 10.0, 21)


def saturating(amplitudes, scale):
    return scale * np.tanh(amplitudes / scale)


class TestNonlinearCalibration(unittest.TestCase):

    def _build(self, target_device_idx, first_mode=0):
        # mode first_mode: linear response; the next two: saturating
        responses = np.vstack([AMPLITUDES, saturating(AMPLITUDES, 5.0), saturating(AMPLITUDES, 8.0)])
        return NonlinearCalibration(AMPLITUDES, responses, first_mode=first_mode,
                                    target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_invalid_arguments(self, target_device_idx, xp):
        for args, kwargs in [(([1.0, 1.0, 2.0], np.zeros((1, 3))), {}),
                             (([1.0, 2.0, 3.0], np.zeros((2, 4))), {}),
                             (([1.0], np.zeros((1, 1))), {}),
                             (([1.0, 2.0], np.zeros((1, 2))), {'first_mode': -1})]:
            with self.assertRaises(ValueError):
                NonlinearCalibration(*args, **kwargs, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_forward_and_invert_match_np_interp(self, target_device_idx, xp):
        calib = self._build(target_device_idx, first_mode=2)
        responses = cpuArray(calib.responses)
        # a modal vector longer than the calibration, with values inside and
        # outside the calibrated range
        values = np.array([3.0, -40.0, 2.5, -7.3, 12.0, 0.4, -5.0])
        forward = cpuArray(calib.forward(xp.asarray(values)))
        invert = cpuArray(calib.invert(xp.asarray(values)))
        for i, v in enumerate(values):
            if 2 <= i < 5:
                np.testing.assert_allclose(forward[i], np.interp(v, AMPLITUDES, responses[i - 2]), rtol=1e-5)
                np.testing.assert_allclose(invert[i], np.interp(v, responses[i - 2], AMPLITUDES), rtol=1e-5)
            else:
                # outside the calibrated modes: unchanged
                self.assertEqual(forward[i], v)
                self.assertEqual(invert[i], v)

    @cpu_and_gpu
    def test_invert_undoes_forward(self, target_device_idx, xp):
        calib = self._build(target_device_idx)
        amplitude = xp.array([4.0, -6.0, 9.5])
        np.testing.assert_allclose(cpuArray(calib.invert(calib.forward(amplitude))),
                                   cpuArray(amplitude), rtol=1e-5)

    @cpu_and_gpu
    def test_short_modal_vector(self, target_device_idx, xp):
        # Only the calibrated modes present in the vector are transformed
        calib = self._build(target_device_idx, first_mode=1)
        out = cpuArray(calib.invert(xp.array([20.0, 20.0])))
        np.testing.assert_allclose(out, [20.0, 10.0])

    @cpu_and_gpu
    def test_non_monotonic_modes(self, target_device_idx, xp):
        responses = np.vstack([AMPLITUDES, np.sin(AMPLITUDES / 5.0), AMPLITUDES ** 3])
        calib = NonlinearCalibration(AMPLITUDES, responses, first_mode=3,
                                     target_device_idx=target_device_idx)
        self.assertEqual(calib.non_monotonic_modes(), [4])

    @cpu_and_gpu
    def test_get_and_set_value(self, target_device_idx, xp):
        calib = self._build(target_device_idx)
        new = np.tile(AMPLITUDES * 2, (3, 1))
        calib.set_value(new)
        np.testing.assert_allclose(cpuArray(calib.get_value()), new)

    @cpu_and_gpu
    def test_save_and_restore(self, target_device_idx, xp):
        calib = self._build(target_device_idx, first_mode=4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'nonlinear.fits')
            calib.save(path)
            restored = NonlinearCalibration.restore(path, target_device_idx=target_device_idx)
        self.assertEqual(restored.first_mode, 4)
        np.testing.assert_allclose(cpuArray(restored.amplitudes), cpuArray(calib.amplitudes))
        np.testing.assert_allclose(cpuArray(restored.responses), cpuArray(calib.responses))


if __name__ == '__main__':
    unittest.main()
