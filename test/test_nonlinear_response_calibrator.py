import os
import tempfile
import unittest
from unittest import mock

import specula
specula.init(0)

from specula import np, cpuArray
from specula.base_value import BaseValue
from specula.data_objects.slopes import Slopes
from specula.data_objects.recmat import Recmat
from specula.data_objects.nonlinear_calibration import NonlinearCalibration
from specula.processing_objects.nonlinear_response_calibrator import NonlinearResponseCalibrator

from test.specula_testlib import cpu_and_gpu


AMPLITUDES = [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]
# Three orthogonal interaction vectors, one per mode, with different norms
INTMAT = np.array([[2.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 3.0],
                   [0.0, 0.0, 0.0]])
RECMAT = np.linalg.pinv(INTMAT)


def response(mode, amplitude):
    '''The sensor model of these tests: mode 1 saturates, the others are linear'''
    return 3.0 * np.tanh(amplitude / 3.0) if mode == 1 else amplitude


class TestNonlinearResponseCalibrator(unittest.TestCase):

    def _calibrator(self, d, target_device_idx, nmodes, first_mode=0, recmat=RECMAT, **kwargs):
        return NonlinearResponseCalibrator(nmodes=nmodes, recmat=Recmat(recmat, target_device_idx=target_device_idx),
                                           data_dir=d, calib_tag='nl', first_mode=first_mode,
                                           target_device_idx=target_device_idx, **kwargs)

    def _sweep(self, calibrator, modes, response_fn=response, repeat=2, slopes_fn=None):
        '''Drive the calibrator one mode at a time over AMPLITUDES. The slopes are
        slopes_fn(mode, amplitude), by default the intmat column scaled by response_fn'''
        if slopes_fn is None:
            slopes_fn = lambda mode, amplitude: response_fn(mode, amplitude) * INTMAT[:, mode]
        xp = calibrator.xp
        commands = BaseValue(value=xp.zeros(INTMAT.shape[1]), target_device_idx=calibrator.target_device_idx)
        slopes = Slopes(length=INTMAT.shape[0], target_device_idx=calibrator.target_device_idx)
        calibrator.inputs['in_commands'].set(commands)
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.setup()
        t = 0
        for mode in modes:
            for amplitude in AMPLITUDES * repeat:
                t += 1
                commands.value[:] = 0
                commands.value[mode] = amplitude
                slopes.slopes[:] = xp.asarray(slopes_fn(mode, amplitude))
                commands.generation_time = slopes.generation_time = t
                calibrator.check_ready(t)
                calibrator.trigger()
                calibrator.post_trigger()
        calibrator.finalize()

    @cpu_and_gpu
    def test_measures_the_response_curves(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as d:
            self._sweep(self._calibrator(d, target_device_idx, nmodes=3), modes=[0, 1, 2])
            calib = NonlinearCalibration.restore(os.path.join(d, 'nl.fits'))
        self.assertEqual(calib.first_mode, 0)
        np.testing.assert_allclose(cpuArray(calib.amplitudes), AMPLITUDES)
        expected = [[response(m, a) for a in AMPLITUDES] for m in range(3)]
        np.testing.assert_allclose(cpuArray(calib.responses), expected, rtol=1e-6)

    @cpu_and_gpu
    def test_response_is_the_linear_reconstruction(self, target_device_idx, xp):
        # A recmat with non-orthogonal rows, and slopes that change shape with the
        # amplitude: the response is the mode's entry of recmat @ slopes
        rng = np.random.default_rng(0)
        recmat = rng.standard_normal(RECMAT.shape)
        extra = rng.standard_normal(INTMAT.shape[0])

        def slopes_fn(mode, amplitude):
            return response(mode, amplitude) * INTMAT[:, mode] + 0.1 * amplitude ** 2 * extra

        with tempfile.TemporaryDirectory() as d:
            self._sweep(self._calibrator(d, target_device_idx, nmodes=3, recmat=recmat),
                        modes=[0, 1, 2], slopes_fn=slopes_fn)
            calib = NonlinearCalibration.restore(os.path.join(d, 'nl.fits'))
        expected = [[(recmat @ slopes_fn(m, a))[m] for a in AMPLITUDES] for m in range(3)]
        np.testing.assert_allclose(cpuArray(calib.responses), expected, rtol=1e-6)

    @cpu_and_gpu
    def test_first_mode_uses_the_recmat_rows_of_the_calibrated_modes(self, target_device_idx, xp):
        # Modes 1 and 2 calibrated: their responses must be reconstructed with
        # recmat rows 1 and 2 (rows 0 and 1 would give zero and the wrong scale)
        with tempfile.TemporaryDirectory() as d:
            self._sweep(self._calibrator(d, target_device_idx, nmodes=2, first_mode=1), modes=[0, 1, 2])
            calib = NonlinearCalibration.restore(os.path.join(d, 'nl.fits'))
        self.assertEqual(calib.first_mode, 1)
        expected = [[response(m, a) for a in AMPLITUDES] for m in (1, 2)]
        np.testing.assert_allclose(cpuArray(calib.responses), expected, rtol=1e-6)

    @cpu_and_gpu
    def test_non_monotonic_response_is_reported(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as d:
            calibrator = self._calibrator(d, target_device_idx, nmodes=1)
            with mock.patch.object(calibrator, 'logger') as logger:
                self._sweep(calibrator, modes=[0], response_fn=lambda m, a: np.sin(a))
            self.assertIn('modes [0]', logger.warning.call_args.args[0])
            self.assertTrue(os.path.exists(os.path.join(d, 'nl.fits')))

    @cpu_and_gpu
    def test_errors(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'not all in the recmat'):
                self._calibrator(d, target_device_idx, nmodes=3, first_mode=1)
            with self.assertRaisesRegex(RuntimeError, 'No calibration samples'):
                self._sweep(self._calibrator(d, target_device_idx, nmodes=1, first_mode=1), modes=[0])
            with self.assertRaisesRegex(RuntimeError, 'not sampled'):
                calibrator = self._calibrator(d, target_device_idx, nmodes=2)
                self._sweep(calibrator, modes=[0])
            open(os.path.join(d, 'nl.fits'), 'w').close()
            with self.assertRaises(FileExistsError):
                self._calibrator(d, target_device_idx, nmodes=1)
            self._calibrator(d, target_device_idx, nmodes=1, overwrite=True)


if __name__ == '__main__':
    unittest.main()
