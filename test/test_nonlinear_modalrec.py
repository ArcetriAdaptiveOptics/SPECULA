import unittest

import specula
specula.init(0)

from specula import np, cp, cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.slopes import Slopes
from specula.data_objects.recmat import Recmat
from specula.data_objects.nonlinear_calibration import NonlinearCalibration
from specula.loop_control import LoopControl
from specula.processing_objects.modalrec import Modalrec
from specula.processing_objects.nonlinear_modalrec import NonlinearModalrec

from test.specula_testlib import cpu_and_gpu


AMPLITUDES = np.linspace(-10.0, 10.0, 41)


def calibration(responses, target_device_idx, first_mode=0):
    return NonlinearCalibration(AMPLITUDES, np.atleast_2d(responses), first_mode=first_mode,
                                target_device_idx=target_device_idx)


def build_rec(recmat_array, calib, target_device_idx, **kwargs):
    recmat = Recmat(recmat_array, target_device_idx=target_device_idx)
    return NonlinearModalrec(recmat=recmat, nonlinear_calib=calib,
                             target_device_idx=target_device_idx, **kwargs)


def run_once(rec, slopes_vec, target_device_idx):
    slopes = Slopes(slopes=slopes_vec, target_device_idx=target_device_idx)
    slopes.generation_time = rec.seconds_to_t(1)
    rec.inputs['in_slopes'].set(slopes)
    rec.setup()
    t = rec.seconds_to_t(1)
    rec.check_ready(t)
    rec.trigger()
    rec.post_trigger()
    return cpuArray(rec.outputs['out_modes'].value).copy()


class TestNonlinearModalrec(unittest.TestCase):

    @cpu_and_gpu
    def test_linear_calibration_gives_modalrec(self, target_device_idx, xp):
        # With linear curves the result is Modalrec's
        rng = np.random.default_rng(0)
        recmat_array = rng.standard_normal((3, 6))
        slopes_vec = xp.asarray(rng.standard_normal(6))
        expected = run_once(Modalrec(Recmat(recmat_array, target_device_idx=target_device_idx),
                                     target_device_idx=target_device_idx), slopes_vec, target_device_idx)
        calib = calibration(np.tile(AMPLITUDES, (3, 1)), target_device_idx)
        rec = build_rec(recmat_array, calib, target_device_idx)
        np.testing.assert_allclose(run_once(rec, slopes_vec, target_device_idx), expected, atol=1e-5)

    @cpu_and_gpu
    def test_desaturates_single_mode_amplitude(self, target_device_idx, xp):
        # The sensor saturates as 5 tanh(a / 5): inverting the curve recovers a
        v = np.array([1.0, 0.0])
        calib = calibration(5.0 * np.tanh(AMPLITUDES / 5.0), target_device_idx)
        true_amplitude = 4.0
        slopes_vec = xp.asarray(5.0 * np.tanh(true_amplitude / 5.0) * v)
        rec = build_rec(v[None], calib, target_device_idx)
        np.testing.assert_allclose(run_once(rec, slopes_vec, target_device_idx), [true_amplitude],
                                   atol=1e-2)

    @cpu_and_gpu
    def test_recovers_a_single_mode_whose_slopes_change_shape(self, target_device_idx, xp):
        # Non-orthogonal interaction vectors, and slopes that saturate and change
        # shape with the amplitude. With the curves measured with the same recmat,
        # as NonlinearResponseCalibrator does, a mode injected alone is recovered
        rng = np.random.default_rng(0)
        intmat_array = rng.standard_normal((12, 3))
        recmat_array = np.linalg.pinv(intmat_array)

        def sensor(amplitudes):
            lin = intmat_array @ amplitudes
            return 8.0 * np.tanh(lin / 8.0) + 0.005 * intmat_array @ amplitudes ** 2

        responses = [[(recmat_array @ sensor(a * np.eye(3)[m]))[m] for a in AMPLITUDES] for m in range(3)]
        rec = build_rec(recmat_array, calibration(responses, target_device_idx), target_device_idx)
        true_amplitude = 4.0
        slopes_vec = xp.asarray(sensor(true_amplitude * np.eye(3)[1]))
        out = run_once(rec, slopes_vec, target_device_idx)
        np.testing.assert_allclose(out[1], true_amplitude, rtol=1e-5)

    @cpu_and_gpu
    def test_modes_outside_the_calibration_are_linear(self, target_device_idx, xp):
        # Two orthogonal modes; only mode 1 is calibrated (first_mode=1)
        intmat_array = np.eye(2)
        calib = calibration(5.0 * np.tanh(AMPLITUDES / 5.0), target_device_idx, first_mode=1)
        rec = build_rec(intmat_array, calib, target_device_idx)
        slopes_vec = xp.array([3.0, 5.0 * np.tanh(4.0 / 5.0)])
        np.testing.assert_allclose(run_once(rec, slopes_vec, target_device_idx), [3.0, 4.0], atol=1e-2)

    @cpu_and_gpu
    def test_invalid_arguments(self, target_device_idx, xp):
        recmat_array = np.ones((2, 3))
        linear = calibration(np.tile(AMPLITUDES, (2, 1)), target_device_idx)
        with self.assertRaisesRegex(ValueError, 'recmat'):
            build_rec(recmat_array, linear, target_device_idx, nmodes=3)
        rolling_over = calibration(np.vstack([AMPLITUDES, np.sin(AMPLITUDES / 5.0)]), target_device_idx)
        with self.assertRaisesRegex(ValueError, r'modes \[1\]'):
            build_rec(recmat_array, rolling_over, target_device_idx)


class SlopesSource(BaseProcessingObj):
    '''Random slopes, drawn on the host at each step'''

    def __init__(self, n, target_device_idx):
        super().__init__(target_device_idx=target_device_idx)
        self.rng = np.random.default_rng(1)
        self.out = Slopes(length=n, target_device_idx=target_device_idx)
        self.outputs['out_slopes'] = self.out

    def trigger_code(self):
        self.out.slopes[:] = self.xp.asarray(self.rng.standard_normal(self.out.slopes.shape[0]))
        self.out.generation_time = self.current_time


@unittest.skipIf(cp is None, 'Cupy not found')
class TestNonlinearModalrecAccelerated(unittest.TestCase):

    def run_loop(self, accelerate, steps=6):
        from iteras.accelerator import Accelerator
        from iteras.data_objects.accelerator import Accelerator as AcceleratorOptions
        rng = np.random.default_rng(0)
        intmat_array = rng.standard_normal((8, 3))
        calib = calibration(np.vstack([3.0 * np.tanh(AMPLITUDES / 3.0)] * 3), 0)
        rec = build_rec(np.linalg.pinv(intmat_array), calib, 0)
        source = SlopesSource(8, 0)
        rec.inputs['in_slopes'].set(source.outputs['out_slopes'])
        loop = LoopControl()
        loop.add(source, 0)
        loop.add(rec, 1)
        if accelerate:
            Accelerator(AcceleratorOptions()).apply([source, rec])
        loop.start(run_time=steps * 1e-3, dt=1e-3)
        out = []
        for _ in range(steps):
            loop.iter()
            out.append(cpuArray(rec.modes.value).copy())
        loop.finish()
        return np.array(out), rec

    def test_accelerated_matches_normal_run(self):
        ref, _ = self.run_loop(False)
        out, rec = self.run_loop(True)
        self.assertIsNotNone(rec._accel.graph)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
