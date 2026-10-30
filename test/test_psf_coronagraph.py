import specula
specula.init(0)  # Default target device

import unittest

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.psf_coronagraph import PsfCoronagraph
from test.specula_testlib import cpu_and_gpu
from specula import cpuArray, np


class TestPsfCoronagraph(unittest.TestCase):

    @cpu_and_gpu
    def test_empty_pupil(self, target_device_idx, xp):
        # No pupil pixel: the coronagraph subtracts nothing, and the result is finite
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)
        psf_coro = PsfCoronagraph(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                                  nd=1.0, target_device_idx=target_device_idx)
        psf_coro.inputs['in_ef'].set(ef)
        psf_coro.setup()
        ef.A[:] = 0.0
        for use_average_field in (True, False):
            psf_coro.use_average_field = use_average_field
            coro_psf = cpuArray(psf_coro.calc_coronagraph_psf(ef.phaseInNm, ef.A, normalize=False))
            self.assertTrue(np.all(np.isfinite(coro_psf)))
            np.testing.assert_array_equal(coro_psf, 0)

    @unittest.skipIf(specula.cp is None, 'Cupy not found')
    def test_accelerated_matches_normal_run(self):
        from iteras.accelerator import Accelerator
        from iteras.data_objects.accelerator import Accelerator as AcceleratorOptions

        def run(accelerate, use_average_field):
            simul_params, ef, wavelengthInNm = self.get_basic_setup(0, pixel_pupil=32)
            psf_coro = PsfCoronagraph(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                                      nd=2.0, use_average_field=use_average_field,
                                      target_device_idx=0)
            psf_coro.inputs['in_ef'].set(ef)
            if accelerate:
                Accelerator(AcceleratorOptions()).apply([psf_coro])
            psf_coro.setup()
            yy, xx = np.mgrid[0:32, 0:32] / 32.0
            out = []
            for step in range(6):
                t = psf_coro.seconds_to_t(0.001 * step)
                # A field that changes at each step, updated in place
                ef.phaseInNm[:] = specula.cp.asarray(50.0 * np.sin(2 * np.pi * (xx * (step + 1) + yy)))
                ef.generation_time = t
                psf_coro.check_ready(t)
                psf_coro.trigger()
                psf_coro.post_trigger()
                out.append((cpuArray(psf_coro.coronagraph_psf.value).copy(),
                            float(psf_coro.sr.value)))
            psf_coro.finalize()
            return out, psf_coro, cpuArray(psf_coro.int_coronagraph_psf.value)

        for use_average_field in (True, False):
            with self.subTest(use_average_field=use_average_field):
                ref, _, ref_int = run(False, use_average_field)
                out, psf_coro, out_int = run(True, use_average_field)
                self.assertEqual(psf_coro._accel.status, 'accelerated')
                for (a, sr_a), (b, sr_b) in zip(ref, out):
                    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-7 * a.max())
                    self.assertAlmostEqual(sr_a, sr_b, places=6)
                np.testing.assert_allclose(out_int, ref_int, rtol=1e-5, atol=1e-7 * ref_int.max())
                # Different steps give different PSFs (the graph is not frozen)
                self.assertGreater(np.abs(out[1][0] - out[4][0]).max(), 1e-3 * out[1][0].max())

    def get_basic_setup(self, target_device_idx, pixel_pupil=16):
        """Create basic setup for coronagraph tests"""
        pixel_pitch = 0.05
        wavelengthInNm = 500.0

        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)

        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1.0,
                           target_device_idx=target_device_idx)

        return simul_params, ef, wavelengthInNm

    @cpu_and_gpu
    def test_initialization_and_output_names(self, target_device_idx, xp):
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)

        psf_coro = PsfCoronagraph(simul_params=simul_params,
                                  wavelengthInNm=wavelengthInNm,
                                  nd=2.0,
                                  target_device_idx=target_device_idx)

        self.assertEqual(psf_coro.nd, 2.0)
        names = PsfCoronagraph.output_names()
        self.assertIn('out_coronagraph_psf', names)
        self.assertIn('out_int_coronagraph_psf', names)
        self.assertIn('out_std_coronagraph_psf', names)
        self.assertIn('out_coronagraph_psf_profile', names)
        self.assertIn('out_int_coronagraph_psf_profile', names)
        self.assertIn('out_std_coronagraph_psf_profile', names)

    @cpu_and_gpu
    def test_calc_coronagraph_psf_flat_wavefront(self, target_device_idx, xp):
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)

        psf_coro = PsfCoronagraph(simul_params=simul_params,
                                  wavelengthInNm=wavelengthInNm,
                                  nd=1.0,
                                  target_device_idx=target_device_idx)
        
        psf_coro.inputs['in_ef'].set(ef)
        psf_coro.setup()

        ef.phaseInNm[:] = 0.0
        ef.A[:] = 1.0

        coro_psf = psf_coro.calc_coronagraph_psf(ef.phaseInNm,
                                                 ef.A,
                                                 normalize=False)

        self.assertEqual(coro_psf.shape, ef.A.shape)
        self.assertAlmostEqual(float(xp.max(coro_psf)), 0.0, places=8)
        self.assertAlmostEqual(float(xp.sum(coro_psf)), 0.0, places=8)

    @cpu_and_gpu
    def test_use_average_field_toggle(self, target_device_idx, xp):
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)
        ef.phaseInNm[:] = 0.0
        ef.A[:] = 1.0

        psf_avg = PsfCoronagraph(simul_params=simul_params,
                                  wavelengthInNm=wavelengthInNm,
                                  nd=1.0,
                                  use_average_field=True,
                                  target_device_idx=target_device_idx)
        psf_avg.inputs['in_ef'].set(ef)
        psf_avg.setup()

        psf_noavg = PsfCoronagraph(simul_params=simul_params,
                                    wavelengthInNm=wavelengthInNm,
                                    nd=1.0,
                                    use_average_field=False,
                                    target_device_idx=target_device_idx)
        psf_noavg.inputs['in_ef'].set(ef)
        psf_noavg.setup()

        coro_avg = psf_avg.calc_coronagraph_psf(ef.phaseInNm, ef.A, normalize=False)
        coro_noavg = psf_noavg.calc_coronagraph_psf(ef.phaseInNm, ef.A, normalize=False)

        self.assertEqual(coro_avg.shape, coro_noavg.shape)
        self.assertAlmostEqual(float(xp.max(coro_avg)), 0.0, places=8)
        self.assertAlmostEqual(float(xp.max(coro_noavg)), 0.0, places=8)

    @cpu_and_gpu
    def test_trigger_flat_wavefront_suppression(self, target_device_idx, xp):
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)

        psf_coro = PsfCoronagraph(simul_params=simul_params,
                                  wavelengthInNm=wavelengthInNm,
                                  nd=1.0,
                                  target_device_idx=target_device_idx)

        psf_coro.inputs['in_ef'].set(ef)
        psf_coro.setup()

        ef.phaseInNm[:] = 0.0
        ef.A[:] = 1.0
        ef.generation_time = 1

        psf_coro.check_ready(1)
        psf_coro.trigger()
        psf_coro.post_trigger()

        self.assertAlmostEqual(float(psf_coro.sr.value), 1.0, places=6)
        coro_max = float(xp.max(psf_coro.coronagraph_psf.value))
        std_max = float(xp.max(psf_coro.psf.value))
        suppression_ratio = coro_max / std_max
        self.assertLess(suppression_ratio, 1e-10)

    @cpu_and_gpu
    def test_finalize_integration_and_std_dev(self, target_device_idx, xp):
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)

        psf_coro = PsfCoronagraph(simul_params=simul_params,
                                  wavelengthInNm=wavelengthInNm,
                                  nd=1.0,
                                  start_time=0.0,
                                  target_device_idx=target_device_idx)

        psf_coro.inputs['in_ef'].set(ef)
        psf_coro.setup()

        for t in range(1, 5):
            ef.phaseInNm[:] = 20.0 * xp.random.randn(*ef.phaseInNm.shape)
            ef.A[:] = 1.0
            ef.generation_time = t

            psf_coro.check_ready(t)
            psf_coro.trigger()
            psf_coro.post_trigger()

        psf_coro.finalize()

        self.assertEqual(psf_coro.count, 4)
        self.assertEqual(psf_coro.int_coronagraph_psf.value.shape, psf_coro.coronagraph_psf.value.shape)
        self.assertEqual(psf_coro.std_coronagraph_psf.value.shape, psf_coro.coronagraph_psf.value.shape)
        self.assertGreaterEqual(float(xp.max(psf_coro.std_coronagraph_psf.value)), 0.0)

    @cpu_and_gpu
    def test_profile_metrics_outputs(self, target_device_idx, xp):
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx, pixel_pupil=40)

        psf_coro = PsfCoronagraph(simul_params=simul_params,
                                  wavelengthInNm=wavelengthInNm,
                                  nd=2.0,
                                  start_time=0.0,
                                  compute_profile_metrics=True,
                                  compute_metrics_in_trigger=True,
                                  target_device_idx=target_device_idx)

        psf_coro.inputs['in_ef'].set(ef)
        psf_coro.setup()

        y, x = xp.ogrid[:ef.size[0], :ef.size[1]]
        phase_pattern = x + 0.5 * y

        for t in range(1, 4):
            ef.phaseInNm[:] = 5.0 * t * phase_pattern
            ef.A[:] = 1.0
            ef.generation_time = t

            psf_coro.check_ready(t)
            psf_coro.trigger()
            psf_coro.post_trigger()

        frame_profile_before_finalize = cpuArray(psf_coro.coronagraph_psf_profile.value.copy())

        psf_coro.finalize()

        profile_data = cpuArray(psf_coro.coronagraph_psf_profile.value)
        int_profile_data = cpuArray(psf_coro.int_coronagraph_psf_profile.value)
        std_profile_data = cpuArray(psf_coro.std_coronagraph_psf_profile.value)

        np.testing.assert_allclose(profile_data, frame_profile_before_finalize)
        self.assertEqual(profile_data.shape[0], 2)
        self.assertEqual(int_profile_data.shape[0], 2)
        self.assertEqual(std_profile_data.shape[0], 2)
        self.assertGreater(float(psf_coro.int_coronagraph_psf.value.max()), 0.0)
        self.assertGreaterEqual(float(psf_coro.std_coronagraph_psf.value.max()), 0.0)
