import specula
specula.init(0)

import unittest

import numpy as np

from specula import cpuArray
from specula.lib.compute_petal_ifunc import compute_petal_ifunc
from specula.data_objects.ifunc import IFunc
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.gerchberg_saxton import GerchbergSaxton
from specula.processing_objects.lift import Lift
from test.specula_testlib import cpu_and_gpu


def build_gs(n_zern=3, fft_res=2, target_device_idx=-1, precision=0, **kwargs):
    ifunc = IFunc(type_str='zernike', nmodes=n_zern, npixels=16,
                  precision=precision, target_device_idx=target_device_idx)
    simul_params = SimulParams(pixel_pupil=16, pixel_pitch=1.0)
    return GerchbergSaxton(
        simul_params=simul_params, nPistons=0, nZern=n_zern,
        wavelengthInNm=750.0, pix_scale=0.01, npix_side=16, cropped_size=8,
        ifunc=ifunc, ref_zern_amp=np.zeros(n_zern, dtype=np.float32),
        fft_res=fft_res, target_device_idx=target_device_idx,
        precision=precision, **kwargs)


def build_petal_case(target_device_idx, fft_res=2, cropped_size=32):
    """6 petal pistons + tip/tilt/defocus, defocus reference as phase diversity."""
    npixels, pixel_pitch, wl = 64, 0.5, 750.0
    n_pistons, n_zern = 6, 3
    petal_modes, mask_2d, _ = compute_petal_ifunc(npixels, n_pistons, xp=np, dtype=np.float32)
    zern = IFunc(type_str='zernike', nmodes=n_zern, npixels=npixels,
                 precision=0, target_device_idx=target_device_idx)
    ifunc = IFunc(np.vstack([petal_modes, cpuArray(zern.influence_function)]).astype(np.float32),
                  mask=mask_2d.astype(np.float32), precision=0,
                  target_device_idx=target_device_idx)
    pix_scale = (wl * 1e-9 / pixel_pitch) * 206264.806247 / npixels * 2 / fft_res
    kwargs = dict(
        simul_params=SimulParams(pixel_pupil=npixels, pixel_pitch=pixel_pitch),
        nPistons=n_pistons, nZern=n_zern, wavelengthInNm=wl, pix_scale=pix_scale,
        npix_side=npixels, cropped_size=cropped_size, ifunc=ifunc,
        ref_zern_amp=np.array([0.0, 0.0, 0.5 * np.pi]), fft_res=fft_res,
        target_device_idx=target_device_idx, precision=0)
    return kwargs


class TestGerchbergSaxton(unittest.TestCase):

    @cpu_and_gpu
    def test_same_interface_as_lift(self, target_device_idx, xp):
        gs = build_gs(target_device_idx=target_device_idx)
        self.assertIsInstance(gs, Lift)
        self.assertEqual(set(gs.outputs.keys()), {'out_pistons', 'out_zern'})
        self.assertEqual(set(gs.inputs.keys()), {'in_pixels'})
        self.assertEqual(GerchbergSaxton.output_names().keys(), Lift.output_names().keys())
        self.assertEqual(GerchbergSaxton.input_names().keys(), Lift.input_names().keys())

    @cpu_and_gpu
    def test_ift_ft2_inverts_ft_ft2(self, target_device_idx, xp):
        gs = build_gs(target_device_idx=target_device_idx)
        rng = np.random.default_rng(0)
        E = gs.xp.asarray(rng.normal(size=(gs.gridSize, gs.gridSize))
                          + 1j * rng.normal(size=(gs.gridSize, gs.gridSize)))
        np.testing.assert_allclose(cpuArray(gs.ift_ft2(gs.ft_ft2(E))), cpuArray(E), atol=1e-10)

    @cpu_and_gpu
    def test_unit_ratio_maps_to_unit_ratio(self, target_device_idx, xp):
        for fft_res in (2, 3):
            gs = build_gs(target_device_idx=target_device_idx, fft_res=fft_res)
            ones = gs.xp.ones((gs.npix_side, gs.npix_side), dtype=gs.dtype)
            r = gs._ratio_to_fft_grid(ones)
            self.assertEqual(r.shape, (gs.fftSize, gs.fftSize))
            np.testing.assert_allclose(cpuArray(r), 1.0, atol=1e-6)

    @cpu_and_gpu
    def test_fit_modes_recovers_coefficients(self, target_device_idx, xp):
        gs = build_gs(target_device_idx=target_device_idx)
        c = np.array([0.3, -0.2, 0.5])
        np.testing.assert_allclose(cpuArray(gs.fitModes(gs.phaseFromCoeffs(c))), c, atol=1e-6)

    @cpu_and_gpu
    def test_trigger_updates_separate_outputs(self, target_device_idx, xp):
        gs = build_gs(target_device_idx=target_device_idx)
        fake_psf = np.ones((gs.gridSize, gs.gridSize), dtype=np.float32)
        coeffs = np.arange(gs.nmodes, dtype=np.float32)
        gs.local_inputs['in_pixels'] = type('_', (), {'get_value': lambda self: fake_psf})()
        gs.phaseEstimation = lambda psf: (gs.xp.zeros_like(gs.mask), coeffs, 1)
        gs.current_time = 123
        gs.trigger()
        np.testing.assert_array_equal(cpuArray(gs.outputs['out_zern'].value), coeffs)
        self.assertEqual(gs.outputs['out_zern'].generation_time, 123)

    @cpu_and_gpu
    def test_trigger_output_precision(self, target_device_idx, xp):
        for precision, expected in ((1, np.float32), (0, np.float64)):
            gs = build_gs(target_device_idx=target_device_idx, precision=precision, n_iter=3)
            fake_psf = np.ones((gs.npix_side, gs.npix_side), dtype=np.float32)
            gs.local_inputs['in_pixels'] = type('_', (), {'get_value': lambda self: fake_psf})()
            gs.current_time = 1
            gs.trigger()
            self.assertEqual(cpuArray(gs.outputs['out_zern'].value).dtype, expected)

    @cpu_and_gpu
    def test_recovers_defocus(self, target_device_idx, xp):
        # A reference defocus is required as phase diversity: without it the
        # sign of an even aberration is ambiguous (twin image), for GS as for LIFT.
        npixels = 64
        ifunc = IFunc(type_str='zernike', nmodes=3, npixels=npixels,
                      precision=0, target_device_idx=target_device_idx)
        pix_scale = (750e-9 / 0.5) * 206264.806247 / npixels
        ref = np.array([0.0, 0.0, 0.5 * np.pi])
        kwargs = dict(simul_params=SimulParams(pixel_pupil=npixels, pixel_pitch=0.5),
                      nPistons=0, nZern=3, wavelengthInNm=750.0, pix_scale=pix_scale,
                      npix_side=npixels, cropped_size=32, ifunc=ifunc, ref_zern_amp=ref,
                      fft_res=2, target_device_idx=target_device_idx, precision=0)
        gen = Lift(**kwargs)
        known = ref + np.array([0.0, 0.0, 0.3])
        psf = cpuArray(gen.focalPlaneImageLIFT(gen.phaseFromCoeffs(known) - gen.phase_ref, set_flux=1e6))
        gs = GerchbergSaxton(n_iter=300, **kwargs)
        _, coeffs, _ = gs.phaseEstimation(psf)
        np.testing.assert_allclose(cpuArray(coeffs)[2], 0.3 * 750 / (2 * np.pi), atol=3.0)

    @cpu_and_gpu
    def test_recovers_petal_pistons_like_lift(self, target_device_idx, xp):
        """GS (zonal and modal) and LIFT recover the same differential pistons."""
        for fft_res in (2, 3):
            kwargs = build_petal_case(target_device_idx, fft_res=fft_res)
            gen = Lift(**kwargs)
            pist = np.array([0.4, -0.3, 0.2, 0.0, -0.5, 0.2])
            c = np.zeros(gen.nmodes)
            c[:6] = pist
            c[6:] = [0.0, 0.0, 0.5 * np.pi]
            psf = cpuArray(gen.focalPlaneImageLIFT(gen.phaseFromCoeffs(c) - gen.phase_ref, set_flux=1e6))
            true_nm = pist * 750 / (2 * np.pi)
            true_nm -= true_nm.mean()
            for modal, tol in ((True, 2.0), (False, 15.0)):
                gs = GerchbergSaxton(n_iter=300, modal_projection=modal, **kwargs)
                _, coeffs, _ = gs.phaseEstimation(psf)
                est = cpuArray(coeffs)[:6]
                est = est - est.mean()
                np.testing.assert_allclose(
                    est, true_nm, atol=tol,
                    err_msg=f'fft_res={fft_res}, modal_projection={modal}')


if __name__ == '__main__':
    unittest.main()
