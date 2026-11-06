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
        # Same forward model as the estimator: this checks the inversion only;
        # see test_focus_from_exact_psf_is_unbiased for an independent PSF.
        # Observed errors (with the pixel-centre Interp2D grid, #877): modal
        # -0.06 nm, zonal -0.65 nm. Zonal GS converges slowly and stops early
        # on abs_tol (the error keeps decreasing with more iterations); its
        # looser tolerance is temporary until this is investigated.
        for modal, tol in ((False, 1.0), (True, 0.5)):
            gs = GerchbergSaxton(n_iter=300, modal_projection=modal, **kwargs)
            _, coeffs, _ = gs.phaseEstimation(psf)
            np.testing.assert_allclose(cpuArray(coeffs)[2], 0.3 * 750 / (2 * np.pi), atol=tol,
                                       err_msg=f'modal_projection={modal}')

    @cpu_and_gpu
    def test_focus_from_exact_psf_is_unbiased(self, target_device_idx, xp):
        """
        Same check as test_lift.py::test_focus_from_exact_psf_is_unbiased: the
        PSF is computed independently with numpy (64-pixel pupil, zero padded
        to 128, |FFT|^2, 2x2 binning), so it does not share GS's forward model.

        GS uses the same forward model as LIFT. Observed focus errors for
        unknown = +0.35, 0, -0.35 rad (CPU, pixel-centre Interp2D grid, #877):
        modal +0.94, +0.66, +0.52 nm (LIFT: +1.07, +0.73, +0.60 nm), so modal GS
        gets the same 2 nm tolerance as LIFT. Zonal GS: -1.2, +1.8, +3.8 nm;
        its tolerance stays at 7.5 nm until its slower convergence is
        investigated. (Before #877: zonal +4.1, +3.5, +4.0 nm, modal +5.0,
        +3.9, +3.0 nm.)
        """
        tol_nm = {False: 7.5, True: 2.0}   # keyed by modal_projection
        npixels = 64
        pixel_pitch = 0.5
        wavelengthInNm = 750.0
        n_zern = 3
        ref_zern_amp = np.array([0.0, 0.0, 0.5 * np.pi], dtype=np.float64)
        pix_scale = (wavelengthInNm * 1e-9 / pixel_pitch) * 206264.806247 / npixels

        ifunc_obj = IFunc(type_str='zernike', nmodes=n_zern, npixels=npixels,
                          precision=0, target_device_idx=target_device_idx)
        mask = cpuArray(ifunc_obj.mask_inf_func).astype(np.float64)
        influence = cpuArray(ifunc_obj.influence_function).astype(np.float64)
        modes2d = np.zeros((n_zern, npixels, npixels), dtype=np.float64)
        for k in range(n_zern):
            modes2d[k][mask != 0] = influence[k]

        kwargs = dict(simul_params=SimulParams(pixel_pupil=npixels, pixel_pitch=pixel_pitch),
                      nPistons=0, nZern=n_zern, wavelengthInNm=wavelengthInNm,
                      pix_scale=pix_scale, npix_side=npixels, cropped_size=16,
                      ifunc=ifunc_obj, ref_zern_amp=ref_zern_amp.astype(np.float32),
                      fft_res=1, target_device_idx=target_device_idx, precision=0)

        for unknown_rad in (0.35, 0.0, -0.35):
            unknown_nm = unknown_rad * wavelengthInNm / (2.0 * np.pi)
            c = ref_zern_amp.copy()
            c[2] += unknown_rad
            phase = np.tensordot(c, modes2d, axes=1)
            field = np.zeros((2 * npixels, 2 * npixels), dtype=np.complex128)
            field[:npixels, :npixels] = mask * np.exp(1j * phase)
            intensity = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
            psf = intensity.reshape(npixels, 2, npixels, 2).sum(axis=(1, 3))

            for modal in (False, True):
                gs = GerchbergSaxton(n_iter=300, modal_projection=modal, **kwargs)
                _, coeffs_out, _ = gs.phaseEstimation(psf)
                focus_nm = float(cpuArray(coeffs_out)[2])
                self.assertLess(
                    abs(focus_nm - unknown_nm), tol_nm[modal],
                    msg=f"modal_projection={modal}, unknown {unknown_rad} rad: "
                        f"focus estimate {focus_nm:.2f} nm, expected {unknown_nm:.2f} nm")

    @cpu_and_gpu
    def test_fractional_edge_pixels_are_updated(self, target_device_idx, xp):
        """
        Pixels with 0 < mask <= 0.5 enter the forward model with their
        fractional amplitude, so the zonal GS update must change their phase
        too (the modal fit keeps using mask > 0.5).
        """
        gs = build_gs(target_device_idx=target_device_idx, n_iter=1)
        known = gs.xp.asarray([0.0, 0.0, 0.3])
        psf = cpuArray(gs.focalPlaneImageLIFT(gs.phaseFromCoeffs(known), set_flux=1e6))
        # Add a ring of low-amplitude pixels just outside the pupil
        m = cpuArray(gs.mask)
        inside = m > 0
        ring = np.zeros_like(inside)
        ring[1:, :] |= inside[:-1, :]
        ring[:-1, :] |= inside[1:, :]
        ring[:, 1:] |= inside[:, :-1]
        ring[:, :-1] |= inside[:, 1:]
        ring &= ~inside
        self.assertGreater(int(ring.sum()), 0)
        m = m.copy()
        m[ring] = 0.3
        gs.mask = gs.xp.asarray(m, dtype=gs.dtype)
        estimate, _, _ = gs.phaseEstimation(psf)
        self.assertGreater(float(np.abs(cpuArray(estimate)[ring]).max()), 0.0)

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
            # Same forward model as the estimator (inversion check). Observed max
            # errors: modal 0.44 / 0.39 nm, zonal 1.7 / 0.6 nm (fft_res 2 / 3).
            for modal, tol in ((True, 1.0), (False, 4.0)):
                gs = GerchbergSaxton(n_iter=300, modal_projection=modal, **kwargs)
                _, coeffs, _ = gs.phaseEstimation(psf)
                est = cpuArray(coeffs)[:6]
                est = est - est.mean()
                np.testing.assert_allclose(
                    est, true_nm, atol=tol,
                    err_msg=f'fft_res={fft_res}, modal_projection={modal}')


if __name__ == '__main__':
    unittest.main()
