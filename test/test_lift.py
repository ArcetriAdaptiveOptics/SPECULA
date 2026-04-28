import specula
specula.init(0)

import unittest

import numpy as np

from specula import cpuArray
from specula.lib.calc_psf import calc_psf
from specula.lib.compute_petal_ifunc import compute_petal_ifunc
from specula.data_objects.ifunc import IFunc
from specula.processing_objects.lift import Lift
from specula.data_objects.simul_params import SimulParams
from test.specula_testlib import cpu_and_gpu


def build_lift(n_pistons=0, n_zern=3, fft_res=2, target_device_idx=-1):
    """Build a Lift instance with [n_pistons piston(s) + Zernike modes] IFunc."""
    # Generate Zernike modes: tip, tilt, defocus, ...  (no piston from compute_zern_ifunc)
    zern_ifunc = IFunc(
        type_str='zernike', nmodes=n_zern, npixels=16,
        precision=1, target_device_idx=target_device_idx,
    )
    if n_pistons == 0:
        ifunc = zern_ifunc
    else:
        mask_2d = cpuArray(zern_ifunc.mask_inf_func)
        zern_modes = cpuArray(zern_ifunc.influence_function)   # (n_zern, n_valid)
        n_valid = zern_modes.shape[1]
        piston = np.ones((n_pistons, n_valid), dtype=np.float32) / np.sqrt(n_valid)
        combined = np.vstack([piston, zern_modes]).astype(np.float32)
        ifunc = IFunc(combined, mask=mask_2d, precision=1, target_device_idx=target_device_idx)
    simul_params = SimulParams(pixel_pupil=16, pixel_pitch=1.0)

    return Lift(
        simul_params=simul_params,
        nPistons=n_pistons,
        nZern=n_zern,
        wavelengthInNm=750.0,
        pix_scale=0.01,
        npix_side=16,
        cropped_size=4,
        ifunc=ifunc,
        ref_zern_amp=np.zeros(n_zern, dtype=np.float32),
        fft_res=fft_res,
        target_device_idx=target_device_idx,
        precision=1,
    )


class TestLift(unittest.TestCase):

    @cpu_and_gpu
    def test_has_correct_outputs(self, target_device_idx, xp):
        lift = build_lift(target_device_idx=target_device_idx)
        self.assertEqual(set(lift.outputs.keys()), {'out_pistons', 'out_zern'})
        self.assertEqual(lift.outputs['out_pistons'].value.shape, (lift.nPistons,))
        self.assertEqual(lift.outputs['out_zern'].value.shape, (lift.nZern,))

    @cpu_and_gpu
    def test_radians_per_pixel_matches_geometry_fft_res(self, target_device_idx, xp):
        lift = build_lift(fft_res=3, target_device_idx=target_device_idx)
        settings = Lift.calc_geometry(
            phase_sampling=16,
            pixel_pitch=1.0,
            wavelengthInNm=750.0,
            pix_scale=0.01,
            npix_side=16,
            fft_res=3,
        )
        expected = np.pi / (2.0 * settings.fft_res)
        self.assertAlmostEqual(lift.radians_per_pixel, expected)

    @cpu_and_gpu
    def test_dtype_applied_to_internal_arrays(self, target_device_idx, xp):
        lift = build_lift(target_device_idx=target_device_idx)
        self.assertEqual(lift.airef.dtype, lift.dtype)
        self.assertEqual(lift.out_pistons.value.dtype, lift.dtype)
        self.assertEqual(lift.out_zern.value.dtype, lift.dtype)
        self.assertEqual(lift.mask.dtype, lift.dtype)
        self.assertEqual(lift.modesCube.dtype, lift.dtype)

    @cpu_and_gpu
    def test_ref_zern_amp_populates_refe_coeff_from_tip_order(self, target_device_idx, xp):
        ref_zern_amp = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
        zern_ifunc = IFunc(
            type_str='zernike', nmodes=4, npixels=16,
            precision=1, target_device_idx=target_device_idx,
        )
        mask_2d = cpuArray(zern_ifunc.mask_inf_func)
        zern_modes = cpuArray(zern_ifunc.influence_function)
        n_valid = zern_modes.shape[1]
        piston = np.ones((1, n_valid), dtype=np.float32) / np.sqrt(n_valid)
        ifunc = IFunc(
            np.vstack([piston, zern_modes]).astype(np.float32),
            mask=mask_2d,
            precision=1,
            target_device_idx=target_device_idx,
        )
        simul_params = SimulParams(pixel_pupil=16, pixel_pitch=1.0)

        lift = Lift(
            simul_params=simul_params,
            nPistons=1,
            nZern=4,
            wavelengthInNm=750.0,
            pix_scale=0.01,
            npix_side=16,
            cropped_size=4,
            ifunc=ifunc,
            ref_zern_amp=ref_zern_amp,
            fft_res=2,
            target_device_idx=target_device_idx,
            precision=1,
        )

        airef = cpuArray(lift.airef)

        expected = np.zeros(lift.nmodes, dtype=np.float32)
        expected[1:] = ref_zern_amp
        np.testing.assert_allclose(airef, expected)

    @cpu_and_gpu
    def test_ref_zern_amp_longer_than_nzern_length(self, target_device_idx, xp):
        ifunc = IFunc(
            type_str='zernike', nmodes=3, npixels=16,
            precision=1, target_device_idx=target_device_idx,
        )
        simul_params = SimulParams(pixel_pupil=16, pixel_pitch=1.0)

        with self.assertRaises(ValueError):
            Lift(
                simul_params=simul_params,
                nPistons=0,
                nZern=3,
                wavelengthInNm=750.0,
                pix_scale=0.01,
                npix_side=16,
                cropped_size=4,
                ifunc=ifunc,
                ref_zern_amp=[0.0, 0.2, 0.0, 0.1],  # length 4, but nZern=3
                fft_res=2,
                target_device_idx=target_device_idx,
                precision=1,
            )

    @cpu_and_gpu
    def test_set_ref_tt_uses_image_coordinates(self, target_device_idx, xp):
        lift = build_lift(target_device_idx=target_device_idx)
        lift.radians_per_pixel = 0.2
        lift.setRefTT(center_x=7.0, center_y=5.0, image_size=10.0)
        self.assertAlmostEqual(lift.ref_tip, 0.0)
        self.assertAlmostEqual(lift.ref_tilt, 0.4)

    @cpu_and_gpu
    def test_trigger_updates_separate_outputs(self, target_device_idx, xp):
        lift = build_lift(target_device_idx=target_device_idx)
        fake_psf = np.ones((lift.gridSize, lift.gridSize), dtype=np.float32)
        coeffs = np.arange(lift.nmodes, dtype=np.float32)
        lift.in_pixels = type('_', (), {'get_value': lambda self: fake_psf})()
        lift.phaseEstimation = lambda psf: (lift.xp.zeros_like(lift.mask), coeffs, 1)
        lift.current_time = 123

        lift.trigger()

        np.testing.assert_array_equal(
            specula.cpuArray(lift.outputs['out_pistons'].value), coeffs[:lift.nPistons])
        np.testing.assert_array_equal(
            specula.cpuArray(lift.outputs['out_zern'].value), coeffs[lift.nPistons:])
        self.assertEqual(lift.outputs['out_pistons'].generation_time, 123)
        self.assertEqual(lift.outputs['out_zern'].generation_time, 123)

    @cpu_and_gpu
    def test_compute_cog_available_and_consistent(self, target_device_idx, xp):
        lift = build_lift(target_device_idx=target_device_idx)
        frame = lift.xp.zeros((16, 16), dtype=lift.dtype)
        frame[3, 5] = 10.0
        yc, xc = lift.computeCoG(frame)
        self.assertAlmostEqual(float(yc), 3.0)
        self.assertAlmostEqual(float(xc), 5.0)

    @cpu_and_gpu
    def test_tip_tilt_coherence_raises_when_modes_are_not_slopes(self, target_device_idx, xp):
        """
        _check_tip_tilt_coherence must raise ValueError when the modes at
        positions nPistons and nPistons+1 are not linear slopes.

        Scenario: a Zernike IFunc holds [tip, tilt, defocus, astig] at rows 0-3.
        With nPistons=1 the check inspects rows 1 (tilt, a slope → ok) and 2
        (defocus, radially symmetric → not a slope → ValueError).
        """
        ifunc = IFunc(
            type_str='zernike', nmodes=4, npixels=16,
            precision=1, target_device_idx=target_device_idx,
        )
        simul_params = SimulParams(pixel_pupil=16, pixel_pitch=1.0)
        with self.assertRaises(ValueError):
            Lift(
                simul_params=simul_params,
                nPistons=1,
                nZern=3,
                wavelengthInNm=750.0,
                pix_scale=0.01,
                npix_side=16,
                cropped_size=4,
                ifunc=ifunc,
                ref_zern_amp=[0.0, 0.0, 0.1],
                fft_res=2,
                target_device_idx=target_device_idx,
                precision=1,
            )

    @cpu_and_gpu
    def test_phase_estimation_recovers_defocus(self, target_device_idx, xp):
        """
        Build a noiseless PSF with calc_psf from a known phase (reference
        defocus + a small unknown defocus), feed it to LIFT, and check that
        the estimated defocus coefficient is close to the known value.

        Geometry: with fft_res=1 LIFT's ft_ft2 is just fftshift(fft2(x))/N,
        identical to calc_psf, so pixel scales match with no imwidth padding.
        """
        npixels = 32
        pixel_pitch = 0.5       # m  →  D = 16 m
        wavelengthInNm = 750.0
        n_pistons = 0
        n_zern = 3              # tip, tilt, defocus
        nmodes = n_pistons + n_zern
        ref_zern_amp = np.array([0.0, 0.0, 0.5 * np.pi], dtype=np.float32)
        unknown_rad = 0.35             # rad  — unknown defocus to recover
        unknown_nm = unknown_rad * wavelengthInNm / (2.0 * np.pi)
        defocus_idx = n_pistons + 2   # = 2: defocus position in modal vector

        # Zernike IFunc: rows = [tip, tilt, defocus]
        ifunc_obj = IFunc(
            type_str='zernike', nmodes=n_zern, npixels=npixels,
            precision=1, target_device_idx=target_device_idx,
        )
        influence = cpuArray(ifunc_obj.influence_function)  # (4, n_valid)
        mask_2d   = cpuArray(ifunc_obj.mask_inf_func)       # (32, 32)
        idx = np.where(mask_2d > 0)

        # Total phase = LIFT reference defocus + unknown defocus (in radians)
        coeffs_in = np.zeros(nmodes, dtype=np.float32)
        coeffs_in[n_pistons:] = ref_zern_amp
        coeffs_in[defocus_idx] += unknown_rad
        phase_rad = np.zeros((npixels, npixels), dtype=np.float32)
        phase_rad[idx] = coeffs_in @ influence
        amp = mask_2d.astype(np.float32)

        # PSF via calc_psf  (no imwidth → 32×32, same pixel scale as LIFT
        # with fft_res=1)
        psf = calc_psf(phase_rad, amp, xp=np, complex_dtype=np.complex64)
        psf = psf.astype(np.float32)

        # pix_scale chosen so sampling_ratio = 1.0  →  gridSize = npixels
        rad2arcsec = 206264.806247
        fov_internal = (wavelengthInNm * 1e-9 / pixel_pitch) * rad2arcsec
        pix_scale = fov_internal / npixels

        simul_params = SimulParams(pixel_pupil=npixels, pixel_pitch=pixel_pitch)
        lift = Lift(
            simul_params=simul_params,
            nPistons=n_pistons,
            nZern=n_zern,
            wavelengthInNm=wavelengthInNm,
            pix_scale=pix_scale,
            npix_side=npixels,
            cropped_size=8,
            ifunc=ifunc_obj,
            ref_zern_amp=ref_zern_amp,
            n_iter=30,
            fft_res=1,
            target_device_idx=target_device_idx,
            precision=1,
        )

        _, coeffs_out, _ = lift.phaseEstimation(psf)
        coeffs_out = cpuArray(coeffs_out)

        np.testing.assert_allclose(
            coeffs_out[defocus_idx], unknown_nm,
            atol=20.0,
            err_msg=f"LIFT defocus estimate {coeffs_out[defocus_idx]:.1f} nm, "
                    f"expected {unknown_nm:.1f} nm",
        )

    @cpu_and_gpu
    def test_phase_estimation_recovers_single_petal_piston_with_zernike_basis(self, target_device_idx, xp):
        """
        Build a mixed modal basis with one real petal-piston mode followed by
        tip, tilt, and defocus. Generate a noiseless PSF with a known piston on
        that petal plus a reference Zernike vector, then verify that the piston
        estimate is recovered correctly.
        """
        npixels = 32
        pixel_pitch = 0.5
        wavelengthInNm = 750.0
        n_pistons = 1
        n_zern = 3
        nmodes = n_pistons + n_zern
        piston_idx = 0
        defocus_idx = n_pistons + 2
        ref_zern_amp = np.array([0.0, 0.0, 0.5 * np.pi], dtype=np.float32)
        unknown_rad = 0.35
        unknown_nm = unknown_rad * wavelengthInNm / (2.0 * np.pi)

        petal_modes, mask_2d, _ = compute_petal_ifunc(
            npixels, 6, xp=np, dtype=np.float32
        )
        single_piston = petal_modes[:1]
        zern_ifunc = IFunc(
            type_str='zernike', nmodes=n_zern, npixels=npixels,
            precision=1, target_device_idx=target_device_idx,
        )
        zern_modes = cpuArray(zern_ifunc.influence_function)
        combined = np.vstack([single_piston, zern_modes]).astype(np.float32)
        ifunc_obj = IFunc(
            combined,
            mask=mask_2d.astype(np.float32),
            precision=1,
            target_device_idx=target_device_idx,
        )

        coeffs_in = np.zeros(nmodes, dtype=np.float32)
        coeffs_in[piston_idx] = unknown_rad
        coeffs_in[n_pistons:] = ref_zern_amp
        idx = np.where(mask_2d > 0)
        phase_rad = np.zeros((npixels, npixels), dtype=np.float32)
        phase_rad[idx] = coeffs_in @ combined
        amp = mask_2d.astype(np.float32)
        psf = calc_psf(phase_rad, amp, xp=np, complex_dtype=np.complex64)
        psf = psf.astype(np.float32)

        rad2arcsec = 206264.806247
        fov_internal = (wavelengthInNm * 1e-9 / pixel_pitch) * rad2arcsec
        pix_scale = fov_internal / npixels

        simul_params = SimulParams(pixel_pupil=npixels, pixel_pitch=pixel_pitch)
        lift = Lift(
            simul_params=simul_params,
            nPistons=n_pistons,
            nZern=n_zern,
            wavelengthInNm=wavelengthInNm,
            pix_scale=pix_scale,
            npix_side=npixels,
            cropped_size=8,
            ifunc=ifunc_obj,
            ref_zern_amp=ref_zern_amp,
            n_iter=30,
            fft_res=1,
            target_device_idx=target_device_idx,
            precision=1,
        )

        _, coeffs_out, _ = lift.phaseEstimation(psf)
        coeffs_out = cpuArray(coeffs_out)

        np.testing.assert_allclose(
            coeffs_out[piston_idx], unknown_nm,
            atol=5.0,
            err_msg=f"LIFT piston estimate {coeffs_out[piston_idx]:.1f} nm, "
                    f"expected {unknown_nm:.1f} nm",
        )
        np.testing.assert_allclose(coeffs_out[defocus_idx], 0.0, atol=5.0)

    @cpu_and_gpu
    def test_set_modalbase_mask_dtype_invariance(self, target_device_idx, xp):
        """
        Test that set_modalbase produces the identical modesCube whether 
        mask2d is provided as an integer or float array.
        """
        # 1. Initialize Lift with a simple Zernike IFunc and a ref_zern_amp of length nZern
        npixels = 16
        pixel_pitch = 1.
        diameter = npixels*pixel_pitch
        ref_zern_amp = np.array([0.1, -0.2, 0.3], dtype=np.float32)

        simul_params = SimulParams(pixel_pupil=npixels, pixel_pitch=pixel_pitch)
        ifunc = IFunc(
            type_str='zernike', nmodes=3, npixels=npixels,
            precision=1, target_device_idx=target_device_idx,
        )
        lift = Lift(
            simul_params=simul_params,
            nPistons=0,
            nZern=3,
            wavelengthInNm=1750,
            pix_scale=0.007,
            npix_side=240,
            cropped_size=8,
            ifunc=ifunc,
            ref_zern_amp=ref_zern_amp,
            n_iter=30,
            fft_res=1,
            target_device_idx=target_device_idx,
            precision=1,
        )
    
        # Create an integer mask (e.g., a simple cross/circle shape)
        mask2d_int = np.zeros((npixels,npixels), dtype=np.int32)
        mask2d_int[4:12, 7:9] = 1
    
        # Create the exact same mask, but as a float
        mask2d_float = mask2d_int.astype(np.float64)
    
        # Create a mock modalbase. Its second dimension must match the number 
        # of non-zero elements in the mask (5 in this case).
        valid_pixels = np.count_nonzero(mask2d_int)
        modalbase = np.random.rand(lift.nmodes, valid_pixels)

        # 2. Run the method with the INT mask and store the result
        lift.set_modalbase(modalbase, mask2d_int, diameter)
    
        # Note: If self.xp is cupy, we use .get() to bring it to the CPU for testing. 
        # The lambda acts as a fallback if it's already a numpy array.
        modesCube_int = getattr(lift.modesCube, 'get', lambda: lift.modesCube)()

        # 3. Run the method with the FLOAT mask and store the result
        lift.set_modalbase(modalbase, mask2d_float, diameter)
        modesCube_float = getattr(lift.modesCube, 'get', lambda: lift.modesCube)()

        # 4. Assert the outputs are exactly identical
        # assert_array_equal checks that shapes and elements match perfectly
        np.testing.assert_array_equal(
            modesCube_int, 
            modesCube_float, 
            err_msg="modesCube differs between int and float mask2d inputs."
        )
