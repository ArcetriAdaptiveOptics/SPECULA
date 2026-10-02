import specula
specula.init(0)  # Default target device

import logging
import unittest

import numpy as np

from specula import cpuArray
from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft
from specula.lib.zernike_generator import ZernikeGenerator
from test.specula_testlib import cpu_and_gpu

class TestGenerateModalBasis(unittest.TestCase):

    @cpu_and_gpu
    def test_influence_functions_and_mask(self, target_device_idx, xp):
        pupil_pixels = 128
        n_actuators = 8
        obsratio = 0.0
        diaratio = 1.0
        circGeom = False
        angleOffset = 0
        doMechCoupling = False
        couplingCoeffs = [0.31, 0.05]
        doSlaving = False
        slavingThr = 0.1
        dtype = xp.float32

        # Generate zonal influence functions
        influence_functions, pupil_mask, _, _ = compute_zonal_ifunc(
            dim=pupil_pixels,
            n_act=n_actuators,
            circ_geom=circGeom,
            angle_offset=angleOffset,
            do_mech_coupling=doMechCoupling,
            coupling_coeffs=couplingCoeffs,
            do_slaving=doSlaving,
            slaving_thr=slavingThr,
            obsratio=obsratio,
            diaratio=diaratio,
            mask=None,
            xp=xp,
            dtype=dtype
        )

        # Test the dimensions of influence functions and mask
        self.assertEqual(influence_functions.shape[0], n_actuators**2)
        self.assertEqual(influence_functions.shape[1], xp.sum(pupil_mask))
        self.assertEqual(pupil_mask.shape, (pupil_pixels, pupil_pixels))
        self.assertGreater(xp.sum(pupil_mask), 0)

    @cpu_and_gpu
    def test_kl_basis_rms(self, target_device_idx, xp):
        pupil_pixels = 128
        n_actuators = 8
        telescope_diameter = 8.0
        r0 = 0.2
        L0 = 25.0
        zern_modes = 5
        oversampling = 2
        obsratio = 0.4
        diaratio = 1.0
        circGeom = True
        angleOffset = 0
        doMechCoupling = False
        couplingCoeffs = [0.31, 0.05]
        doSlaving = True
        slavingThr = 0.1
        dtype = xp.float32

        # Generate zonal influence functions
        influence_functions, pupil_mask, _, _ = compute_zonal_ifunc(
            dim=pupil_pixels,
            n_act=n_actuators,
            circ_geom=circGeom,
            angle_offset=angleOffset,
            do_mech_coupling=doMechCoupling,
            coupling_coeffs=couplingCoeffs,
            do_slaving=doSlaving,
            slaving_thr=slavingThr,
            obsratio=obsratio,
            diaratio=diaratio,
            mask=None,
            xp=xp,
            dtype=dtype
        )

        # Generate the modal base
        kl_basis, _, _ = make_modal_base_from_ifs_fft(
            pupil_mask=pupil_mask,
            diameter=telescope_diameter,
            influence_functions=influence_functions,
            r0=r0,
            L0=L0,
            zern_modes=zern_modes,
            oversampling=oversampling,
            if_max_condition_number=None,
            xp=xp,
            dtype=dtype
        )

        # Test RMS of each mode
        for i, mode in enumerate(kl_basis):
            rms = xp.sqrt(xp.mean(mode**2))
            self.assertAlmostEqual(float(rms), 1.0, places=2, msg=f"Mode {i+1} RMS is not close to 1.0")


    @cpu_and_gpu
    def test_if_condition_number(self, target_device_idx, xp):
        pupil_pixels = 128
        n_actuators = 8
        telescope_diameter = 8.0
        r0 = 0.2
        L0 = 25.0
        zern_modes = 5
        oversampling = 2
        obsratio = 0.4
        diaratio = 1.0
        circGeom = True
        angleOffset = 0
        doMechCoupling = False
        couplingCoeffs = [0.31, 0.05]
        doSlaving = True
        slavingThr = 0.1
        dtype = xp.float32

        # Generate zonal influence functions
        influence_functions, pupil_mask, _, _ = compute_zonal_ifunc(
            dim=pupil_pixels,
            n_act=n_actuators,
            circ_geom=circGeom,
            angle_offset=angleOffset,
            do_mech_coupling=doMechCoupling,
            coupling_coeffs=couplingCoeffs,
            do_slaving=doSlaving,
            slaving_thr=slavingThr,
            obsratio=obsratio,
            diaratio=diaratio,
            mask=None,
            xp=xp,
            dtype=dtype
        )

        # Generate the modal base
        kl_basis, _, _ = make_modal_base_from_ifs_fft(
            pupil_mask=pupil_mask,
            diameter=telescope_diameter,
            influence_functions=influence_functions,
            r0=r0,
            L0=L0,
            zern_modes=zern_modes,
            oversampling=oversampling,
            if_max_condition_number=1e-18,  # Small enough that all modes are removed
            log_level=logging.INFO,
            xp=xp,
            dtype=dtype
        )

        assert kl_basis.shape[0] == zern_modes

class TestModalBasisFiltModes(unittest.TestCase):
    """Tests for the filt_modes argument of make_modal_base_from_ifs_fft."""

    LOGGER_NAME = 'specula.lib.modal_base_generator'

    def _setup(self, xp, dtype, pupil_pixels=64, n_actuators=10):
        ifs, pupil_mask, _, _ = compute_zonal_ifunc(
            dim=pupil_pixels, n_act=n_actuators, circ_geom=True, angle_offset=0,
            do_mech_coupling=False, coupling_coeffs=[0.31, 0.05],
            do_slaving=True, slaving_thr=0.1, obsratio=0.1, diaratio=1.0,
            mask=None, xp=xp, dtype=dtype)
        idx_mask = xp.where(pupil_mask.ravel())[0]
        return ifs, pupil_mask, idx_mask

    def _zernikes(self, xp, dtype, pupil_mask, idx_mask, noll_list):
        zg = ZernikeGenerator(pupil_mask.shape[0], xp=xp, dtype=dtype)
        modes = [zg.getZernike(j).ravel()[idx_mask] for j in noll_list]
        return xp.stack(modes).astype(dtype)

    def _make(self, xp, dtype, ifs, pupil_mask, zern_modes, filt_modes):
        return make_modal_base_from_ifs_fft(
            pupil_mask=pupil_mask, diameter=8.0, influence_functions=ifs,
            r0=0.2, L0=25.0, zern_modes=zern_modes, oversampling=2,
            filt_modes=filt_modes, if_max_condition_number=None,
            xp=xp, dtype=dtype)

    @staticmethod
    def _max_norm_corr(a, b, xp):
        """Max abs normalized correlation between rows of a and rows of b."""
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        a = a / xp.linalg.norm(a, axis=1, keepdims=True)
        b = b / xp.linalg.norm(b, axis=1, keepdims=True)
        return float(xp.max(xp.abs(a @ b.T)))

    @cpu_and_gpu
    def test_none_and_empty_filt_modes_are_identical(self, target_device_idx, xp):
        dtype = xp.float32
        ifs, mask, _ = self._setup(xp, dtype)
        for zern_modes in (0, 3):
            kl_none, m2c_none, _ = self._make(xp, dtype, ifs, mask, zern_modes, None)
            empty = xp.zeros((0, int(xp.sum(mask))), dtype=dtype)
            kl_empty, m2c_empty, _ = self._make(xp, dtype, ifs, mask, zern_modes, empty)
            self.assertEqual(kl_none.shape, kl_empty.shape)
            np.testing.assert_array_equal(cpuArray(kl_none), cpuArray(kl_empty))
            np.testing.assert_array_equal(cpuArray(m2c_none), cpuArray(m2c_empty))
            # Piston removed (-1) and Zernikes included in the basis
            self.assertEqual(kl_none.shape[0], ifs.shape[0] - 1)

    @cpu_and_gpu
    def test_filt_modes_as_list(self, target_device_idx, xp):
        dtype = xp.float32
        ifs, mask, idx = self._setup(xp, dtype)
        filt = self._zernikes(xp, dtype, mask, idx, [5, 6])
        kl_arr, m2c_arr, _ = self._make(xp, dtype, ifs, mask, 0, filt)
        kl_list, m2c_list, _ = self._make(xp, dtype, ifs, mask, 0, cpuArray(filt).tolist())
        np.testing.assert_allclose(cpuArray(kl_list), cpuArray(kl_arr), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(cpuArray(m2c_list), cpuArray(m2c_arr), rtol=1e-5, atol=1e-6)

    @cpu_and_gpu
    def test_kl_orthogonal_to_filt_modes(self, target_device_idx, xp):
        dtype = xp.float32
        ifs, mask, idx = self._setup(xp, dtype)
        filt = self._zernikes(xp, dtype, mask, idx, [5, 6, 7])
        for zern_modes in (0, 3):
            kl_basis, _, _ = self._make(xp, dtype, ifs, mask, zern_modes, filt)
            kl = kl_basis[zern_modes:]
            self.assertLess(self._max_norm_corr(kl, filt, xp), 1e-3)
            # Zernike part is left untouched
            self.assertEqual(kl_basis.shape[0], zern_modes + kl.shape[0])

    @cpu_and_gpu
    def test_m2c_consistent_with_basis(self, target_device_idx, xp):
        """IF.T @ m2c gives the modes of the basis, piston included"""
        dtype = xp.float32
        ifs, mask, idx = self._setup(xp, dtype)
        filt = self._zernikes(xp, dtype, mask, idx, [5, 6, 7])
        for zern_modes in (0, 3):
            for filt_modes in (None, filt):
                with self.subTest(zern_modes=zern_modes, filt=filt_modes is not None):
                    kl_basis, m2c, _ = self._make(xp, dtype, ifs, mask, zern_modes, filt_modes)
                    self.assertEqual(m2c.shape, (ifs.shape[0], kl_basis.shape[0]))
                    rec = (ifs.T @ m2c).T
                    peak = float(xp.max(xp.abs(kl_basis)))
                    # Without projection on the IF span (zern_modes=0, no filt_modes)
                    # the part of piston outside the span is left (~7e-5 here)
                    tol = 1e-5 if zern_modes > 0 or filt_modes is not None else 2e-4
                    self.assertLess(float(xp.max(xp.abs(rec - kl_basis))) / peak, tol)

    @cpu_and_gpu
    def test_m2c_has_no_piston(self, target_device_idx, xp):
        dtype = xp.float32
        ifs, mask, idx = self._setup(xp, dtype)
        filt = self._zernikes(xp, dtype, mask, idx, [5, 6, 7])
        for zern_modes in (0, 3):
            for filt_modes in (None, filt):
                with self.subTest(zern_modes=zern_modes, filt=filt_modes is not None):
                    kl_basis, m2c, _ = self._make(xp, dtype, ifs, mask, zern_modes, filt_modes)
                    piston = (ifs.T @ m2c).mean(axis=0)
                    peak = xp.max(xp.abs(kl_basis), axis=1)
                    self.assertLess(float(xp.max(xp.abs(piston) / peak)), 1e-5)

    @cpu_and_gpu
    def test_number_of_modes(self, target_device_idx, xp):
        dtype = xp.float32
        ifs, mask, idx = self._setup(xp, dtype)
        filt = self._zernikes(xp, dtype, mask, idx, [5, 6, 7, 8])
        for zern_modes in (0, 3):
            kl_ref, _, _ = self._make(xp, dtype, ifs, mask, zern_modes, None)
            for n_filt in (1, 4):
                kl, m2c, _ = self._make(xp, dtype, ifs, mask, zern_modes, filt[:n_filt])
                self.assertEqual(kl.shape[0], kl_ref.shape[0] - n_filt)
                self.assertEqual(m2c.shape[1], kl.shape[0])

    @cpu_and_gpu
    def test_filt_mode_outside_if_span_is_dropped(self, target_device_idx, xp):
        dtype = xp.float32
        ifs, mask, _ = self._setup(xp, dtype)
        kl_ref, _, _ = self._make(xp, dtype, ifs, mask, 0, None)
        # Random vector minus its least-squares projection on the IFs (in float64)
        ifs64 = cpuArray(ifs).astype(np.float64)
        v = np.random.default_rng(0).normal(size=ifs64.shape[1])
        v -= ifs64.T @ np.linalg.lstsq(ifs64.T, v, rcond=None)[0]
        filt = xp.asarray(v[None, :], dtype=dtype)
        with self.assertLogs(self.LOGGER_NAME, level=logging.WARNING) as cm:
            kl, _, _ = self._make(xp, dtype, ifs, mask, 0, filt)
        self.assertTrue(any('filt_modes[0] is outside the IF span' in m for m in cm.output))
        self.assertEqual(kl.shape[0], kl_ref.shape[0])

    @cpu_and_gpu
    def test_filt_mode_duplicating_zernike_is_dropped(self, target_device_idx, xp):
        dtype = xp.float32
        ifs, mask, idx = self._setup(xp, dtype)
        zern_modes = 3
        kl_ref, _, _ = self._make(xp, dtype, ifs, mask, zern_modes, None)
        # Noll 3 is already in zern_modes; Noll 5 is a genuine new mode
        filt = self._zernikes(xp, dtype, mask, idx, [3, 5])
        with self.assertLogs(self.LOGGER_NAME, level=logging.WARNING) as cm:
            kl, _, _ = self._make(xp, dtype, ifs, mask, zern_modes, filt)
        self.assertEqual(len(cm.output), 1)
        self.assertIn('filt_modes[0] is in the span of piston/Zernike', cm.output[0])
        self.assertEqual(kl.shape[0], kl_ref.shape[0] - 1)

    @cpu_and_gpu
    def test_duplicated_filt_mode_is_dropped(self, target_device_idx, xp):
        dtype = xp.float32
        ifs, mask, idx = self._setup(xp, dtype)
        kl_ref, _, _ = self._make(xp, dtype, ifs, mask, 0, None)
        z = self._zernikes(xp, dtype, mask, idx, [5, 6])
        filt = xp.vstack((z[0:1], z[0:1], z[1:2]))
        with self.assertLogs(self.LOGGER_NAME, level=logging.WARNING) as cm:
            kl, _, _ = self._make(xp, dtype, ifs, mask, 0, filt)
        self.assertEqual(len(cm.output), 1)
        self.assertIn('filt_modes[1] is in the span of piston/Zernike', cm.output[0])
        self.assertEqual(kl.shape[0], kl_ref.shape[0] - 2)

    @cpu_and_gpu
    def test_wrong_filt_modes_shape_raises(self, target_device_idx, xp):
        dtype = xp.float32
        ifs, mask, _ = self._setup(xp, dtype)
        npix = int(xp.sum(mask))
        with self.assertRaises(ValueError):
            self._make(xp, dtype, ifs, mask, 0, xp.ones((2, npix + 1), dtype=dtype))
        with self.assertRaises(ValueError):
            self._make(xp, dtype, ifs, mask, 0, xp.ones(npix, dtype=dtype))
        # A 1-D empty array is a wrong shape too, not "no filt_modes"
        with self.assertRaises(ValueError):
            self._make(xp, dtype, ifs, mask, 0, xp.zeros(0, dtype=dtype))
