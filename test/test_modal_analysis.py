import os
import sys
import unittest
import warnings
import matplotlib.pyplot as plt

import specula
specula.init(0)  # Default target device

from specula import cpuArray

from specula.data_objects.source import Source
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_infinite_evolution import AtmoInfiniteEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.processing_objects.modal_analysis import ModalAnalysis
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.electric_field import ElectricField
from specula.lib.compute_zern_ifunc import compute_zern_ifunc
from test.specula_testlib import cpu_and_gpu
from skimage.restoration import unwrap_phase

import numpy as np
from unittest.mock import patch

@unittest.skipIf((os.environ.get('CI') == 'true' and
                  sys.platform == 'linux' and
                  sys.version_info[:2] >= (3, 11) and
                  sys.version_info[:2] <= (3, 13)), "Disabled because of CI issues")
class TestModalAnalysisUnwrapping(unittest.TestCase):

    @cpu_and_gpu
    def test_ls_vs_skimage_unwrapping(self, target_device_idx, xp):
        simul_params = SimulParams(zenithAngleInDeg=0.0, pixel_pupil=120,
                                   pixel_pitch=0.01, time_step=1)

        # Atmosphere
        seeing = WaveGenerator(constant=15.0, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0, 0, 0, 0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 0, 0, 0], target_device_idx=target_device_idx)
        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,  # [m] Outer scale
                                     heights=[0., 40., 120., 200.],
                                     Cn2=[0.769, 0.104, 0.127, 0.0],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)

        # Physical and geometrical propagation to source
        uplink_source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=400, wavelengthInNm=1550)
        prop_up = AtmoPropagation(simul_params, source_dict={'uplink_source': uplink_source},
                                  target_device_idx=target_device_idx, wavelengthInNm=1550)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop_up.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])
        for objlist in [[seeing, wind_speed, wind_direction], [atmo], [prop_up]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()

        # unwrapped phase from geometrical propagation
        phase = prop_up.outputs['out_uplink_source_ef'].phi_at_lambda(1550)

        # wrapped phase
        ef = prop_up.outputs['out_uplink_source_ef'].ef_at_lambda(1550)
        wrapped_phase = xp.angle(ef)

        # unwrap phase again
        modal_analysis = ModalAnalysis(npixels=120, nmodes=10, type_str='zernike', wavelengthInNm=1550)
        unwrapped_phase = modal_analysis.unwrap_2d(wrapped_phase)
        unwrapped_phase_skimage = unwrap_phase(cpuArray(wrapped_phase), rng=1)

        rel_error_1 = np.mean(np.abs((cpuArray(phase) - cpuArray(unwrapped_phase))) / np.abs(cpuArray(phase)))
        rel_error_2 = np.mean(np.abs((cpuArray(phase) - cpuArray(unwrapped_phase_skimage))) / np.abs(cpuArray(phase)))

        np.testing.assert_array_less(rel_error_1, rel_error_2)


    @cpu_and_gpu
    def test_modal_analysis_unwrapping_physical_propagation(self, target_device_idx, xp):
        simul_params = SimulParams(zenithAngleInDeg=0.0, pixel_pupil=120,
                                   pixel_pitch=0.01, time_step=1)

        # Atmosphere
        seeing = WaveGenerator(constant=2.5, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0, 0, 0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 0, 0], target_device_idx=target_device_idx)
        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,  # [m] Outer scale
                                     heights=[0., 40., 120.],
                                     Cn2=[0.769, 0.104, 0.127],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)

        # Physical and geometrical propagation to source
        uplink_source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=500,
                               wavelengthInNm=1550)
        prop_up_phys = AtmoPropagation(simul_params, source_dict={'uplink_source': uplink_source},
                                  target_device_idx=target_device_idx, wavelengthInNm=1550, doFresnel=True,
                                         upwards=True, padding_factor=3)
        prop_up_geom = AtmoPropagation(simul_params, source_dict={'uplink_source': uplink_source},
                                  target_device_idx=target_device_idx)

        # Modal analysis
        modal_analsis_phys = ModalAnalysis(npixels=120, nmodes=10,
                                           type_str='zernike', wavelengthInNm=1550)
        modal_analsis_geom = ModalAnalysis(npixels=120, nmodes=10, type_str='zernike')

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop_up_phys.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])
        prop_up_geom.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])
        modal_analsis_phys.inputs['in_ef'].set(prop_up_phys.outputs['out_uplink_source_ef'])
        modal_analsis_geom.inputs['in_ef'].set(prop_up_geom.outputs['out_uplink_source_ef'])
        for objlist in [[seeing, wind_speed, wind_direction], [atmo], \
                        [prop_up_phys, prop_up_geom], [modal_analsis_phys, modal_analsis_geom]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()

        modes_phys = cpuArray(modal_analsis_phys.outputs['out_modes'].value)
        modes_geom = cpuArray(modal_analsis_geom.outputs['out_modes'].value)
        # Use global relative L2 error to reduce sensitivity to backend-specific
        # floating-point differences on single modal coefficients.
        rel_error = np.linalg.norm(modes_phys - modes_geom) / np.linalg.norm(modes_geom)
        np.testing.assert_array_less(rel_error, 0.16)

    @cpu_and_gpu
    def test_modal_analysis_ifunc_inv_nmodes_does_not_mutate_input(self, target_device_idx, xp):
        ifunc_inv_data = xp.random.rand(4, 3).astype(xp.float32)
        mask = xp.ones((2, 2), dtype=xp.uint8)
        ifunc_inv = IFuncInv(ifunc_inv_data, mask,
                             target_device_idx=target_device_idx)
        original_shape = ifunc_inv.size

        modal_analysis = ModalAnalysis(ifunc_inv=ifunc_inv, nmodes=2,
                                       target_device_idx=target_device_idx)

        self.assertEqual(ifunc_inv.size, original_shape)
        self.assertEqual(modal_analysis.phase2modes.size, (4, 2))

    @cpu_and_gpu
    def test_modal_analysis_ifunc_inv_nmodes_none_shares_ifunc_inv_data(self, target_device_idx, xp):
        ifunc_inv_data = xp.random.rand(4, 3).astype(xp.float32)
        mask = xp.ones((2, 2), dtype=xp.uint8)
        ifunc_inv = IFuncInv(ifunc_inv_data, mask,
                             target_device_idx=target_device_idx)

        modal_analysis = ModalAnalysis(ifunc_inv=ifunc_inv, nmodes=None,
                                       target_device_idx=target_device_idx)

        # IFuncInv may cast the input to its own dtype: compare with its internal array
        phase2modes_data = modal_analysis.phase2modes.ifunc_inv
        if hasattr(xp, 'may_share_memory'):
            self.assertTrue(xp.may_share_memory(phase2modes_data, ifunc_inv.ifunc_inv))
        else:
            self.assertEqual(phase2modes_data.data.ptr, ifunc_inv.ifunc_inv.data.ptr)

    @cpu_and_gpu
    def test_modal_analysis_forwards_remove_piston_default(self, target_device_idx, xp):
        ifunc_data = xp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=xp.float32)
        mask = xp.array([[0, 1, 0], [0, 1, 0]], dtype=xp.uint8)
        ifunc = IFunc(ifunc_data, mask=mask, target_device_idx=target_device_idx)
        expected_inv = IFuncInv(xp.zeros((3, 2), dtype=xp.float32), mask=mask,
                                target_device_idx=target_device_idx)

        with patch.object(ifunc, 'inverse', return_value=expected_inv) as inverse_mock:
            ModalAnalysis(ifunc=ifunc, target_device_idx=target_device_idx)

        inverse_mock.assert_called_once_with(nmodes=None, remove_piston=True)

    def _zernike_efs(self, coeffs, npixels, t, target_device_idx, xp):
        """Build one ElectricField per coefficient vector, with the phase set
        to the corresponding combination of Zernike modes"""
        ifunc, mask = compute_zern_ifunc(npixels, nzern=len(coeffs[0]), obsratio=0.0,
                                         diaratio=1.0, xp=xp, dtype=xp.float64)
        idx = xp.where(mask)
        efs = []
        for c in coeffs:
            ef = ElectricField(npixels, npixels, 0.1, target_device_idx=target_device_idx)
            phase = xp.zeros((npixels, npixels), dtype=ef.dtype)
            phase[idx] = xp.dot(c, ifunc)
            ef.phaseInNm[:] = phase
            ef.generation_time = t
            efs.append(ef)
        return efs

    def _run_once(self, obj, t):
        obj.setup()
        obj.check_ready(t)
        obj.trigger()
        obj.post_trigger()

    @cpu_and_gpu
    def test_modal_analysis_list_mode_only(self, target_device_idx, xp):
        """List mode with only in_ef_list connected (in_ef left unset)"""
        npixels = 32
        nmodes = 5
        t = 1

        coeffs = [xp.array([10.0, -20.0, 30.0, 0.0, 5.0]),
                  xp.array([-7.0, 0.0, 15.0, 40.0, -3.0])]
        efs = self._zernike_efs(coeffs, npixels, t, target_device_idx, xp)

        modal_analysis = ModalAnalysis(type_str='zernike', npixels=npixels, nmodes=nmodes,
                                       obsratio=0.0, diaratio=1.0, n_inputs=2,
                                       target_device_idx=target_device_idx)
        modal_analysis.inputs['in_ef_list'].set(efs)
        self._run_once(modal_analysis, t)

        out_list = modal_analysis.outputs['out_modes_list']
        self.assertEqual(len(out_list), 2)
        for out, c in zip(out_list, coeffs):
            self.assertEqual(out.generation_time, t)
            np.testing.assert_allclose(cpuArray(out.value), cpuArray(c), rtol=1e-4, atol=1e-3)

    @cpu_and_gpu
    def test_modal_analysis_debug_log(self, target_device_idx, xp):
        """post_trigger() logs the modes and the RMS of each output"""
        npixels = 32
        nmodes = 5
        t = 1

        coeffs = [xp.array([10.0, -20.0, 30.0, 0.0, 5.0]),
                  xp.array([-7.0, 0.0, 15.0, 40.0, -3.0])]
        efs = self._zernike_efs(coeffs, npixels, t, target_device_idx, xp)

        # Single input
        single = ModalAnalysis(type_str='zernike', npixels=npixels, nmodes=nmodes,
                               obsratio=0.0, diaratio=1.0,
                               target_device_idx=target_device_idx)
        single.inputs['in_ef'].set(efs[0])
        with self.assertLogs('specula.ModalAnalysis', level='DEBUG') as cm:
            self._run_once(single, t)
        msgs = [r.getMessage() for r in cm.records]
        self.assertEqual(sum('First residual values' in m for m in msgs), 1)
        self.assertEqual(sum('Phase RMS' in m for m in msgs), 1)

        # List mode
        multi = ModalAnalysis(type_str='zernike', npixels=npixels, nmodes=nmodes,
                              obsratio=0.0, diaratio=1.0, n_inputs=2,
                              target_device_idx=target_device_idx)
        multi.inputs['in_ef_list'].set(efs)
        with self.assertLogs('specula.ModalAnalysis', level='DEBUG') as cm:
            self._run_once(multi, t)
        msgs = [r.getMessage() for r in cm.records]
        self.assertEqual(sum('First residual values' in m for m in msgs), 2)
        self.assertEqual(sum('Phase RMS' in m for m in msgs), 2)

    # -- Helpers for the precision / RMS tests below -----------

    def _zern_basis(self, npixels, nmodes, xp):
        """Zernike influence functions and pupil pixel indices (double precision,
        independent of the ModalAnalysis object's own dtype)."""
        ifunc, mask = compute_zern_ifunc(npixels, nzern=nmodes, obsratio=0.0,
                                         diaratio=1.0, xp=xp, dtype=xp.float64)
        idx = xp.where(mask)
        return ifunc, idx

    def _set_phase(self, ef, coeffs, ifunc, idx, t):
        """Write a Zernike combination into ef's phase buffer in place."""
        phase = ef.xp.zeros(ef.field[1].shape, dtype=ef.dtype)
        phase[idx] = ef.xp.dot(coeffs, ifunc)
        ef.phaseInNm[:] = phase
        ef.generation_time = t

    def _make_ma(self, npixels, nmodes, target_device_idx,
                n_inputs=1, wavelengthInNm=0.0, precision=None):
        return ModalAnalysis(type_str='zernike', npixels=npixels, nmodes=nmodes,
                             obsratio=0.0, diaratio=1.0, n_inputs=n_inputs,
                             wavelengthInNm=wavelengthInNm,
                             target_device_idx=target_device_idx, precision=precision)

    @cpu_and_gpu
    def test_modal_analysis_precision_dtypes(self, target_device_idx, xp):
        """precision=0/1 must propagate to all outputs and to unwrap_2d()."""
        npixels, nmodes = 20, 4
        ifunc, idx = self._zern_basis(npixels, nmodes, xp)

        for precision in (0, 1):
            expected_dtype = xp.float64 if precision == 0 else xp.float32

            ef = ElectricField(npixels, npixels, 0.1, target_device_idx=target_device_idx)
            self._set_phase(ef, xp.array([1.0, -2.0, 3.0, 0.5]), ifunc, idx, t=1)

            single = self._make_ma(npixels, nmodes, target_device_idx,
                                   precision=precision)
            single.inputs['in_ef'].set(ef)

            ef_list = [ElectricField(npixels, npixels, 0.1, target_device_idx=target_device_idx)
                      for _ in range(2)]
            for e in ef_list:
                self._set_phase(e, xp.array([1.0, -2.0, 3.0, 0.5]), ifunc, idx, t=1)
            multi = self._make_ma(npixels, nmodes, target_device_idx,
                                  n_inputs=2, precision=precision)
            multi.inputs['in_ef_list'].set(ef_list)

            self._run_once(single, 1)
            self._run_once(multi, 1)

            self.assertEqual(single.outputs['out_modes'].value.dtype, expected_dtype)
            self.assertEqual(single.outputs['rms'].value.dtype, expected_dtype)
            for v in multi.outputs['out_modes_list']:
                self.assertEqual(v.value.dtype, expected_dtype)
            for v in multi.outputs['rms_list']:
                self.assertEqual(v.value.dtype, expected_dtype)

            unwrapped = single.unwrap_2d(xp.zeros((npixels, npixels), dtype=expected_dtype))
            self.assertEqual(unwrapped.dtype, expected_dtype)

    @cpu_and_gpu
    def test_modal_analysis_rms_values(self, target_device_idx, xp):
        """rms must equal xp.std() of the (unwrapped, if applicable) phase over
        the mask pixels; in list mode each input has its own, generally
        different, RMS."""
        npixels, nmodes = 20, 4
        ifunc, idx = self._zern_basis(npixels, nmodes, xp)

        efs = [ElectricField(npixels, npixels, 0.1, target_device_idx=target_device_idx)
              for _ in range(2)]
        coeffs_list = [xp.array([10.0, -5.0, 3.0, 1.0]), xp.array([-2.0, 8.0, 0.0, -4.0])]
        for ef, coeffs in zip(efs, coeffs_list):
            self._set_phase(ef, coeffs, ifunc, idx, t=1)

        multi = self._make_ma(npixels, nmodes, target_device_idx, n_inputs=2)
        multi.inputs['in_ef_list'].set(efs)
        self._run_once(multi, 1)

        rms_list = [cpuArray(v.value)[0] for v in multi.outputs['rms_list']]
        self.assertEqual(len(multi.outputs['rms_list']), 2)

        expected = [float(cpuArray(xp.std(ef.phaseInNm[idx]))) for ef in efs]
        np.testing.assert_allclose(rms_list, expected, rtol=1e-5, atol=1e-6)
        # Different phases must give different RMS values.
        self.assertNotAlmostEqual(rms_list[0], rms_list[1], places=3)

        single = self._make_ma(npixels, nmodes, target_device_idx)
        single.inputs['in_ef'].set(efs[0])
        self._run_once(single, 1)
        rms_single = float(cpuArray(single.outputs['rms'].value)[0])
        self.assertAlmostEqual(rms_single, expected[0], places=3)

    @cpu_and_gpu
    def test_modal_analysis_dorms_deprecation(self, target_device_idx, xp):
        """Passing dorms (True or False) must emit a FutureWarning; omitting
        it must not emit any FutureWarning."""
        with self.assertWarns(FutureWarning):
            ModalAnalysis(type_str='zernike', npixels=16, nmodes=3, obsratio=0.0, diaratio=1.0,
                          dorms=True, target_device_idx=target_device_idx)
        with self.assertWarns(FutureWarning):
            ModalAnalysis(type_str='zernike', npixels=16, nmodes=3, obsratio=0.0, diaratio=1.0,
                          dorms=False, target_device_idx=target_device_idx)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ModalAnalysis(type_str='zernike', npixels=16, nmodes=3, obsratio=0.0, diaratio=1.0,
                          target_device_idx=target_device_idx)
        future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
        self.assertEqual(len(future_warnings), 0)
