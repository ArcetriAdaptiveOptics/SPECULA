import specula
specula.init(0)  # Default target device

import pytest
import unittest
import numpy as np

from specula import cpuArray, RAD2ASEC
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.subap_data import SubapData
from specula.lib.make_mask import make_mask
from specula.processing_objects.ideal_derivative_sensor import IdealDerivativeSensor
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_slopec import ShSlopec
from specula.lib.zernike_generator import ZernikeGenerator
from test.specula_testlib import cpu_and_gpu


class TestIdealDerivativeSensor(unittest.TestCase):

    def create_test_setup(self, target_device_idx, xp):
        """Create common test setup for both SH and IdealDerivativeSensor"""
        # Common parameters
        pixel_pupil = 160
        pixel_pitch = 0.05  # 1m pupil
        wavelength_nm = 500
        subap_on_diameter = 10  # 10x10 subapertures
        subap_npx = 24
        pxscale_arcsec = 0.1
        fov_arcsec = subap_npx * pxscale_arcsec

        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)

        # Create electric field
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1,
                        target_device_idx=target_device_idx)
        ef.A = make_mask(pixel_pupil)

        # Create SH sensor
        sh = SH(wavelengthInNm=wavelength_nm,
                subap_wanted_fov=fov_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=subap_on_diameter,
                subap_npx=subap_npx,
                target_device_idx=target_device_idx)

        # Create SubapData exactly like in test_sh_slopec.py
        idxs = {}
        map_data = {}
        mask_subap = np.ones((subap_on_diameter*subap_npx, subap_on_diameter*subap_npx))

        count = 0
        for i in range(subap_on_diameter):
            for j in range(subap_on_diameter):
                mask_subap *= 0
                mask_subap[i*subap_npx:(i+1)*subap_npx, j*subap_npx:(j+1)*subap_npx] = 1
                idxs[count] = np.where(mask_subap == 1)
                map_data[count] = j * subap_on_diameter + i
                count += 1

        # Convert to required format for SH/ShSlopec
        v = np.zeros((len(idxs), subap_npx*subap_npx), dtype=int)
        m = np.zeros(len(idxs), dtype=int)
        for k, idx in idxs.items():
            v[k] = np.ravel_multi_index(idx, mask_subap.shape)
            m[k] = map_data[k]

        subapdata = SubapData(idxs=v, display_map=m, nx=subap_on_diameter,
                              ny=subap_on_diameter, target_device_idx=target_device_idx)

        # Create IdealDerivativeSensor
        ideal_sensor = IdealDerivativeSensor(simul_params=simul_params,
                                        subapdata=subapdata,
                                        fov=fov_arcsec,
                                        target_device_idx=target_device_idx)

        # Create ShSlopec for processing SH output
        slopec = ShSlopec(subapdata=subapdata, target_device_idx=target_device_idx)

        return ef, sh, ideal_sensor, slopec, subapdata

    @pytest.mark.filterwarnings('ignore:.*divide by zero encountered*:RuntimeWarning')
    @cpu_and_gpu
    def test_flat_wavefront(self, target_device_idx, xp):
        """Test that both sensors give zero slopes for flat wavefront"""
        ef, sh, ideal_sensor, slopec, subapdata = self.create_test_setup(target_device_idx, xp)
        t = 1

        # Set flat wavefront (all zeros)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = t

        # Process with SH
        sh.inputs['in_ef'].set(ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        # Process SH output with slopec
        pixels = Pixels(*sh.outputs['out_i'].i.shape, target_device_idx=target_device_idx)
        pixels.pixels = sh.outputs['out_i'].i
        pixels.generation_time = t

        slopec.inputs['in_pixels'].set(pixels)
        slopec.setup()
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

        # Process with IdealDerivativeSensor
        ideal_sensor.inputs['in_ef'].set(ef)
        ideal_sensor.setup()
        ideal_sensor.check_ready(t)
        ideal_sensor.trigger()
        ideal_sensor.post_trigger()

        # Compare slopes (should be close to zero for both)
        sh_slopes_x = cpuArray(slopec.outputs['out_slopes'].xslopes)
        sh_slopes_y = cpuArray(slopec.outputs['out_slopes'].yslopes)
        ideal_slopes_x = cpuArray(ideal_sensor.outputs['out_slopes'].xslopes)
        ideal_slopes_y = cpuArray(ideal_sensor.outputs['out_slopes'].yslopes)

        # Both should be close to zero
        np.testing.assert_allclose(sh_slopes_x, 0, atol=1e-3)
        np.testing.assert_allclose(sh_slopes_y, 0, atol=1e-3)
        np.testing.assert_allclose(ideal_slopes_x, 0, atol=1e-3)
        np.testing.assert_allclose(ideal_slopes_y, 0, atol=1e-3)

    @pytest.mark.filterwarnings('ignore:.*divide by zero encountered*:RuntimeWarning')
    @cpu_and_gpu
    def test_tilt_comparison(self, target_device_idx, xp):
        """Test tilt case: compare SH and IdealDerivativeSensor slopes"""
        ef, sh, ideal_sensor, slopec, subapdata = self.create_test_setup(target_device_idx, xp)
        t = 1

        # Create Zernike generator for tilt
        zg = ZernikeGenerator(ef.size[0], xp=xp, dtype=ef.dtype)

        # Create X-tilt (Noll index 2) - 1000 nm amplitude
        tilt_amplitude = 500.0  # nm
        x_tilt = zg.getZernike(2) * tilt_amplitude

        print("min max of x_tilt:", x_tilt.min(), x_tilt.max())

        ef.phaseInNm[:] = x_tilt
        ef.generation_time = t

        # Process with SH
        sh.inputs['in_ef'].set(ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        # Process SH output with slopec
        pixels = Pixels(*sh.outputs['out_i'].i.shape, target_device_idx=target_device_idx)
        pixels.pixels = sh.outputs['out_i'].i
        pixels.generation_time = t

        slopec.inputs['in_pixels'].set(pixels)
        slopec.setup()
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

        # Process with IdealDerivativeSensor
        ideal_sensor.inputs['in_ef'].set(ef)
        ideal_sensor.setup()
        ideal_sensor.check_ready(t)
        ideal_sensor.trigger()
        ideal_sensor.post_trigger()

        # Compare slopes
        sh_slopes_x = cpuArray(slopec.outputs['out_slopes'].xslopes)
        ideal_slopes_x = cpuArray(ideal_sensor.outputs['out_slopes'].xslopes)
        ideal_slopes_y = cpuArray(ideal_sensor.outputs['out_slopes'].yslopes)

        # average x and y slope
        sh_avg_x = cpuArray(sh_slopes_x[sh_slopes_x != 0].mean()) if np.any(sh_slopes_x != 0) else 0.0
        ideal_avg_x = cpuArray(ideal_slopes_x[ideal_slopes_x != 0].mean()) if np.any(ideal_slopes_x != 0) else 0.0
        ideal_avg_y = cpuArray(ideal_slopes_y[ideal_slopes_y != 0].mean()) if np.any(ideal_slopes_y != 0) else 0.0

        # For X-tilt, average Y slopes should be near zero
        np.testing.assert_allclose(ideal_avg_y, 0, atol=1e-3)

        # For X-tilt, average X slopes should be close to the expected value
        # Correction to ensure consistent scaling with SH
        # make_xy (used in in the SH slope computer) has spacing 2.0/np_sub while linspace has 2.0/(np_sub-1)
        np_sub = subapdata.np_sub
        spacing_correction = (np_sub - 1) / np_sub
        expected_value = (ef.phaseInNm.max() - ef.phaseInNm.min()) * 1e-9 / \
                         (ef.pixel_pitch*ef.A.shape[0]) * RAD2ASEC / (ideal_sensor.fov/2) \
                         * spacing_correction

        np.testing.assert_allclose(ideal_avg_x, float(expected_value), rtol=1e-2)

        np.testing.assert_allclose(ideal_avg_x, sh_avg_x, rtol=1e-1)

    @cpu_and_gpu
    def test_focus_comparison(self, target_device_idx, xp):
        """Test focus case: compare SH and IdealDerivativeSensor slopes"""
        ef, sh, ideal_sensor, slopec, subapdata = self.create_test_setup(target_device_idx, xp)
        t = 1

        # Create Zernike generator for focus
        zg = ZernikeGenerator(ef.size[0], xp=xp, dtype=ef.dtype)

        # Create focus (Noll index 4) - 200 nm amplitude
        focus_amplitude = 200.0  # nm
        focus = zg.getZernike(4) * focus_amplitude

        ef.phaseInNm[:] = focus
        ef.generation_time = t

        # Process with IdealDerivativeSensor
        ideal_sensor.inputs['in_ef'].set(ef)
        ideal_sensor.setup()
        ideal_sensor.check_ready(t)
        ideal_sensor.trigger()
        ideal_sensor.post_trigger()

        # Compare slopes
        ideal_slopes_x = cpuArray(ideal_sensor.outputs['out_slopes'].xslopes)
        ideal_slopes_y = cpuArray(ideal_sensor.outputs['out_slopes'].yslopes)

        # min and max should be equal with different sign
        np.testing.assert_allclose(ideal_slopes_x.min(), -ideal_slopes_x.max(), rtol=1e-3)
        np.testing.assert_allclose(ideal_slopes_x.min(), ideal_slopes_y.min(), rtol=1e-3)
        np.testing.assert_allclose(ideal_slopes_x.max(), ideal_slopes_y.max(), rtol=1e-3)
