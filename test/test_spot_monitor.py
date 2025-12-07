import unittest

import specula
specula.init(-1)  # Use CPU for tests

from specula import np, cpuArray

from astropy.modeling import models as _models
from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.spot_monitor import SpotMonitor
from test.specula_testlib import cpu_and_gpu


class TestSpotMonitor(unittest.TestCase):

    def make_subapdata(self, subap_on_diameter=2, subap_npx=12, xp=np):
        # Build idxs and display_map exactly as in other tests
        idxs = {}
        dmap = {}
        mask = xp.ones((subap_on_diameter*subap_npx, subap_on_diameter*subap_npx))
        count = 0
        for i in range(subap_on_diameter):
            for j in range(subap_on_diameter):
                mask *= 0
                mask[i*subap_npx:(i+1)*subap_npx, j*subap_npx:(j+1)*subap_npx] = 1
                idxs[count] = xp.where(mask == 1)
                dmap[count] = j * subap_on_diameter + i
                count += 1

        v = np.zeros((len(idxs), subap_npx*subap_npx), dtype=int)
        m = np.zeros(len(idxs), dtype=int)
        for k, idx in idxs.items():
            v[k] = np.ravel_multi_index((cpuArray(idx[0]), cpuArray(idx[1])), mask.shape)
            m[k] = int(dmap[k])

        subapdata = SubapData(idxs=v, display_map=m, nx=subap_on_diameter, ny=subap_on_diameter)
        return subapdata

    @cpu_and_gpu
    def test_fit_moffat_on_summed_subaps(self, target_device_idx, xp):
        # Geometry
        subap_on_diameter = 2
        np_sub = 12
        dim = subap_on_diameter * np_sub

        subapdata = self.make_subapdata(subap_on_diameter, np_sub, xp=np)

        # Create a single subaperture Moffat + sky patch
        yy, xx = np.mgrid[0:np_sub, 0:np_sub]
        cx = cy = (np_sub - 1) / 2.0
        amp_true = 100.0
        sky_true = 10.0
        gamma_true = 2.5
        alpha_true = 1.7
        
        # Create Moffat + constant background
        patch = _models.Const2D(amplitude=sky_true)(xx, yy) + \
                _models.Moffat2D(amplitude=amp_true, x_0=cx, y_0=cy, 
                               gamma=gamma_true, alpha=alpha_true)(xx, yy)

        # Build the full sensor image (tile same patch into every subap)
        full_img = np.zeros((dim, dim), dtype=np.float64)
        
        # Place patch in each subaperture
        for i in range(subap_on_diameter):
            for j in range(subap_on_diameter):
                y_start = i * np_sub
                y_end = (i + 1) * np_sub
                x_start = j * np_sub
                x_end = (j + 1) * np_sub
                full_img[y_start:y_end, x_start:x_end] = patch

        # Create Pixels object
        pixels = Pixels(dim, dim, target_device_idx=target_device_idx)
        pixels.set_value(full_img)
        pixels.generation_time = 1

        # Run SpotMonitor
        sm = SpotMonitor(subapdata=subapdata, target_device_idx=target_device_idx)
        sm.inputs['in_pixels'].set(pixels)
        sm.check_ready(1)
        sm.trigger()
        sm.post_trigger()

        # Retrieve outputs
        params = cpuArray(sm.outputs['out_params'].value)
        amp_fit, x0_fit, y0_fit, gamma_fit, alpha_fit, sky_fit, fwhm_fit, chi2, success = params

        sum_img = cpuArray(sm.outputs['out_sum_pixels'].pixels)
        model_img = cpuArray(sm.outputs['out_model_pixels'].pixels)
        resid_img = cpuArray(sm.outputs['out_residual_pixels'].pixels)

        # Basic checks
        self.assertEqual(int(success), 1, "Fit did not converge")
        self.assertEqual(sum_img.shape, (np_sub, np_sub))
        self.assertEqual(model_img.shape, (np_sub, np_sub))
        self.assertEqual(resid_img.shape, (np_sub, np_sub))

        # Since we summed 4 identical patches, amplitude and sky scale by 4
        n_subaps = subap_on_diameter * subap_on_diameter
        self.assertAlmostEqual(amp_fit/n_subaps, amp_true, delta=amp_true*0.15)
        self.assertAlmostEqual(sky_fit/n_subaps, sky_true, delta=sky_true*0.3)

        # Shape parameters should be close
        self.assertAlmostEqual(gamma_fit, gamma_true, delta=0.5)
        self.assertAlmostEqual(alpha_fit, alpha_true, delta=0.3)

        # Centroid near center
        self.assertAlmostEqual(x0_fit, cx, delta=0.5)
        self.assertAlmostEqual(y0_fit, cy, delta=0.5)

        # FWHM check
        expected_fwhm = 2 * gamma_true * np.sqrt(2**(1/alpha_true) - 1)
        self.assertAlmostEqual(fwhm_fit, expected_fwhm, delta=0.5)

        # Chi2 should be reasonable
        self.assertLess(chi2, 10.0)

        # Residual RMS should be small vs signal
        self.assertLess(np.sqrt(np.mean(resid_img**2)), 0.15 * np.max(sum_img))

    @cpu_and_gpu
    def test_fit_with_noise(self, target_device_idx, xp):
        """Test fitting with Poisson noise"""
        subap_on_diameter = 2
        np_sub = 12
        dim = subap_on_diameter * np_sub

        subapdata = self.make_subapdata(subap_on_diameter, np_sub, xp=np)

        # Create clean patch
        yy, xx = np.mgrid[0:np_sub, 0:np_sub]
        cx = cy = (np_sub - 1) / 2.0
        amp_true = 500.0  # Higher amplitude for better SNR
        sky_true = 50.0
        gamma_true = 2.0
        alpha_true = 2.5
        
        patch = _models.Const2D(amplitude=sky_true)(xx, yy) + \
                _models.Moffat2D(amplitude=amp_true, x_0=cx, y_0=cy, 
                               gamma=gamma_true, alpha=alpha_true)(xx, yy)

        # Add Poisson noise
        rng = np.random.RandomState(42)
        patch_noisy = rng.poisson(patch)

        # Build full image
        full_img = np.zeros((dim, dim), dtype=np.float64)
        for i in range(subap_on_diameter):
            for j in range(subap_on_diameter):
                y_start = i * np_sub
                y_end = (i + 1) * np_sub
                x_start = j * np_sub
                x_end = (j + 1) * np_sub
                full_img[y_start:y_end, x_start:x_end] = patch_noisy

        pixels = Pixels(dim, dim, target_device_idx=target_device_idx)
        pixels.set_value(full_img)
        pixels.generation_time = 1

        sm = SpotMonitor(subapdata=subapdata, target_device_idx=target_device_idx)
        sm.inputs['in_pixels'].set(pixels)
        sm.check_ready(1)
        sm.trigger()
        sm.post_trigger()

        params = cpuArray(sm.outputs['out_params'].value)
        amp_fit, x0_fit, y0_fit, gamma_fit, alpha_fit, sky_fit, fwhm_fit, chi2, success = params

        # Fit should still converge
        self.assertEqual(int(success), 1, "Fit did not converge with noise")

        # Parameters should be reasonably close (relaxed tolerances for noisy data)
        n_subaps = subap_on_diameter * subap_on_diameter
        self.assertAlmostEqual(amp_fit/n_subaps, amp_true, delta=amp_true*0.3)
        self.assertAlmostEqual(sky_fit/n_subaps, sky_true, delta=sky_true*0.5)
        self.assertAlmostEqual(gamma_fit, gamma_true, delta=1.0)
        self.assertAlmostEqual(alpha_fit, alpha_true, delta=0.8)

    @cpu_and_gpu  
    def test_single_subap(self, target_device_idx, xp):
        """Test with single subaperture (no summing)"""
        subap_on_diameter = 1
        np_sub = 16
        dim = np_sub

        subapdata = self.make_subapdata(subap_on_diameter, np_sub, xp=np)

        yy, xx = np.mgrid[0:np_sub, 0:np_sub]
        cx = cy = (np_sub - 1) / 2.0
        amp_true = 200.0
        sky_true = 20.0
        gamma_true = 3.0
        alpha_true = 1.5
        
        patch = _models.Const2D(amplitude=sky_true)(xx, yy) + \
                _models.Moffat2D(amplitude=amp_true, x_0=cx, y_0=cy, 
                               gamma=gamma_true, alpha=alpha_true)(xx, yy)

        pixels = Pixels(dim, dim, target_device_idx=target_device_idx)
        pixels.set_value(patch)
        pixels.generation_time = 1

        sm = SpotMonitor(subapdata=subapdata, target_device_idx=target_device_idx)
        sm.inputs['in_pixels'].set(pixels)
        sm.check_ready(1)
        sm.trigger()
        sm.post_trigger()

        params = cpuArray(sm.outputs['out_params'].value)
        amp_fit, x0_fit, y0_fit, gamma_fit, alpha_fit, sky_fit, fwhm_fit, chi2, success = params

        self.assertEqual(int(success), 1)
        
        # With single subap, no scaling needed
        self.assertAlmostEqual(amp_fit, amp_true, delta=amp_true*0.1)
        self.assertAlmostEqual(sky_fit, sky_true, delta=sky_true*0.2)
        self.assertAlmostEqual(gamma_fit, gamma_true, delta=0.3)
        self.assertAlmostEqual(alpha_fit, alpha_true, delta=0.2)
