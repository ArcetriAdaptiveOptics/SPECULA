import specula
specula.init(0)  # Default target device

import unittest

from specula import np, RAD2ASEC
from specula import cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData
from specula.lib.make_mask import make_mask
from specula.lib.zernike_generator import ZernikeGenerator

# Import CWFS modules
from specula.processing_objects.curvature_sensor import CurvatureSensor
from specula.processing_objects.cur_wfs_slopec import CurWfsSlopec

from test.specula_testlib import cpu_and_gpu

class TestCurvatureSensor(unittest.TestCase):

    @cpu_and_gpu
    def test_flat_wavefront_flux(self, target_device_idx, xp):
        """ Test that propagator conserves flux with flat wavefront """
        t = 1
        size = 128
        wavelength = 500.0 # nm
        defocus_rms = 1000.0 # nm
        pxscale = 0.1
        wanted_fov = 12.0

        # 1. Create Propagator
        cwfs = CurvatureSensor(wavelengthInNm=wavelength,
                               wanted_fov=wanted_fov,
                               pxscale=pxscale,
                               number_px=size,
                               defocus_rms_nm=defocus_rms,
                               target_device_idx=target_device_idx)

        # Calculate a pixel pitch that results in a magnification of ~1.0
        # dx = lambda / (N * pxscale_rad)
        req_dx = (wavelength * 1e-9) / (size * (pxscale / RAD2ASEC))

        # 2. Create Flat Input Electric Field
        ef = ElectricField(size, size, req_dx, S0=100, target_device_idx=target_device_idx)
        ef.A = make_mask(size, xp=xp) # Circular pupil
        ef.generation_time = t

        # 3. Trigger
        cwfs.inputs['in_ef'].set(ef)
        cwfs.setup()
        cwfs.check_ready(t)
        cwfs.trigger()
        cwfs.post_trigger()

        i1 = cwfs.outputs['out_i1']
        i2 = cwfs.outputs['out_i2']

        # 4. Check Flux Conservation
        # Sum(I1) should equal InputFlux, and Sum(I2) should equal InputFlux
        input_flux = 100 * ef.masked_area()
        sum1 = xp.sum(i1.i)
        sum2 = xp.sum(i2.i)

        np.testing.assert_allclose(cpuArray(sum1), cpuArray(input_flux), rtol=1e-4)
        np.testing.assert_allclose(cpuArray(sum2), cpuArray(input_flux), rtol=1e-4)

    @cpu_and_gpu
    def test_full_chain_focus_response(self, target_device_idx, xp):
        """ 
        Integration Test: 
        Input Focus Aberration -> Propagator -> Slopec 
        Check if we get a meaningful pixel-by-pixel curvature signal.
        """
        t = 1
        size = 128
        wavelength = 500.0
        defocus_rms = 500.0
        pxscale = 0.1
        wanted_fov = 12.0

        # --- Setup Components ---
        cwfs = CurvatureSensor(wavelengthInNm=wavelength,
                               wanted_fov=wanted_fov,
                               pxscale=pxscale,
                               number_px=size,
                               defocus_rms_nm=defocus_rms,
                               target_device_idx=target_device_idx)

        # Setup Slopec
        slopec = CurWfsSlopec(diameter=int(0.8*size),
                              ccd_size=(size, size),
                              target_device_idx=target_device_idx)

        # Extract valid pupil indices from PupData for later assertions
        mask_ids = slopec.pupdata.ind_pup

        # Calculate a pixel pitch that results in a magnification of ~1.0
        req_dx = (wavelength * 1e-9) / (size * (pxscale / RAD2ASEC))

        # --- Setup Input with Zernike Focus ---
        ef = ElectricField(size, size, req_dx, S0=100, target_device_idx=target_device_idx)
        ef.A = make_mask(size, xp=xp)

        # Add Focus (Z4) to input
        zg = ZernikeGenerator(size, xp=xp, dtype=ef.dtype)
        input_focus_amp = 100.0 # nm RMS of input aberration
        ef.phaseInNm = zg.getZernike(4) * input_focus_amp
        ef.generation_time = t

        # --- Connect & Run ---
        cwfs.inputs['in_ef'].set(ef)
        cwfs.setup()

        cwfs.check_ready(t)
        cwfs.trigger()
        cwfs.post_trigger()

        # Convert Intensity to Pixels manually (Simulate Ideal CCD)
        i1 = cwfs.outputs['out_i1']
        i2 = cwfs.outputs['out_i2']

        pix1 = Pixels(size, size, target_device_idx=target_device_idx)
        pix1.pixels = i1.i
        pix1.generation_time = t

        pix2 = Pixels(size, size, target_device_idx=target_device_idx)
        pix2.pixels = i2.i
        pix2.generation_time = t

        # Connect Pixels to Slopec
        slopec.inputs['in_pixels1'].set(pix1)
        slopec.inputs['in_pixels2'].set(pix2)

        slopec.setup()
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

        # --- Assertions ---
        slopes = cpuArray(slopec.outputs['out_slopes'].slopes)

        # 1. Check Output Shapes
        self.assertEqual(len(slopes), len(mask_ids))

        # 2. Check Physics (Curvature Signal)
        slopes_2d = np.zeros((size, size))
        slopes_2d.flat[cpuArray(mask_ids)] = slopes

        # Signal at the center of the pupil
        center_val = slopes_2d[size//2, size//2]

        # Signal towards the edge of the pupil (but inside the mask)
        edge_val = slopes_2d[size//2, size//2 + int(0.25*size)]

        verbose = False
        if verbose: # pragma: no cover
            print(f"\nCurvature Signals (Focus Input): Center={center_val:.4f},"
                  f" Edge={edge_val:.4f}")

        # Signals should not be zero
        self.assertNotAlmostEqual(center_val, 0.0)

        # The curvature signal generated by a parabola (Focus) is non-zero
        self.assertTrue(np.abs(center_val - edge_val) > 0.001 or np.abs(center_val) > 0.01)
