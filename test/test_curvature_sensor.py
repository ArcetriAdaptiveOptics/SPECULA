import specula
specula.init(0)  # Default target device

import unittest
import yaml

from specula import np
from specula import cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.lib.make_mask import make_mask
from specula.lib.zernike_generator import ZernikeGenerator

# Import CWFS modules
from specula.processing_objects.curvature_sensor import CurvatureSensor
from specula.processing_objects.cur_wfs_slopec import CurWfsSlopec
from specula.data_objects.cur_wfs_geometry import CurWfsGeometry

from test.specula_testlib import cpu_and_gpu

class TestCurvatureSensor(unittest.TestCase):

    def _get_config(self):
        # Define a simple geometry:
        # 1 Central button (radius 0.0 -> 0.5)
        # 4 Outer sectors (radius 0.5 -> 1.0)
        rings_config = [
            {'inner': 0.0, 'outer': 0.5, 'segments': 1},
            {'inner': 0.5, 'outer': 1.0, 'segments': 4}
        ]
        return rings_config

    def _get_yaml_string(self):
        """ Returns a YAML string representing a realistic configuration for the CWFS and Slopec.
            This is used to test that the objects can be instantiated from a config dictionary."""
        return """
        geometry:
          size_pixels: 128
          rings_config:
            - inner: 0.0
              outer: 0.5
              segments: 1
            - inner: 0.5
              outer: 1.0
              segments: 4
        
        sensor:
          wavelengthInNm: 700.0
          defocus_rms_nm: 150.0
        
        slopec:
          # Parametri opzionali passati a Slopec
          interleave: False
        """

    @cpu_and_gpu
    def test_geometry_generation(self, target_device_idx, xp):
        """ Test that CurvatureSensorGeometry creates correct masks """
        size = 128
        config = self._get_config()

        geo = CurWfsGeometry(size_pixels=size, rings_config=config,
                                      target_device_idx=target_device_idx)

        # Check dimensions
        expected_subaps = 1 + 4
        self.assertEqual(geo.n_subaps, expected_subaps)
        self.assertEqual(geo.masks.shape, (expected_subaps, size, size))

        # Check masks are valid (binary or float weights) and not empty
        masks_cpu = cpuArray(geo.masks)
        self.assertGreater(np.sum(masks_cpu), 0)

        # Check no overlap between central button and outer ring (sanity check)
        # Mask 0 is center, Mask 1 is first outer sector
        overlap = np.sum(masks_cpu[0] * masks_cpu[1])
        self.assertEqual(overlap, 0, "Sectors should not overlap")

    @cpu_and_gpu
    def test_flat_wavefront_flux(self, target_device_idx, xp):
        """ Test that propagator conserves flux with flat wavefront """
        t = 1
        size = 128
        wavelength = 500.0 # nm
        defocus_rms = 250.0 # nm

        # 1. Create Propagator
        cwfs = CurvatureSensor(wavelengthInNm=wavelength,
                               defocus_rms_nm=defocus_rms,
                               target_device_idx=target_device_idx)

        # 2. Create Flat Input Electric Field
        ef = ElectricField(size, size, 0.1, S0=100, target_device_idx=target_device_idx)
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
        Check if we get a meaningful curvature signal.
        """
        t = 1
        size = 128
        wavelength = 500.0
        defocus_rms = 500.0
        rings_config = self._get_config() # 1 center + 4 outer

        # --- Setup Components ---
        cwfs = CurvatureSensor(wavelengthInNm=wavelength,
                               defocus_rms_nm=defocus_rms,
                               target_device_idx=target_device_idx)

        # Setup Geometry explicitly
        geo = CurWfsGeometry(size_pixels=size, rings_config=rings_config,
                                      target_device_idx=target_device_idx)

        # Setup Slopec passing the geometry object
        slopec = CurWfsSlopec(cwfs_geometry=geo,
                                       target_device_idx=target_device_idx)

        # --- Setup Input with Zernike Focus ---
        ef = ElectricField(size, size, 0.1, S0=100, target_device_idx=target_device_idx)
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
        self.assertEqual(len(slopes), 5) # 1 center + 4 outer

        # 2. Check Physics (Curvature Signal)
        center_signal = slopes[0]
        outer_signals = slopes[1:]

        verbose = False
        if verbose: # pragma: no cover
            print(f"\nCurvature Signals (Focus Input): Center={center_signal:.4f},"
                f" Outer Avg={np.mean(outer_signals):.4f}")

        # Signals should not be zero
        self.assertNotAlmostEqual(center_signal, 0.0)

        # Center signal should have opposite sign (or significantly different magnitude)
        # relative to outer
        self.assertTrue(np.all(np.abs(outer_signals - center_signal) > 0.1))

        # Outer signals should be roughly symmetric
        np.testing.assert_allclose(outer_signals, np.mean(outer_signals), rtol=0.1)

    @cpu_and_gpu
    def test_load_from_yaml(self, target_device_idx, xp):
        """ 
        Verifies that objects can be instantiated directly from a configuration dictionary
        (YAML style).
        """
        # 1. Parsing YAML
        full_config = yaml.safe_load(self._get_yaml_string())

        geo_conf = full_config['geometry']
        sens_conf = full_config['sensor']
        slopec_conf = full_config['slopec']
        size = geo_conf['size_pixels']

        # 2. Geometry instantiation
        # The manager would create this and pass it to Slopec,
        # but here we do it manually for testing.
        geo = CurWfsGeometry(target_device_idx=target_device_idx,
                             **geo_conf)

        verbose = False
        if verbose: # pragma: no cover
            print(f"\n[Config Test] Geometry created with {geo.n_subaps} subaps.")
        self.assertEqual(geo.n_subaps, 5)
        self.assertEqual(geo.size, 128)

        # 3. Sensor instantiation (Propagator)
        cwfs = CurvatureSensor(target_device_idx=target_device_idx,
                               **sens_conf)

        if verbose: # pragma: no cover
            print(f"[Config Test] Sensor created for lambda={cwfs.wavelength_in_nm}nm.")
        self.assertEqual(cwfs.wavelength_in_nm, 700.0)
        self.assertEqual(cwfs.defocus_rms_nm, 150.0)

        # 4. Slopec instantiation (Slope Computer)
        # Here we pass the geometry object explicitly, and the rest of the config via kwargs.
        slopec = CurWfsSlopec(cwfs_geometry=geo,
                              target_device_idx=target_device_idx,
                              **slopec_conf)

        if verbose: # pragma: no cover
            print(f"[Config Test] Slopec linked to geometry with {slopec.nsubaps()} subaps.")
        self.assertEqual(slopec.nsubaps(), 5)

        # Verifies that setup works correctly with these configured objects.

        ef = ElectricField(size, size, 0.1, S0=100, target_device_idx=target_device_idx)
        ef.A = make_mask(size, xp=xp)
        ef.generation_time = 0

        cwfs.inputs['in_ef'].set(ef)
        cwfs.setup()

        i1 = cwfs.outputs['out_i1']
        i2 = cwfs.outputs['out_i2']

        pix1 = Pixels(size, size, target_device_idx=target_device_idx)
        pix1.pixels = i1.i
        pix1.generation_time = 0

        pix2 = Pixels(size, size, target_device_idx=target_device_idx)
        pix2.pixels = i2.i
        pix2.generation_time = 0

        slopec.inputs['in_pixels1'].set(pix1)
        slopec.inputs['in_pixels2'].set(pix2)
        slopec.setup()

        # Verifies that the internal mask matrix in Slopec is correctly built from the geometry
        self.assertIsNotNone(slopec.mask_matrix)
        self.assertEqual(slopec.mask_matrix.shape, (5, 128*128))
