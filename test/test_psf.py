import specula
specula.init(0)  # Default target device

import unittest
import numpy as np

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.psf import PSF
from specula.processing_objects.psf_coronagraph import PsfCoronagraph
from test.specula_testlib import cpu_and_gpu
from specula import cpuArray


class TestPSF(unittest.TestCase):

    def get_basic_setup(self, target_device_idx, pixel_pupil=20):
        """Create basic setup for PSF tests"""
        pixel_pitch = 0.05
        wavelengthInNm = 500.0

        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)

        # Create electric field
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1,
                           target_device_idx=target_device_idx)

        return simul_params, ef, wavelengthInNm

    @cpu_and_gpu
    def test_psf_initialization(self, target_device_idx, xp):
        """Test PSF object initialization with different parameters"""
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)

        # Test with nd parameter
        psf = PSF(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                  nd=2.0, target_device_idx=target_device_idx)
        self.assertEqual(psf.nd, 2.0)

        # Test with pixel_size_mas parameter
        psf_mas = PSF(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                      pixel_size_mas=10.0, target_device_idx=target_device_idx)
        self.assertIsNotNone(psf_mas.nd)

        # Test that both nd and pixel_size_mas cannot be set
        with self.assertRaises(ValueError):
            PSF(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                nd=2.0, pixel_size_mas=10.0, target_device_idx=target_device_idx)

        # Test invalid wavelength
        with self.assertRaises(ValueError):
            PSF(simul_params=simul_params, wavelengthInNm=-500.0,
                target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_calc_psf_sampling(self, target_device_idx, xp):
        """Test PSF sampling calculation"""
        pixel_pupil = 20
        pixel_pitch = 0.05
        wavelength_nm = 500.0

        # Test normal case
        sampling = PSF.calc_psf_sampling(pixel_pupil, pixel_pitch, wavelength_nm, 10.0)
        self.assertIsInstance(sampling, float)
        self.assertGreater(sampling, 0)

        # Test case where requested pixel size is too large
        dim_pup_in_m = pixel_pupil * pixel_pitch
        max_pixel_size_mas = (wavelength_nm * 1e-9 / dim_pup_in_m * 3600 * 180 / np.pi) * 1000

        with self.assertRaises(ValueError):
            PSF.calc_psf_sampling(pixel_pupil, pixel_pitch, wavelength_nm, max_pixel_size_mas * 2)

    @cpu_and_gpu
    def test_psf_with_zero_phase(self, target_device_idx, xp):
        """Test PSF calculation with zero phase - should give SR = 1"""
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)

        # Create PSF object
        psf = PSF(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                  nd=1.0, target_device_idx=target_device_idx)

        # Set up inputs
        psf.inputs['in_ef'].set(ef)
        psf.setup()

        # Zero phase (flat wavefront)
        ef.phaseInNm[:] = 0.0
        ef.A[:] = 1.0
        ef.generation_time = 1

        # Trigger PSF calculation
        psf.check_ready(1)
        psf.trigger()
        psf.post_trigger()

        # With zero phase, SR should be 1 (perfect wavefront)
        self.assertAlmostEqual(float(psf.sr.value), 1.0, places=6)

        # PSF should be normalized
        self.assertAlmostEqual(float(xp.sum(psf.psf.value)), 1.0, places=6)

    @cpu_and_gpu
    def test_psf_with_tilt(self, target_device_idx, xp):
        """Test PSF calculation with a simple tilt"""
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)

        psf = PSF(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                  nd=1.0, target_device_idx=target_device_idx)

        psf.inputs['in_ef'].set(ef)
        psf.setup()

        # Apply a tilt (linear phase)
        y, x = xp.ogrid[:ef.size[0], :ef.size[1]]
        tilt_phase = 0.1 * x  # Small tilt in nm
        ef.phaseInNm[:] = tilt_phase
        ef.A[:] = 1.0
        ef.generation_time = 1

        psf.check_ready(1)
        psf.trigger()
        psf.post_trigger()

        # SR should be less than 1 due to tilt
        self.assertLess(float(psf.sr.value), 1.0)
        self.assertGreater(float(psf.sr.value), 0.0)

    @cpu_and_gpu
    def test_psf_integration(self, target_device_idx, xp):
        """Test PSF integration over multiple frames"""
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)

        psf = PSF(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                  nd=1.0, start_time=0.0, target_device_idx=target_device_idx)

        psf.inputs['in_ef'].set(ef)
        psf.setup()

        # Process multiple frames
        for t in range(1, 4):
            ef.phaseInNm[:] = 0.0  # Flat wavefront
            ef.A[:] = 1.0
            ef.generation_time = t

            psf.check_ready(t)
            psf.trigger()
            psf.post_trigger()

        # Finalize to compute averages
        psf.finalize()

        # Check that integration worked
        self.assertEqual(psf.count, 3)
        self.assertAlmostEqual(float(psf.int_sr.value), 1.0, places=6)

    @cpu_and_gpu
    def test_coronagraph_with_zero_phase(self, target_device_idx, xp):
        """Test coronagraph PSF with zero phase - should give perfect suppression"""
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)

        # Create coronagraph PSF object
        psf_coro = PsfCoronagraph(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                                  nd=1.0, target_device_idx=target_device_idx)

        psf_coro.inputs['in_ef'].set(ef)
        psf_coro.setup()

        # Zero phase (flat wavefront)
        ef.phaseInNm[:] = 0.0
        ef.A[:] = 1.0
        ef.generation_time = 1

        psf_coro.check_ready(1)
        psf_coro.trigger()
        psf_coro.post_trigger()

        # Standard PSF should have SR = 1
        self.assertAlmostEqual(float(psf_coro.sr.value), 1.0, places=6)

        # Coronagraph PSF should be very small (perfect suppression for flat wavefront)
        coro_max = float(xp.max(psf_coro.coronagraph_psf.value))
        std_max = float(xp.max(psf_coro.psf.value))
        suppression_ratio = coro_max / std_max

        # Should have very good suppression (< 1e-10)
        self.assertLess(suppression_ratio, 1e-10)

    @cpu_and_gpu
    def test_coronagraph_with_aber(self, target_device_idx, xp):
        """Test coronagraph PSF with phase aberrations"""
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx, pixel_pupil=100)

        # Add some phase aberrations with random noise
        ef.phaseInNm[:] = 100.0*xp.random.randn(*ef.phaseInNm.shape)
        ef.A[:] = 1.0
        ef.generation_time = 1

        psf_coro = PsfCoronagraph(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                                  nd=3.0, target_device_idx=target_device_idx)

        psf_coro.inputs['in_ef'].set(ef)
        psf_coro.setup()

        psf_coro.check_ready(1)
        psf_coro.trigger()
        psf_coro.post_trigger()

        # Standard PSF should have reduced SR
        self.assertLess(float(psf_coro.sr.value), 1.0)

        # Coronagraph should still provide some suppression
        coro_max = float(xp.max(psf_coro.coronagraph_psf.value))
        std_max = float(xp.max(psf_coro.psf.value))
        suppression_ratio = coro_max / std_max

        plot_debug = False # Set to True to enable plotting
        if plot_debug:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            # Calculate common scale for both images
            std_psf = cpuArray(psf_coro.psf.value)
            coro_psf = cpuArray(psf_coro.coronagraph_psf.value)
            vmin = std_psf[std_psf > 0].min()
            vmax = std_psf.max()
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title("Standard PSF")
            im1 = plt.imshow(std_psf, norm=LogNorm(vmin=vmin, vmax=vmax))
            plt.colorbar(im1)
            plt.subplot(1, 2, 2)
            plt.title("Coronagraph PSF")
            im2 = plt.imshow(coro_psf, norm=LogNorm(vmin=vmin, vmax=vmax))
            plt.colorbar(im2)
            plt.tight_layout()
            plt.show()

        # Should still have suppression, though not perfect
        self.assertLess(suppression_ratio, 1.0)

    @cpu_and_gpu
    def test_calc_psf_method(self, target_device_idx, xp):
        """Test the calc_psf method directly"""
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx)

        psf = PSF(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                  nd=1.0, target_device_idx=target_device_idx)

        # Create test phase and amplitude
        phase = xp.zeros((10, 10))
        amp = xp.ones((10, 10))

        # Test basic PSF calculation
        result = psf.calc_psf(phase, amp, normalize=True)
        self.assertEqual(result.shape, (10, 10))
        self.assertAlmostEqual(float(xp.sum(result)), 1.0, places=6)

        # Test with different output size
        result_big = psf.calc_psf(phase, amp, imwidth=20, normalize=True)
        self.assertEqual(result_big.shape, (20, 20))

        # Test without centering
        result_nocenter = psf.calc_psf(phase, amp, nocenter=True)
        self.assertEqual(result_nocenter.shape, (10, 10))