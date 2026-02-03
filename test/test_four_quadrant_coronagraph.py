import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cpuArray
from specula.lib.calc_psf import calc_psf
from specula.lib.make_mask import make_mask
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.four_quadrant_coronagraph import FourQuadrantCoronagraph

from test.specula_testlib import cpu_and_gpu

class TestFourQuadrantCoronagraph(unittest.TestCase):

    def setUp(self):
        # Basic simulation parameters
        self.pixel_pupil = 120
        self.pixel_pitch = 0.05
        self.wavelength_nm = 500

        self.simul_params = SimulParams(
            pixel_pupil=self.pixel_pupil,
            pixel_pitch=self.pixel_pitch
        )
        # make a round mask for the pupil
        self.mask = make_mask(self.pixel_pupil, obsratio=0.0, xp=np)

    def get_coro_field(self, coro, in_ef):
        coro.inputs['in_ef'].set(in_ef)
        coro.setup()
        coro.check_ready(1)
        coro.prepare_trigger(1)
        coro.trigger_code()
        coro.post_trigger()
        return coro.outputs['out_ef']


    def test_raise_value_error(self):
        # Test inner > outer pupil stop size
        with self.assertRaises(ValueError):
            _ = FourQuadrantCoronagraph(
                simul_params=self.simul_params,
                wavelengthInNm=self.wavelength_nm,
                innerStopAsRatioOfPupil=0.9,
                outerStopAsRatioOfPupil=0.1,
            )

    @cpu_and_gpu
    def test_mask_shape(self, target_device_idx, xp):
        """Test that coronagraph masks have the expected shape"""
        coro = FourQuadrantCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            innerStopAsRatioOfPupil=0.0,
            outerStopAsRatioOfPupil=0.9,
            target_device_idx=target_device_idx
        )

        self.assertEqual(coro.pupil_mask.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(coro.fp_mask.shape, (coro.fft_totsize, coro.fft_totsize))
        self.assertEqual(coro.apodizer, 1.0) # no apodizer

        debug_plot = False
        if debug_plot: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cpuArray(xp.angle(coro.fp_mask)), cmap='gray')
            plt.colorbar()
            plt.title('Focal plane mask')
            plt.subplot(1,2,2)
            plt.imshow(cpuArray(coro.pupil_mask), cmap='gray')
            plt.colorbar()
            plt.title('Pupil plane mask')

    @cpu_and_gpu
    def test_output_shape(self, target_device_idx, xp):
        """Test that output ElectricField has expected shape"""
        coro = FourQuadrantCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            innerStopAsRatioOfPupil=0.0,
            outerStopAsRatioOfPupil=0.9,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        out_ef = self.get_coro_field(coro, ef)
        self.assertEqual(out_ef.A.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(out_ef.phaseInNm.shape, (self.pixel_pupil, self.pixel_pupil))


    @cpu_and_gpu
    def test_psf_with_and_without_coronagraph(self, target_device_idx, xp):
        """Test PSF with and without a coronagraph using calc_psf"""
        # No filter (no obstruction)
        nodelay_coro = FourQuadrantCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            phase_delay=0.0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        ef_nocoro = self.get_coro_field(nodelay_coro, ef)

        # With filter: central obstruction of 2 lambda/D
        coro = FourQuadrantCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            target_device_idx=target_device_idx
        )
        ef_coro = self.get_coro_field(coro, ef)

        # Compute PSF for both cases using calc_psf
        nm2rad = 2*xp.pi/self.wavelength_nm
        psf_nocoro = calc_psf(ef_nocoro.phaseInNm*nm2rad, ef_nocoro.A,
                              xp=xp, complex_dtype=xp.complex64, normalize=True)
        psf_coro = calc_psf(ef_coro.phaseInNm*nm2rad, ef_coro.A,
                           xp=xp, complex_dtype=xp.complex64, normalize=True)

        # Check shapes
        self.assertEqual(psf_nocoro.shape, psf_coro.shape)

        # Check that the coronagraph has an effect (PSFs should be smaller)
        diff = np.abs(psf_nocoro - psf_coro).sum()
        self.assertGreater(cpuArray(diff), 0.0, "Coronagraph does not affect the PSF!")

        debug_plot = False
        if debug_plot: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cpuArray(xp.log(psf_nocoro)), cmap='twilight', vmax=0, vmin=-24)
            plt.colorbar()
            plt.title('Input PSF')
            plt.xlim([coro.fft_totsize//2-50,coro.fft_totsize//2+50])
            plt.ylim([coro.fft_totsize//2-50,coro.fft_totsize//2+50])
            plt.subplot(1,2,2)
            plt.imshow(cpuArray(xp.log(psf_coro)), cmap='twilight', vmax=0, vmin=-24)
            plt.colorbar()
            plt.title('Output PSF')
            plt.xlim([coro.fft_totsize//2-50,coro.fft_totsize//2+50])
            plt.ylim([coro.fft_totsize//2-50,coro.fft_totsize//2+50])
            plt.show()

    @cpu_and_gpu
    def test_phase_and_amplitude_preservation(self, target_device_idx, xp):
        """Test that a flat input phase results in a flat output phase
         and a nonzero output amplitude for the no coronagraph case"""
        nocoro = FourQuadrantCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            phase_delay=0.0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        out_ef = self.get_coro_field(nocoro,ef)

        # Output phase should be (almost) constant for a flat input
        mask = ef.A > 0
        valid_phases = out_ef.phaseInNm[mask]
        min_phase = float(xp.min(valid_phases))
        max_phase = float(xp.max(valid_phases))

        # max and min phase should be close to zero (within 5 nm)
        self.assertLess(np.abs(max_phase), 5)
        self.assertLess(np.abs(min_phase), 5)

        # Output amplitude should not be all zeros and should be approximately the same as the input one
        self.assertGreater(float(out_ef.A.sum()), 0.0)
        self.assertLess(float(out_ef.A.max()), 2.0*float(ef.A.max()))

    @cpu_and_gpu
    def test_centering_modes(self, target_device_idx, xp):
        """Test that center_on_pixel affects the PSF symmetry and structure"""

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        # Coronagraph centered on single pixel
        coro_pixel = FourQuadrantCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            target_device_idx=target_device_idx
        )
        coro_pixel.center_on_pixel=True

        # Coronagraph centered on 4-pixel intersection
        coro_4pixel = FourQuadrantCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            target_device_idx=target_device_idx
        )

        ef_pixel = self.get_coro_field(coro_pixel, ef)
        ef_4pixel = self.get_coro_field(coro_4pixel, ef)

        # Compute PSFs
        nm2rad = 2*xp.pi/self.wavelength_nm
        psf_pixel = calc_psf(ef_pixel.phaseInNm*nm2rad, ef_pixel.A,
                            xp=xp, complex_dtype=xp.complex64, normalize=True)
        psf_4pixel = calc_psf(ef_4pixel.phaseInNm*nm2rad, ef_4pixel.A,
                             xp=xp, complex_dtype=xp.complex64, normalize=True)

        # The PSFs should be different
        diff = cpuArray(xp.abs(psf_pixel - psf_4pixel).sum())
        self.assertGreater(diff, 0.0,
                          "center_on_pixel should affect the PSF structure")

        # For 4-quadrant coronagraph, center_on_pixel=False should give better nulling
        # Check the central region suppression
        center = psf_pixel.shape[0] // 2
        size = 10  # central 10x10 pixels

        central_pixel = psf_pixel[center-size:center+size, center-size:center+size]
        central_4pixel = psf_4pixel[center-size:center+size, center-size:center+size]

        # Verify we have data
        self.assertGreater(central_pixel.size, 0, "Empty central region for pixel-centered PSF")
        self.assertGreater(central_4pixel.size, 0, "Empty central region for 4-pixel-centered PSF")

        # The 4-pixel centered version should have better suppression
        max_pixel = cpuArray(central_pixel.max())
        max_4pixel = cpuArray(central_4pixel.max())

        self.assertLess(max_4pixel, max_pixel,
                    f"4-pixel centering should provide better nulling for 4Q coronagraph "
                    f"(max_4pixel={max_4pixel:.2e}, max_pixel={max_pixel:.2e})")

        debug_plot = False
        if debug_plot: # pragma: no cover
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            psf_crop_dim = 20
            
            # Focal plane masks phase
            axes[0,0].imshow(cpuArray(xp.angle(coro_pixel.fp_mask_centered)), cmap='twilight')
            axes[0,0].set_title('FP mask (pixel centered)')
            axes[0,0].axhline(center, color='r', linestyle='--', alpha=0.5)
            axes[0,0].axvline(center, color='r', linestyle='--', alpha=0.5)
            axes[0,0].set_xlim([center-psf_crop_dim/2, center+psf_crop_dim/2])
            axes[0,0].set_ylim([center-psf_crop_dim/2, center+psf_crop_dim/2])

            axes[0,1].imshow(cpuArray(xp.angle(coro_4pixel.fp_mask_centered)), cmap='twilight')
            axes[0,1].set_title('FP mask (4-pixel centered)')
            axes[0,1].axhline(center, color='r', linestyle='--', alpha=0.5)
            axes[0,1].axvline(center, color='r', linestyle='--', alpha=0.5)
            axes[0,1].set_xlim([center-psf_crop_dim/2, center+psf_crop_dim/2])
            axes[0,1].set_ylim([center-psf_crop_dim/2, center+psf_crop_dim/2])

            # Phase shift
            axes[0,2].imshow(cpuArray(xp.angle(coro_4pixel.phase_shift)), cmap='twilight')
            axes[0,2].set_title('Phase shift (0.5 pixel)')

            # PSFs
            vmax = float(psf_pixel.max())
            vmin = vmax * 1e-4
            axes[1,0].imshow(cpuArray(psf_pixel),
                           cmap='viridis',
                           norm=LogNorm(vmin=vmin, vmax=vmax))
            axes[1,0].set_title('PSF (pixel centered)')
            axes[1,0].set_xlim([center-psf_crop_dim/2, center+psf_crop_dim/2])
            axes[1,0].set_ylim([center-psf_crop_dim/2, center+psf_crop_dim/2])

            axes[1,1].imshow(cpuArray(psf_4pixel),
                           cmap='viridis',
                           norm=LogNorm(vmin=vmin, vmax=vmax))
            axes[1,1].set_title('PSF (4-pixel centered)')
            axes[1,1].set_xlim([center-psf_crop_dim/2, center+psf_crop_dim/2])
            axes[1,1].set_ylim([center-psf_crop_dim/2, center+psf_crop_dim/2])

            # Difference
            diff_psf = xp.abs(psf_pixel - psf_4pixel)
            im = axes[1,2].imshow(cpuArray(diff_psf),
                                  cmap='viridis',
                                  norm=LogNorm(vmin=vmin, vmax=vmax))
            axes[1,2].set_title('Difference (log10)')
            axes[1,2].set_xlim([center-psf_crop_dim/2, center+psf_crop_dim/2])
            axes[1,2].set_ylim([center-psf_crop_dim/2, center+psf_crop_dim/2])
            plt.colorbar(im, ax=axes[1,2])

            plt.tight_layout()
            plt.show()

    @cpu_and_gpu
    def test_phase_shift_reversibility(self, target_device_idx, xp):
        """Test that phase shift is properly reversed in the round trip"""

        # Create a coronagraph with 4-pixel centering
        coro = FourQuadrantCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            phase_delay=0.0,  # No mask effect
            target_device_idx=target_device_idx
        )

        # Input field
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        ef_out = self.get_coro_field(coro, ef)

        # With no mask (phase_delay=0), the output should be very similar to input
        # Check phase preservation
        mask = ef.A > 0
        valid_phases_out = ef_out.phaseInNm[mask]

        # Phase should be close to zero (within numerical precision)
        self.assertLess(cpuArray(xp.std(valid_phases_out)), 5.0,
                       "Phase shift should be properly reversed in round trip")

        # Check amplitude preservation (should be close to 1.0)
        valid_amp_out = ef_out.A[mask]
        mean_amp = cpuArray(xp.mean(valid_amp_out))
        self.assertAlmostEqual(mean_amp, 1.0, places=1,
                              msg="Amplitude should be preserved with no mask")
