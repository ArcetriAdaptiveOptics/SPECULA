import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cpuArray
from specula.lib.calc_psf import calc_psf
from specula.lib.make_mask import make_mask
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.apodizer_coronagraph import APPCoronagraph

from test.specula_testlib import cpu_and_gpu

class TestAPPCoronagraph(unittest.TestCase):

    def setUp(self):
        # Basic simulation parameters
        self.pixel_pupil = 40
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

    @cpu_and_gpu
    def test_mask_shape(self, target_device_idx, xp):
        """Test that coronagraph masks have the expected shape"""
        coro = APPCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            pupil=self.mask.copy(),
            contrastInDarkHole=1e-4,
            iwaInLambdaOverD=4.0,
            owaInLambdaOverD=12.0,
            target_device_idx=target_device_idx
        )

        self.assertEqual(coro.pupil_mask.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(coro.fp_mask.shape, (coro.fft_totsize,coro.fft_totsize)) 
        self.assertEqual(coro.apodizer.shape, (self.pixel_pupil, self.pixel_pupil)) # apodizer

        debug_plot = False
        if debug_plot: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cpuArray(xp.angle(coro.apodizer)), cmap='RdBu')
            plt.colorbar()
            plt.title('Focal plane mask')
            plt.subplot(1,2,2)
            plt.imshow(cpuArray(coro.pupil_mask), cmap='gray')
            plt.colorbar()
            plt.title('Pupil plane mask')

    @cpu_and_gpu
    def test_output_shape(self, target_device_idx, xp):
        """Test that output ElectricField has expected shape"""
        coro = APPCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            pupil=self.mask,
            contrastInDarkHole=1e-4,
            iwaInLambdaOverD=4.0,
            owaInLambdaOverD=12.0,
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

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        # Coronagraph
        coro = APPCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            pupil=self.mask,
            contrastInDarkHole=1e-4,
            iwaInLambdaOverD=4.0,
            owaInLambdaOverD=12.0,
            target_device_idx=target_device_idx
        )
        ef_coro = self.get_coro_field(coro, ef)

        # Compute PSF for both cases using calc_psf
        nm2rad = 2*xp.pi/self.wavelength_nm
        psf_nocoro = calc_psf(ef.phaseInNm*nm2rad, ef.A,
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


    # @cpu_and_gpu
    # def test_s0_scaling_with_coronagraph(self, target_device_idx, xp):
    #     """Test that S0 is scaled correctly when using the coronagraph"""
    #     # Test with coronagraph - S0 should decrease
    #     coro = APPCoronagraph(
    #         simul_params=self.simul_params,
    #         wavelengthInNm=self.wavelength_nm,
    #         pupil=self.mask,
    #         contrastInDarkHole=1e-5,
    #         iwaInLambdaOverD=4.0,
    #         owaInLambdaOverD=12.0,
    #         target_device_idx=target_device_idx
    #     )

    #     # Create input electric field
    #     ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
    #                        S0=100.0, target_device_idx=target_device_idx)
    #     ef.A[:] = xp.array(self.mask)
    #     ef.phaseInNm[:] = 0.0
    #     ef.S0 = 100.0
    #     ef.generation_time = 1

    #     # Test with obstruction
    #     ef_coro = self.get_coro_field(coro, ef)
    #     s0_with_coro = ef_coro.S0

    #     # S0 with coronagraph should be less than SO without coronagraph
    #     self.assertLess(s0_with_coro, ef.S0, "S0 should decrease with coronagraph!")
