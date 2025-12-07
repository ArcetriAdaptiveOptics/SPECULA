import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cpuArray
from specula.lib.calc_psf import calc_psf
from specula.lib.make_mask import make_mask
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.lyot_coronograph import LyotCoronograph

from test.specula_testlib import cpu_and_gpu

class TestLyotCoronograph(unittest.TestCase):

    def setUp(self):
        # Basic simulation parameters
        self.pixel_pupil = 120
        self.pixel_pitch = 0.05
        self.wavelength_nm = 500

        self.simul_params = SimulParams(
            pixel_pupil=self.pixel_pupil,
            # pixel_pitch=self.pixel_pitch
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
    def test_output_shape(self, target_device_idx, xp):
        """Test that output ElectricField has expected shape"""
        lyot = LyotCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=1,
            owaInLambdaOverD=20,
            innerStopAsRatioOfPupil=0.0,
            outerStopAsRatioOfPupil=0.9,
            knife_edge=False,
            target_device_idx=target_device_idx
        )

        # Test wrong knife edge input (both owa not None and knife_edge = True)
        with self.assertRaises(ValueError):
            kedge = LyotCoronograph(
                simul_params=self.simul_params,
                wavelengthInNm=self.wavelength_nm,
                iwaInLambdaOverD=1,
                owaInLambdaOverD=20,
                knife_edge=True,
            )

        kedge = LyotCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=1,
            owaInLambdaOverD=None,
            innerStopAsRatioOfPupil=0.0,
            outerStopAsRatioOfPupil=0.9,
            knife_edge=True,
            target_device_idx=target_device_idx
        )


        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        # test classic Lyot
        out_ef = self.get_coro_field(lyot, ef)
        self.assertEqual(out_ef.A.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(out_ef.phaseInNm.shape, (self.pixel_pupil, self.pixel_pupil))

        # test knife edge
        out_ef = self.get_coro_field(kedge, ef)
        self.assertEqual(out_ef.A.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(out_ef.phaseInNm.shape, (self.pixel_pupil, self.pixel_pupil))

        # psf_in = xp.abs(xp.fft.fftshift(lyot.propagate_to_focal_plane(lyot.ef_in)))**2
        # psf_out = xp.abs(xp.fft.fftshift(lyot.propagate_to_focal_plane(lyot.ef_out)))**2

        # psf_out /= xp.max(psf_in)
        # psf_in/= xp.max(psf_in)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(cpuArray(lyot.fp_mask), cmap='gray')
        # plt.colorbar()
        # plt.title('Focal plane mask')
        # plt.subplot(1,2,2)
        # plt.imshow(cpuArray(lyot.pp_mask), cmap='gray')
        # plt.colorbar()
        # plt.title('Pupil plane mask')
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(cpuArray(xp.log(psf_in)), cmap='twilight', vmax=0, vmin=-24)
        # plt.colorbar()
        # plt.title('Input PSF')
        # plt.xlim([lyot.fft_totsize//2-50,lyot.fft_totsize//2+50])
        # plt.ylim([lyot.fft_totsize//2-50,lyot.fft_totsize//2+50])
        # plt.subplot(1,2,2)
        # plt.imshow(cpuArray(xp.log(psf_out)), cmap='twilight', vmax=0, vmin=-24)
        # plt.colorbar()
        # plt.title('Output PSF')
        # plt.xlim([lyot.fft_totsize//2-50,lyot.fft_totsize//2+50])
        # plt.ylim([lyot.fft_totsize//2-50,lyot.fft_totsize//2+50])
        # plt.show()

    @cpu_and_gpu
    def test_psf_with_and_without_obstruction(self, target_device_idx, xp):
        """Test PSF with and without a central obstruction using calc_psf"""
        # No filter (no obstruction)
        lyot_nostop = LyotCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        ef_nostop = self.get_coro_field(lyot_nostop, ef)

        # With filter: central obstruction of 2 lambda/D
        lyot_stop = LyotCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=2,
            target_device_idx=target_device_idx
        )
        ef_stop = self.get_coro_field(lyot_stop, ef)

        # Compute PSF for both cases using calc_psf
        psf = calc_psf(ef.phaseInNm, ef.A, xp=xp, complex_dtype=xp.complex64, normalize=True)
        psf_nofilter = calc_psf(ef_nostop.phaseInNm, ef_nostop.A, xp=xp, complex_dtype=xp.complex64, normalize=True)
        psf_obs = calc_psf(ef_stop.phaseInNm, ef_stop.A, xp=xp, complex_dtype=xp.complex64, normalize=True)

        max_psf = float(psf.max())
        max_psf_nofilter = float(psf_nofilter.max())
        max_psf_obs = float(psf_obs.max())

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cpuArray(ef_nostop.A), cmap='gray')
            plt.colorbar()
            plt.title('Amplitude without obstruction')
            plt.subplot(1,2,2)
            plt.imshow(cpuArray(ef_stop.A), cmap='gray')
            plt.colorbar()
            plt.title('Amplitude with 2 lambda/D obstruction')
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cpuArray(ef_nostop.phaseInNm), cmap='twilight')
            plt.colorbar()
            plt.title('Phase without obstruction')
            plt.subplot(1,2,2)
            plt.imshow(cpuArray(ef_stop.phaseInNm), cmap='twilight')
            plt.colorbar()
            plt.title('Phase with 2 lambda/D obstruction')
            plt.show()
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(cpuArray(psf), cmap='viridis', norm=colors.LogNorm(vmin=1e-6*max_psf, vmax=max_psf))
            plt.colorbar()
            plt.title('PSF input wavefront')
            plt.subplot(1,3,2)
            plt.imshow(cpuArray(psf_nofilter), cmap='viridis', norm=colors.LogNorm(vmin=1e-6*max_psf_nofilter, vmax=max_psf_nofilter))
            plt.colorbar()
            plt.title('PSF without obstruction')
            plt.subplot(1,3,3)
            plt.imshow(cpuArray(psf_obs), cmap='viridis', norm=colors.LogNorm(vmin=1e-6*max_psf_obs, vmax=max_psf_obs))
            plt.colorbar()
            plt.title('PSF with 2 lambda/D obstruction')
            plt.show()

        # Check shapes
        self.assertEqual(psf_nofilter.shape, psf_obs.shape)

        # Check that the mask has an effect (PSFs must differ)
        diff = np.abs(psf_nofilter - psf_obs).sum()
        self.assertGreater(cpuArray(diff), 0.0, "Obstruction mask does not affect the PSF!")

    @cpu_and_gpu
    def test_phase_preservation(self, target_device_idx, xp):
        """Test that a flat input phase results in a flat output phase (no mask)"""
        lyot_nostop = LyotCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        out_ef = self.get_coro_field(lyot_nostop,ef)

        # Output phase should be (almost) constant for a flat input
        mask = ef.A > 0
        valid_phases = out_ef.phaseInNm[mask]
        min_phase = float(xp.min(valid_phases))
        max_phase = float(xp.max(valid_phases))

        # max and min phase should be close to zero (within 5 nm)
        self.assertLess(np.abs(max_phase), 5)
        self.assertLess(np.abs(min_phase), 5)

    @cpu_and_gpu
    def test_amplitude_preservation(self, target_device_idx, xp):
        """Test that a flat input amplitude results in a nonzero output amplitude (no mask)"""
        lyot_nostop = LyotCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = 1
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        out_ef = self.get_coro_field(lyot_nostop,ef)

        # Output amplitude should not be all zeros and should be approximately the same as the input one
        self.assertGreater(float(out_ef.A.sum()), 0.0)
        self.assertLess(float(out_ef.A.max()), 2.0*float(ef.A.max()))

    @cpu_and_gpu
    def test_s0_scaling_with_obstruction(self, target_device_idx, xp):
        """Test that S0 is scaled correctly when using obstruction"""
        # Test with obstruction - S0 should decrease
        lyot_stop = LyotCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=2,
            target_device_idx=target_device_idx
        )

        # Test without obstruction - S0 should remain similar
        lyot_nostop = LyotCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=0,
            target_device_idx=target_device_idx
        )

        # Create input electric field
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=100.0, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.S0 = 100.0
        ef.generation_time = 1

        # Test with obstruction
        ef_stop = self.get_coro_field(lyot_stop, ef)
        s0_with_obs = ef_stop.S0

        # Test without obstruction
        ef_nostop = self.get_coro_field(lyot_nostop, ef)
        s0_no_obs = ef_nostop.S0

        # S0 with obstruction should be less than without obstruction
        self.assertLess(s0_with_obs, s0_no_obs, "S0 should decrease with obstruction!")

        # Both should be less than or equal to original S0
        self.assertLessEqual(s0_with_obs, 100.0)
        self.assertLessEqual(s0_no_obs, 100.0)
