import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cpuArray
from specula.lib.calc_psf import calc_psf
from specula.lib.make_mask import make_mask
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.lyot_coronagraph import LyotCoronagraph

from test.specula_testlib import cpu_and_gpu

class TestLyotCoronagraph(unittest.TestCase):

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
            _ = LyotCoronagraph(
                simul_params=self.simul_params,
                wavelengthInNm=self.wavelength_nm,
                iwaInLambdaOverD=1,
                owaInLambdaOverD=20,
                innerStopAsRatioOfPupil=0.9,
                outerStopAsRatioOfPupil=0.1,
            )
        # Test wrong knife edge input (both owa not None and knife_edge = True)
        with self.assertRaises(ValueError):
            _ = LyotCoronagraph(
                simul_params=self.simul_params,
                wavelengthInNm=self.wavelength_nm,
                iwaInLambdaOverD=1,
                owaInLambdaOverD=20,
                knife_edge=True,
            )

    @cpu_and_gpu
    def test_mask_shape(self, target_device_idx, xp):
        """Test that coronagraph masks have the expected shape"""
        lyot = LyotCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=1,
            owaInLambdaOverD=20,
            innerStopAsRatioOfPupil=0.0,
            outerStopAsRatioOfPupil=0.9,
            knife_edge=False,
            target_device_idx=target_device_idx
        )

        kedge = LyotCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=1,
            owaInLambdaOverD=None,
            innerStopAsRatioOfPupil=0.0,
            outerStopAsRatioOfPupil=0.9,
            knife_edge=True,
            target_device_idx=target_device_idx
        )

        self.assertEqual(lyot.pupil_mask.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(lyot.apodizer, 1.0) # no apodizer

        # test knife edge
        self.assertEqual(kedge.pupil_mask.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(kedge.apodizer, 1.0) # no apodizer

        debug_plot = False
        if debug_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cpuArray(lyot.fp_mask), cmap='gray')
            plt.colorbar()
            plt.title('Focal plane mask')
            plt.subplot(1,2,2)
            plt.imshow(cpuArray(lyot.pupil_mask), cmap='gray')
            plt.colorbar()
            plt.title('Pupil plane mask')

    @cpu_and_gpu
    def test_output_shape(self, target_device_idx, xp):
        """Test that output ElectricField has expected shape"""
        lyot = LyotCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=1,
            owaInLambdaOverD=20,
            innerStopAsRatioOfPupil=0.0,
            outerStopAsRatioOfPupil=0.9,
            knife_edge=False,
            target_device_idx=target_device_idx
        )

        kedge = LyotCoronagraph(
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
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
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


    @cpu_and_gpu
    def test_psf_with_and_without_coronagraph(self, target_device_idx, xp):
        """Test PSF with and without a coronagraph using calc_psf"""
        # No filter (no coronagraph)
        lyot_nocoro = LyotCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=0.0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        ef_nocoro = self.get_coro_field(lyot_nocoro, ef)

        # With coronagraph: central 2 lambda/D obstruction
        lyot_coro = LyotCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=2,
            outerStopAsRatioOfPupil=0.95,
            target_device_idx=target_device_idx
        )
        ef_coro = self.get_coro_field(lyot_coro, ef)

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
        if debug_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cpuArray(xp.log(psf_nocoro)), cmap='twilight', vmax=0, vmin=-24)
            plt.colorbar()
            plt.title('Input PSF')
            plt.xlim([lyot_coro.fft_totsize//2-50,lyot_coro.fft_totsize//2+50])
            plt.ylim([lyot_coro.fft_totsize//2-50,lyot_coro.fft_totsize//2+50])
            plt.subplot(1,2,2)
            plt.imshow(cpuArray(xp.log(psf_coro)), cmap='twilight', vmax=0, vmin=-24)
            plt.colorbar()
            plt.title('Output PSF')
            plt.xlim([lyot_coro.fft_totsize//2-50,lyot_coro.fft_totsize//2+50])
            plt.ylim([lyot_coro.fft_totsize//2-50,lyot_coro.fft_totsize//2+50])
            plt.show()

    @cpu_and_gpu
    def test_phase_and_amplitude_preservation(self, target_device_idx, xp):
        """Test that a flat input phase results in a flat output phase
         and a nonzero output amplitude for the no coronagraph case"""
        lyot_nocoro = LyotCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=0.0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil,
                           self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        out_ef = self.get_coro_field(lyot_nocoro,ef)

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
    def test_s0_scaling_with_coronagraph(self, target_device_idx, xp):
        """Test that S0 is scaled correctly when using the coronagraph"""
        # Test with coronagraph - S0 should decrease
        lyot_coro = LyotCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=2,
            target_device_idx=target_device_idx
        )

        # Test without coronagraph - S0 should remain similar
        lyot_nocoro = LyotCoronagraph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            iwaInLambdaOverD=0.0,
            target_device_idx=target_device_idx
        )

        # Create input electric field
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=100.0, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.S0 = 100.0
        ef.generation_time = 1

        # Test with corograph
        ef_coro = self.get_coro_field(lyot_coro, ef)
        s0_with_coro = ef_coro.S0

        # Test without coronagraph
        ef_nocoro = self.get_coro_field(lyot_nocoro, ef)
        s0_no_coro = ef_nocoro.S0

        # S0 with coronagraph should be less than without coronagraph
        self.assertLess(s0_with_coro, s0_no_coro, "S0 should decrease with coronagraph!")

        # Both should be less than or equal to original S0
        self.assertLessEqual(s0_with_coro, 100.0)
        self.assertLessEqual(s0_no_coro, 100.0)