import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cpuArray
from specula.lib.calc_psf import calc_psf
from specula.lib.make_mask import make_mask
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.vortex_coronograph import VortexCoronograph

from test.specula_testlib import cpu_and_gpu

class TestVortexCoronograph(unittest.TestCase):

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
            _ = VortexCoronograph(
                simul_params=self.simul_params,
                wavelengthInNm=self.wavelength_nm,
                vortexCharge=6.0,
                innerStopAsRatioOfPupil=0.9,
                outerStopAsRatioOfPupil=0.1,
            )
        # inner vortex boolean = false but inner vortex parameter is still passed
        with self.assertRaises(ValueError):
            _ = VortexCoronograph(
                simul_params=self.simul_params,
                wavelengthInNm=self.wavelength_nm,
                vortexCharge=6.0,
                inVortexShift=0.2,
                addInVortex=False,
                innerStopAsRatioOfPupil=0.9,
                outerStopAsRatioOfPupil=0.1,
            )
    
    @cpu_and_gpu
    def test_mask_shape(self, target_device_idx, xp):
        """Test that coronograph masks have the expected shape"""
        coro = VortexCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            vortexCharge=6.0,
            addInVortex=True,
            innerStopAsRatioOfPupil=0.0,
            outerStopAsRatioOfPupil=0.9,
            target_device_idx=target_device_idx
        )

        self.assertEqual(coro.pupil_mask.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(coro.fp_mask.shape, (coro.fft_totsize, coro.fft_totsize))
        self.assertEqual(coro.apodizer, 1.0) # no apodizer

        debug_plot = True
        if debug_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cpuArray(xp.angle(coro.fp_mask)), cmap='RdBu')
            plt.colorbar()
            plt.title('Focal plane mask')
            plt.subplot(1,2,2)
            plt.imshow(cpuArray(coro.pupil_mask), cmap='gray')
            plt.colorbar()
            plt.title('Pupil plane mask')

    @cpu_and_gpu
    def test_output_shape(self, target_device_idx, xp):
        """Test that output ElectricField has expected shape"""
        coro = VortexCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            vortexCharge=6.0,
            addInVortex=True,
            innerStopAsRatioOfPupil=0.0,
            outerStopAsRatioOfPupil=0.9,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        out_ef = self.get_coro_field(coro, ef)
        self.assertEqual(out_ef.A.shape, (self.pixel_pupil, self.pixel_pupil))
        self.assertEqual(out_ef.phaseInNm.shape, (self.pixel_pupil, self.pixel_pupil))


    @cpu_and_gpu
    def test_psf_with_and_without_coronograph(self, target_device_idx, xp):
        """Test PSF with and without a coronograph using calc_psf"""
        # No filter (no obstruction)
        nodelay_coro = VortexCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            vortexCharge=0.0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A[:] = xp.array(self.mask)
        ef.phaseInNm[:] = 0.0
        ef.generation_time = 1

        ef_nocoro = self.get_coro_field(nodelay_coro, ef)

        # With filter: central obstruction of 2 lambda/D
        coro = VortexCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            vortexCharge=6.0,
            addInVortex=True,
            target_device_idx=target_device_idx
        )
        ef_coro = self.get_coro_field(coro, ef)

        # Compute PSF for both cases using calc_psf
        nm2rad = 2*xp.pi/self.wavelength_nm
        psf_nocoro = calc_psf(ef_nocoro.phaseInNm*nm2rad, ef_nocoro.A, xp=xp, complex_dtype=xp.complex64, normalize=True)
        psf_coro = calc_psf(ef_coro.phaseInNm*nm2rad, ef_coro.A, xp=xp, complex_dtype=xp.complex64, normalize=True)

        # Check shapes
        self.assertEqual(psf_nocoro.shape, psf_coro.shape)

        # Check that the coronograph has an effect (PSFs should be smaller)
        diff = np.abs(psf_nocoro - psf_coro).sum()
        self.assertGreater(cpuArray(diff), 0.0, "Coronograph does not affect the PSF!")

        debug_plot = True
        if debug_plot:
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
         and a nonzero output amplitude for the no coronograph case"""
        nocoro = VortexCoronograph(
            simul_params=self.simul_params,
            wavelengthInNm=self.wavelength_nm,
            vortexCharge=0.0,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1, target_device_idx=target_device_idx)
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


    # @cpu_and_gpu
    # def test_s0_scaling_with_coronograph(self, target_device_idx, xp):
    #     """Test that S0 is scaled correctly when using the coronograph"""
    #     # Test with coronograph - S0 should decrease
    #     coro = FourQuadrantCoronograph(
    #         simul_params=self.simul_params,
    #         wavelengthInNm=self.wavelength_nm,
    #         outerStopAsRatioOfPupil=0.95,
    #         innerStopAsRatioOfPupil=0.02,
    #         target_device_idx=target_device_idx
    #     )

    #     # Test without coronograph - S0 should remain similar
    #     nodelay_coro = FourQuadrantCoronograph(
    #         simul_params=self.simul_params,
    #         wavelengthInNm=self.wavelength_nm,
    #         phase_delay=0.0,
    #         target_device_idx=target_device_idx
    #     )

    #     # Create input electric field
    #     ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=100.0, target_device_idx=target_device_idx)
    #     ef.A[:] = xp.array(self.mask)
    #     ef.phaseInNm[:] = 0.0
    #     ef.S0 = 100.0
    #     ef.generation_time = 1

    #     # Test with obstruction
    #     ef_coro = self.get_coro_field(coro, ef)
    #     s0_with_coro = ef_coro.S0

    #     # Test without obstruction
    #     ef_nocoro = self.get_coro_field(nodelay_coro, ef)
    #     s0_no_coro = ef_nocoro.S0

    #     # S0 with obstruction should be less than without obstruction
    #     self.assertLess(s0_with_coro, s0_no_coro, "S0 should decrease with obstruction!")

    #     # Both should be less than or equal to original S0
    #     self.assertLessEqual(s0_with_coro, 100.0)
    #     self.assertLessEqual(s0_no_coro, 100.0)
