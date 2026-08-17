import specula
specula.init(0)  # Default target device

import unittest

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.psf import PSF
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.source import Source
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.layer import Layer
from specula.processing_objects.power_loss import PowerLoss
from test.specula_testlib import cpu_and_gpu

from specula import np


class TestPowerloss(unittest.TestCase):

    def get_basic_setup(self, target_device_idx, xp, doFresnel=False):
        """Create basic setup for Powerloss tests"""
        wavelength = 500
        pixel_pupil = 120
        pixel_pitch = 1/120.
        height = 500

        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)

        pupilstop = Pupilstop(simul_params=simul_params, mask_diam=1, target_device_idx=target_device_idx)
        layer = Layer(dimx=pupilstop.A.shape[0], dimy=pupilstop.A.shape[1], pixel_pitch=simul_params.pixel_pitch,
                      height=0.0, target_device_idx=target_device_idx)

        on_axis_source = Source(polar_coordinates=[0, 0], magnitude=0.0, height=height, wavelengthInNm=wavelength,
                                target_device_idx=target_device_idx)

        prop = AtmoPropagation(simul_params=simul_params, source_dict={'on_axis': on_axis_source}, doFresnel=doFresnel,
                               upwards=True, wavelengthInNm=wavelength, padding_factor=2,
                               target_device_idx=target_device_idx)
        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])

        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.A = pupilstop.A

        psf = PSF(simul_params=simul_params, wavelengthInNm=wavelength, nd=2, start_time=0.0,
                  target_device_idx=target_device_idx)
        psf.inputs['in_ef'].set(ef)

        power_loss = PowerLoss(simul_params=simul_params, prop=prop, target_device_idx=target_device_idx)
        power_loss.inputs['in_ef'].set(ef)

        prop.setup()
        psf.setup()

        # Generate multiple frames with varying phase
        n_frames = 10
        power_loss_loop = []
        power_loss_psf = []

        for t in range(1, n_frames + 1):
            # Add random phase variations
            ef.phaseInNm[:] = 50.0 * xp.random.randn(*ef.phaseInNm.shape)
            ef.A[:] = 1.0
            ef.generation_time = t

            psf.check_ready(t)
            psf.trigger()
            psf.post_trigger()

            # Store snapshot for manual calculation
            power_loss.check_ready(t)
            power_loss.trigger()
            power_loss.post_trigger()
            power_loss_loop.append(power_loss.power_loss.value.copy())

            # Calculate power loss manually
            power_loss_psf.append(10 * np.log10(psf.sr.value.copy()))

        psf.finalize()

        return power_loss_psf, power_loss_loop

    @cpu_and_gpu
    def test_power_loss_calculation_Fresnel(self, target_device_idx, xp):
        """Test power loss calculation with Fresnel propagation"""
        power_loss_psf, power_loss_loop = self.get_basic_setup(target_device_idx, xp, doFresnel=True)

        # Check power loss computation with Fresnel propagation gives better Strehl ratio
        xp.testing.assert_array_equal(xp.array(power_loss_psf) < xp.array(power_loss_loop), True)

    @cpu_and_gpu
    def test_power_loss_calculation(self, target_device_idx, xp):
        """Test power loss calculation with geometrical propagation"""
        power_loss_psf, power_loss_loop = self.get_basic_setup(target_device_idx, xp, doFresnel=False)

        # Check power loss with geometrical propagation is the same as from psf
        xp.testing.assert_allclose(xp.array(power_loss_psf), xp.array(power_loss_loop), rtol=1e-5, atol=1e-8,
                                   err_msg="Power loss does not match manual calculation")

