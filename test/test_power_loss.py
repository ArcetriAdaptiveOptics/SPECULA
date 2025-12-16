import specula
specula.init(0)  # Default target device

import unittest

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.psf import PSF
from specula.processing_objects.power_loss import PowerLoss
from test.specula_testlib import cpu_and_gpu
from specula import np


class TestPowerloss(unittest.TestCase):

    def get_basic_setup(self, target_device_idx, pixel_pupil=20):
        """Create basic setup for Powerloss tests"""
        pixel_pitch = 0.05
        wavelengthInNm = 500.0

        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)

        # Create electric field
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1,
                           target_device_idx=target_device_idx)

        return simul_params, ef, wavelengthInNm

    @cpu_and_gpu
    def test_power_loss_calculation(self, target_device_idx, xp):
        """Test power loss calculation"""
        simul_params, ef, wavelengthInNm = self.get_basic_setup(target_device_idx, pixel_pupil=50)
        receiver_diam = 0.1
        prop_distance = 400e3
        nd = 2

        dx_sat_sq = wavelengthInNm * 1e-9 * prop_distance / (simul_params.pixel_pupil * nd * simul_params.pixel_pitch)

        psf = PSF(simul_params=simul_params, wavelengthInNm=wavelengthInNm,
                  nd=nd, start_time=0.0, target_device_idx=target_device_idx)
        psf.inputs['in_ef'].set(ef)
        psf.setup()

        power_loss = PowerLoss(simul_params=simul_params, wavelengthInNm=wavelengthInNm, nd=nd, prop_distance=prop_distance, receiver_diam=receiver_diam, target_device_idx=target_device_idx)
        power_loss.inputs['se_sr'].set(psf.sr)

        # Generate multiple frames with varying phase
        n_frames = 10
        power_loss_loop = []
        power_loss_manually = []

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
            se_sr = psf.sr
            flux = se_sr.value / dx_sat_sq
            power = flux * receiver_diam
            power_loss_manually.append(10 * np.log10(power))

        psf.finalize()

        # Check power loss
        xp.testing.assert_allclose(xp.array(power_loss_manually), xp.array(power_loss_loop), rtol=1e-5, atol=1e-8,
                                   err_msg="Power loss does not match manual calculation")

