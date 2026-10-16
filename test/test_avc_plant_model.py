import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from specula.data_objects.iir_filter_data import IirFilterData
from specula.lib.avc_plant_model import (
    open_loop_transfer_function,
    closed_loop_sensitivity,
    avc_command_to_residual_tf,
    generate_avc_init_conditions,
)

T = 0.001


def _reference_generate_avc_init_conditions(fAVC, KAO, T, f0, csi, DMdc,
                                            delRTC, nSampDelAVC):
    """Line by line translation of ESO's generateAVCInitConditions.m,
    used as a reference, without the WFS and DAC zero-order holds (see the
    module docstring)."""
    f = np.arange(0.1, 1 / (2 * T) + 1e-9, 0.01)

    w0 = 2 * np.pi * f0
    CTR = KAO / (1 - np.exp(-2j * np.pi * T * f))
    DM = DMdc / ((2j * np.pi * f)**2 / w0**2 + 2 * (2j * np.pi * f) * csi / w0 + 1)
    RTC = np.exp(-2j * np.pi * delRTC * f)
    delAVC = np.exp(-2j * np.pi * (nSampDelAVC * T) * f)
    HOL = RTC * CTR * DM
    Hs = 1.0 / (1 + HOL)
    Pavc = RTC * delAVC * DM * Hs

    idx = np.searchsorted(f, fAVC)
    value = Pavc[idx]
    if abs(value) > 0:
        return value.real, value.imag
    return 1.0, 0.0


class TestAvcPlantModel(unittest.TestCase):

    def test_zero_or_negative_frequency_raises(self):
        with self.assertRaises(ValueError):
            open_loop_transfer_function(0.0, T, loop_gain=0.5, loop_delay=2 * T)
        with self.assertRaises(ValueError):
            closed_loop_sensitivity(-10.0, T, loop_gain=0.5, loop_delay=2 * T)

    def test_zero_loop_gain(self):
        """Open loop: S = 1 and P_avc is the pure delay of the AVC path,
        with the sign of the feedback."""
        f = np.array([5.0, 47.0, 200.0])
        loop_delay, extra_delay = 2 * T, -T
        S = closed_loop_sensitivity(f, T, loop_gain=0.0, loop_delay=loop_delay)
        P = avc_command_to_residual_tf(f, T, loop_gain=0.0, loop_delay=loop_delay,
                                       avc_extra_delay=extra_delay)

        np.testing.assert_allclose(S, np.ones(f.size), rtol=1e-12)
        expected = -np.exp(-2j * np.pi * f * (loop_delay + extra_delay))
        np.testing.assert_allclose(P, expected, rtol=1e-12)

    def test_low_frequency_rejection_of_the_integrator(self):
        """Well below the loop bandwidth |S| ~ 2 pi f T / loop_gain."""
        for loop_gain in (0.2, 0.5):
            for f in (0.5, 1.0):
                S = closed_loop_sensitivity(f, T, loop_gain=loop_gain, loop_delay=2 * T)
                self.assertAlmostEqual(abs(S[0]) / (2 * np.pi * f * T / loop_gain), 1.0, delta=0.05)

    def test_nyquist_frequency(self):
        """At f = 1 / (2 T), z = -1: with a loop delay of 2 steps
        HOL = loop_gain / 2, so S = 2 / (2 + loop_gain), and with one more
        step on the AVC path P_avc = -(-1)^3 * S = S."""
        loop_gain = 0.4
        f = 1 / (2 * T)
        S = closed_loop_sensitivity(f, T, loop_gain=loop_gain, loop_delay=2 * T)
        P = avc_command_to_residual_tf(f, T, loop_gain=loop_gain, loop_delay=2 * T,
                                       avc_extra_delay=-T)

        expected = 2 / (2 + loop_gain)
        np.testing.assert_allclose(S, [expected], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(P, [expected], rtol=1e-9, atol=1e-12)

    def test_dm_resonance_at_natural_frequency(self):
        """At its natural frequency a 2nd-order DM responds with
        dm_dc_gain / (2j * dm_csi)."""
        f0, csi, dc_gain = 300.0, 0.05, 2.0
        hol_ideal = open_loop_transfer_function(f0, T, loop_gain=0.5, loop_delay=2 * T)
        hol_dm = open_loop_transfer_function(f0, T, loop_gain=0.5, loop_delay=2 * T,
                                             dm_dc_gain=dc_gain, dm_f0=f0, dm_csi=csi)

        np.testing.assert_allclose(hol_dm, hol_ideal * dc_gain / (2j * csi), rtol=1e-12)

    def test_low_pass_matches_specula_filter(self):
        """The command low-pass is the frequency response of SPECULA's
        first-order IirFilterData.lpf_from_fc."""
        fs, fc = 1 / T, 80.0
        f = np.array([10.0, 47.0, 80.0, 300.0])
        lpf = IirFilterData.lpf_from_fc(fc, fs, n_ord=1)
        expected = lpf.frequency_response(lpf.num[0], lpf.den[0], fs, freq=f)

        hol = open_loop_transfer_function(f, T, loop_gain=0.5, loop_delay=2 * T)
        hol_lpf = open_loop_transfer_function(f, T, loop_gain=0.5, loop_delay=2 * T, lpf_fc=fc)

        np.testing.assert_allclose(hol_lpf / hol, expected, rtol=1e-5)

    def test_matches_matlab_reference(self):
        # The Matlab source evaluates Pavc on a 0.01 Hz grid, this module
        # exactly at fAVC: they agree only up to the grid quantization.
        # The sign is opposite, see the module docstring.
        for fAVC, KAO, f0, csi, DMdc, delRTC, nSampDelAVC in [
            (47.0, 0.5, 500.0, 0.3, 1.0, 0.002, 0),
            (40.0, 0.5, 500.0, 0.3, 1.0, 0.003, -1),
            (120.0, 0.3, 300.0, 0.1, 1.2, 0.003, 1),
        ]:
            with self.subTest(fAVC=fAVC):
                re_ref, im_ref = _reference_generate_avc_init_conditions(
                    fAVC, KAO, T, f0, csi, DMdc, delRTC, nSampDelAVC)
                p_re, p_im = generate_avc_init_conditions(
                    fAVC, T, loop_gain=KAO, loop_delay=delRTC,
                    avc_extra_delay=nSampDelAVC * T,
                    dm_dc_gain=DMdc, dm_f0=f0, dm_csi=csi)

                self.assertGreater(np.hypot(re_ref, im_ref), 0.1)
                np.testing.assert_allclose([p_re, p_im], [-re_ref, -im_ref], rtol=1e-2, atol=1e-3)

    def test_matches_specula_loop(self):
        """Response of the params_avc_test.yml loop (Integrator delay 1,
        int_gain 0.4) measured in SPECULA by injecting a tone summed to the
        integrator command: the discrete model is exact."""
        for f, gain, phase_deg in [(47.0, 0.7707, 234.3),
                                   (100.0, 1.6403, 182.6),
                                   (200.0, 1.4609, 99.2)]:
            with self.subTest(f=f):
                p = avc_command_to_residual_tf(f, T, loop_gain=0.4, loop_delay=2 * T,
                                               avc_extra_delay=-T)[0]
                self.assertAlmostEqual(abs(p), gain, delta=1e-3)
                self.assertAlmostEqual(np.degrees(np.angle(p)) % 360, phase_deg, delta=0.2)

    def test_matches_specula_scao_loop(self):
        """Response measured in the params_scao_sh_test.yml loop (int_gain
        0.5, delay 2) by injecting a 47 Hz tone on the tip command,
        downstream of the integrator: 0.80 at 255 degrees."""
        p = avc_command_to_residual_tf(47.0, T, loop_gain=0.5, loop_delay=3 * T,
                                       avc_extra_delay=-2 * T)[0]
        self.assertAlmostEqual(abs(p), 0.80, delta=0.02)
        self.assertAlmostEqual(np.degrees(np.angle(p)) % 360, 255.0, delta=10.0)

    def test_init_conditions_are_the_transfer_function(self):
        for f in (20.0, 47.0, 130.0):
            with self.subTest(f=f):
                p_re, p_im = generate_avc_init_conditions(f, T, loop_gain=0.4, loop_delay=2 * T,
                                                          avc_extra_delay=-T, lpf_fc=150.0)
                p = avc_command_to_residual_tf(f, T, loop_gain=0.4, loop_delay=2 * T,
                                               avc_extra_delay=-T, lpf_fc=150.0)[0]
                self.assertEqual(complex(p_re, p_im), p)


if __name__ == '__main__':
    unittest.main()
