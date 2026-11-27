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

from specula import np, cpuArray


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
    def test_psf_dtype_follows_precision(self, target_device_idx, xp):
        """out_psf must be allocated in the object's own dtype, not always float64."""
        wavelength = 500
        pixel_pupil = 16
        pixel_pitch = 1 / 16.

        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)
        on_axis_source = Source(polar_coordinates=[0, 0], magnitude=0.0, height=500,
                                wavelengthInNm=wavelength, target_device_idx=target_device_idx)
        prop = AtmoPropagation(simul_params=simul_params, source_dict={'on_axis': on_axis_source},
                               doFresnel=False, upwards=True, wavelengthInNm=wavelength,
                               padding_factor=2, target_device_idx=target_device_idx)

        power_loss32 = PowerLoss(simul_params=simul_params, prop=prop, precision=1,
                                 target_device_idx=target_device_idx)
        self.assertEqual(power_loss32.psf.value.dtype, xp.float32)

        power_loss64 = PowerLoss(simul_params=simul_params, prop=prop, precision=0,
                                 target_device_idx=target_device_idx)
        self.assertEqual(power_loss64.psf.value.dtype, xp.float64)

    @cpu_and_gpu
    def test_power_loss_calculation(self, target_device_idx, xp):
        """Test power loss calculation with geometrical propagation"""
        power_loss_psf, power_loss_loop = self.get_basic_setup(target_device_idx, xp, doFresnel=False)

        # Check power loss with geometrical propagation is the same as from psf
        xp.testing.assert_allclose(xp.array(power_loss_psf), xp.array(power_loss_loop), rtol=1e-5, atol=1e-8,
                                   err_msg="Power loss does not match manual calculation")

    @cpu_and_gpu
    def test_outputs_written_in_place(self, target_device_idx, xp):
        """The sr and power_loss arrays are the same at each trigger, so that an
        accelerated reader keeps a valid graph"""
        wavelength = 500
        pixel_pupil = 16
        pixel_pitch = 1 / 16.
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)
        layer = Layer(dimx=pixel_pupil, dimy=pixel_pupil, pixel_pitch=pixel_pitch,
                      height=0.0, target_device_idx=target_device_idx)
        source = Source(polar_coordinates=[0, 0], magnitude=0.0, height=500,
                        wavelengthInNm=wavelength, target_device_idx=target_device_idx)
        prop = AtmoPropagation(simul_params=simul_params, source_dict={'on_axis': source},
                               doFresnel=False, upwards=True, wavelengthInNm=wavelength,
                               padding_factor=2, target_device_idx=target_device_idx)
        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])
        prop.setup()
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1,
                           target_device_idx=target_device_idx)
        power_loss = PowerLoss(simul_params=simul_params, prop=prop, target_device_idx=target_device_idx)
        power_loss.inputs['in_ef'].set(ef)

        first = None
        srs = []
        for t in range(1, 4):
            # Tilt growing with t: the SR decreases
            ef.phaseInNm[:] = xp.arange(pixel_pupil)[:, None] * 10.0 * (t - 1)
            ef.A[:] = 1.0
            ef.generation_time = t
            power_loss.check_ready(t)
            power_loss.trigger()
            power_loss.post_trigger()
            values = (power_loss.sr.value, power_loss.power_loss.value)
            if first is None:
                first = values
            self.assertIs(values[0], first[0])
            self.assertIs(values[1], first[1])
            srs.append(float(first[0]))
            self.assertAlmostEqual(float(first[1]), 10 * np.log10(srs[-1]), places=4)
        self.assertAlmostEqual(srs[0], 1.0, places=5)
        self.assertLess(srs[2], srs[1])
        self.assertLess(srs[1], 1.0)

    # --- received power: geometric (far field of the pupil field) vs Fresnel (field at the receiver) ----

    def _upwards_prop(self, target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm, height,
                      doFresnel, phase=None):
        """Upwards propagation from the pupil (annular pupil stop + one phase layer at 0 m) to a
        source at height [m]. Returns the propagation object and the pupil stop."""
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)
        pupilstop = Pupilstop(simul_params=simul_params, mask_diam=1.0, obs_diam=0.33,
                              target_device_idx=target_device_idx, precision=0)
        layer = Layer(dimx=pixel_pupil, dimy=pixel_pupil, pixel_pitch=pixel_pitch, height=0.0,
                      target_device_idx=target_device_idx, precision=0)
        if phase is not None:
            layer.phaseInNm[:] = xp.asarray(phase)
        source = Source(polar_coordinates=[0, 0], magnitude=0.0, height=height, wavelengthInNm=wavelengthInNm,
                        target_device_idx=target_device_idx)
        prop = AtmoPropagation(simul_params=simul_params, source_dict={'src': source}, doFresnel=doFresnel,
                               upwards=True, wavelengthInNm=wavelengthInNm, padding_factor=2, precision=0,
                               target_device_idx=target_device_idx)
        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([pupilstop, layer])
        prop.setup()
        for obj in (pupilstop, layer):
            obj.generation_time = 1
        return prop, pupilstop

    @staticmethod
    def _propagate(prop):
        prop.check_ready(1)
        prop.trigger()
        prop.post_trigger()
        return prop.outputs['out_src_ef']

    @staticmethod
    def _power_loss_sr(prop, ef, target_device_idx, **kwargs):
        """Strehl ratio of PowerLoss for the field ef after the propagation prop;
        returns (sr, power_loss object)"""
        simul_params = SimulParams(pixel_pupil=prop.pixel_pupil, pixel_pitch=prop.pixel_pitch)
        power_loss = PowerLoss(simul_params=simul_params, prop=prop, precision=0,
                               target_device_idx=target_device_idx, **kwargs)
        power_loss.inputs['in_ef'].set(ef)
        ef.generation_time = 1
        power_loss.check_ready(1)
        power_loss.trigger()
        power_loss.post_trigger()
        return float(cpuArray(power_loss.sr.value)), power_loss

    @staticmethod
    def _tilt_and_noise(pixel_pupil, wavelengthInNm, seed):
        """Phase [nm]: tilt of 0.3 lambda/D plus 40 nm rms white noise"""
        rng = np.random.default_rng(seed)
        y, x = (np.mgrid[0:pixel_pupil, 0:pixel_pupil] + 0.5) / pixel_pupil - 0.5
        return 0.3 * wavelengthInNm * x + 40.0 * rng.standard_normal((pixel_pupil, pixel_pupil))

    @staticmethod
    def _fresnel_intensity(field, pixel_pitch, wavelengthInNm, z, positions):
        """Independent Fresnel integral of a pupil field: intensity at receiver positions (x, y) [m]"""
        n = field.shape[0]
        k = 2 * np.pi / (wavelengthInNm * 1e-9)
        c = (np.arange(n) - n / 2 + 0.5) * pixel_pitch
        X, Y = np.meshgrid(c, c, indexing='ij')
        return np.array([abs(np.sum(field * np.exp(1j * k / (2 * z) * ((X - u) ** 2 + (Y - v) ** 2)))) ** 2
                         for (u, v) in positions])

    def _ef_from_pupil(self, target_device_idx, xp, pupilstop, phase, pixel_pupil, pixel_pitch):
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx,
                           precision=0)
        ef.A[:] = pupilstop.A
        ef.phaseInNm[:] = xp.asarray(phase)
        return ef

    @cpu_and_gpu
    def test_power_loss_geometric_is_psf_ratio(self, target_device_idx, xp):
        """Geometric: Strehl = on-axis far field of the pupil field / diffraction-limited value"""
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1075.0
        phase = self._tilt_and_noise(pixel_pupil, wavelengthInNm, 0)

        prop, pupilstop = self._upwards_prop(target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm,
                                             3.6e7, doFresnel=False)
        ef = self._ef_from_pupil(target_device_idx, xp, pupilstop, phase, pixel_pupil, pixel_pitch)
        sr, _ = self._power_loss_sr(prop, ef, target_device_idx)

        A = cpuArray(pupilstop.A)
        field = A * np.exp(2j * np.pi * phase / wavelengthInNm)
        np.testing.assert_allclose(sr, abs(np.sum(field)) ** 2 / abs(np.sum(A)) ** 2, rtol=1e-6)

    @cpu_and_gpu
    def test_power_loss_fresnel_far_field_equals_geometric(self, target_device_idx, xp):
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1075.0
        height = 3.6e7
        phase = self._tilt_and_noise(pixel_pupil, wavelengthInNm, 0)

        prop, pupilstop = self._upwards_prop(target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm,
                                             height, doFresnel=False)
        ef = self._ef_from_pupil(target_device_idx, xp, pupilstop, phase, pixel_pupil, pixel_pitch)
        sr_geo, _ = self._power_loss_sr(prop, ef, target_device_idx)

        prop, _ = self._upwards_prop(target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm,
                                     height, doFresnel=True, phase=phase)
        sr_fresnel, _ = self._power_loss_sr(prop, self._propagate(prop), target_device_idx)
        np.testing.assert_allclose(sr_fresnel, sr_geo, rtol=2e-3)

    @cpu_and_gpu
    def test_power_loss_fresnel_near_field(self, target_device_idx, xp):
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1075.0
        height = 300e3
        phase = self._tilt_and_noise(pixel_pupil, wavelengthInNm, 1)

        prop, pupilstop = self._upwards_prop(target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm,
                                             height, doFresnel=True, phase=phase)
        sr_fresnel, _ = self._power_loss_sr(prop, self._propagate(prop), target_device_idx)

        h = pixel_pitch / 2
        centre = [(-h, -h), (-h, h), (h, -h), (h, h)]               # the 4 central pixels (point receiver)
        A = cpuArray(pupilstop.A)
        field = A * np.exp(2j * np.pi * phase / wavelengthInNm)
        expected = self._fresnel_intensity(field, pixel_pitch, wavelengthInNm, height, centre).sum() / \
            self._fresnel_intensity(A.astype(complex), pixel_pitch, wavelengthInNm, height, centre).sum()
        np.testing.assert_allclose(sr_fresnel, expected, rtol=1e-6)

        prop, pupilstop = self._upwards_prop(target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm,
                                             height, doFresnel=False)
        ef = self._ef_from_pupil(target_device_idx, xp, pupilstop, phase, pixel_pupil, pixel_pitch)
        sr_geo, _ = self._power_loss_sr(prop, ef, target_device_idx)
        np.testing.assert_allclose(sr_geo, sr_fresnel, rtol=0.02)

    @cpu_and_gpu
    def test_power_loss_fresnel_near_field_focus(self, target_device_idx, xp):
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1075.0
        height = 300e3
        c = (np.arange(pixel_pupil) - pixel_pupil / 2 + 0.5) * pixel_pitch
        phase = -(c[:, None] ** 2 + c[None, :] ** 2) / (2 * height) * 1e9     # [nm] focus at height

        prop, pupilstop = self._upwards_prop(target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm,
                                             height, doFresnel=True, phase=phase)
        sr_fresnel, _ = self._power_loss_sr(prop, self._propagate(prop), target_device_idx)
        h = pixel_pitch / 2
        centre = [(-h, -h), (-h, h), (h, -h), (h, h)]
        A = cpuArray(pupilstop.A)
        field = A * np.exp(2j * np.pi * phase / wavelengthInNm)
        expected = self._fresnel_intensity(field, pixel_pitch, wavelengthInNm, height, centre).sum() / \
            self._fresnel_intensity(A.astype(complex), pixel_pitch, wavelengthInNm, height, centre).sum()
        np.testing.assert_allclose(sr_fresnel, expected, rtol=1e-6)
        self.assertGreater(sr_fresnel, 1.3)

        prop, pupilstop = self._upwards_prop(target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm,
                                             height, doFresnel=False)
        ef = self._ef_from_pupil(target_device_idx, xp, pupilstop, phase, pixel_pupil, pixel_pitch)
        sr_geo, _ = self._power_loss_sr(prop, ef, target_device_idx)
        self.assertLess(sr_geo, 1.0)

    @cpu_and_gpu
    def test_power_loss_receiver_diameter(self, target_device_idx, xp):
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1075.0
        height = 300e3
        receiver_diameter = 0.3
        phase = self._tilt_and_noise(pixel_pupil, wavelengthInNm, 2)

        prop, pupilstop = self._upwards_prop(target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm,
                                             height, doFresnel=True, phase=phase)
        sr, power_loss = self._power_loss_sr(prop, self._propagate(prop), target_device_idx,
                                             receiver_diameter=receiver_diameter)
        c = (np.arange(pixel_pupil) - pixel_pupil / 2 + 0.5) * pixel_pitch
        inside = np.hypot(c[:, None], c[None, :]) <= receiver_diameter / 2
        mask = cpuArray(power_loss.receiver_mask) > 0
        np.testing.assert_array_equal(mask, inside)

        positions = [(c[i], c[j]) for i, j in zip(*np.nonzero(mask))]
        A = cpuArray(pupilstop.A)
        field = A * np.exp(2j * np.pi * phase / wavelengthInNm)
        expected = self._fresnel_intensity(field, pixel_pitch, wavelengthInNm, height, positions).sum() / \
            self._fresnel_intensity(A.astype(complex), pixel_pitch, wavelengthInNm, height, positions).sum()
        np.testing.assert_allclose(sr, expected, rtol=1e-6)

    @cpu_and_gpu
    def test_power_loss_fresnel_no_fft(self, target_device_idx, xp):
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1075.0

        prop, _ = self._upwards_prop(target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm,
                                     3.6e7, doFresnel=True)
        ef = self._propagate(prop)
        sr, _ = self._power_loss_sr(prop, ef, target_device_idx)
        np.testing.assert_allclose(sr, 1.0, rtol=1e-9)

        ef.A[:] *= 0.5
        ef.phaseInNm[:] = xp.asarray(300.0 * np.random.default_rng(3).standard_normal((pixel_pupil, pixel_pupil)))
        sr, _ = self._power_loss_sr(prop, ef, target_device_idx)
        np.testing.assert_allclose(sr, 0.25, rtol=1e-9)

    @cpu_and_gpu
    def test_power_loss_invalid_receiver_diameter(self, target_device_idx, xp):
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1075.0
        prop, _ = self._upwards_prop(target_device_idx, xp, pixel_pupil, pixel_pitch, wavelengthInNm,
                                     3.6e7, doFresnel=True)
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)
        with self.assertRaises(ValueError):
            PowerLoss(simul_params=simul_params, prop=prop, receiver_diameter=0.0,
                      target_device_idx=target_device_idx)
