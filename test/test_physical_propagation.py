import specula

specula.init(0)  # Default target device

import unittest
from specula import np
from specula.data_objects.source import Source
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_infinite_evolution import AtmoInfiniteEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation, fraunhofer_far_field_propagation, angular_spectrum_propagation
from specula.data_objects.simul_params import SimulParams
from test.specula_testlib import cpu_and_gpu
from specula import cpuArray
from scipy.special import fresnel
from specula.data_objects.layer import Layer

from scipy.special import j1


class Test(unittest.TestCase):

    @cpu_and_gpu
    def test_physicalProp(self, target_device_idx, xp):
        simul_params = SimulParams(zenithAngleInDeg=0.0, pixel_pupil=120, pixel_pitch=0.008333, time_step=1)

        seeing = WaveGenerator(constant=0.7, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0, 0, 0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 0, 0], target_device_idx=target_device_idx)

        source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=150, wavelengthInNm=1550)

        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,  # [m] Outer scale
                                     heights=[0., 40., 120.],
                                     Cn2=[0.5, 0.4, 0.1],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)

        prop_down = AtmoPropagation(simul_params, source_dict={'downlink_source': source},
                                    target_device_idx=target_device_idx, wavelengthInNm=1550, doFresnel=True, padding_factor=3)
        prop_up = AtmoPropagation(simul_params, source_dict={'uplink_source': source},
                                  target_device_idx=target_device_idx, wavelengthInNm=1550, upwards=True,
                                  doFresnel=True, padding_factor=3)
        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop_down.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])
        prop_up.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])

        for objlist in [[seeing, wind_speed, wind_direction], [atmo], [prop_down, prop_up]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()
        downlink_phase = prop_down.outputs['out_downlink_source_ef'].phaseInNm
        uplink_phase = prop_up.outputs['out_uplink_source_ef'].phaseInNm

        self.assertTrue(id(prop_down.ef_fresnel) != id(prop_down.ft_ef1))
        self.assertTrue(id(prop_up.ef_fresnel) != id(prop_up.ft_ef1))

        rms = xp.sqrt(xp.mean((downlink_phase / np.max(downlink_phase) - uplink_phase / np.max(uplink_phase)) ** 2))
        self.assertTrue(rms < 0.1)

    @cpu_and_gpu
    def test_physicalProp_padding(self, target_device_idx, xp):
        simul_params = SimulParams(zenithAngleInDeg=0.0, pixel_pupil=240, pixel_pitch=0.008333, time_step=1)

        seeing = WaveGenerator(constant=0.7, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0, 0, 0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 0, 0], target_device_idx=target_device_idx)

        downlink_source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=150., wavelengthInNm=1550)

        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,  # [m] Outer scale
                                     heights=[0., 40., 120.],
                                     Cn2=[0.5, 0.4, 0.1],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)

        prop_down1 = AtmoPropagation(simul_params, source_dict={'downlink_source': downlink_source},
                                     target_device_idx=target_device_idx, wavelengthInNm=589, doFresnel=True,
                                     upwards=False, padding_factor=3)
        prop_down2 = AtmoPropagation(simul_params, source_dict={'downlink_source': downlink_source},
                                     target_device_idx=target_device_idx, wavelengthInNm=589, doFresnel=True,
                                     upwards=False)
        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop_down1.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])
        prop_down2.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])

        for objlist in [[seeing, wind_speed, wind_direction], [atmo], [prop_down1, prop_down2]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()

        downlink_phase1 = prop_down1.outputs['out_downlink_source_ef'].phaseInNm
        downlink_phase2 = prop_down2.outputs['out_downlink_source_ef'].phaseInNm

        rms = xp.sqrt(
            xp.mean((downlink_phase1 / np.max(downlink_phase1) - downlink_phase2 / np.max(downlink_phase2)) ** 2))
        self.assertTrue(rms < 0.1)

    def field_propagator(self, distanceInM, xp, ef_size, wavelengthInNm, pitch):
        L_pad = ef_size * pitch
        df = 1 / L_pad
        fx, fy = xp.meshgrid(df * xp.arange(-ef_size // 2, ef_size // 2),
                                  df * xp.arange(-ef_size // 2, ef_size // 2))
        fsq = fx ** 2 + fy ** 2
        H_AS = xp.exp(-1j * np.pi * distanceInM * wavelengthInNm * 1e-9 * fsq)
        return H_AS

    @cpu_and_gpu
    def test_physicalProp_accuracy(self, target_device_idx, xp):
        # Setup simulation parameters
        pixel_pupil = 120
        pixel_pitch = 0.00833
        padding = 8
        wavelengthInNm = 1550
        prop_distance = 3e3
        simul_params = SimulParams(pixel_pupil, pixel_pitch)

        seeing = WaveGenerator(constant=1.5, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0], target_device_idx=target_device_idx)
        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,  # [m] Outer scale
                                     heights=[1e3],
                                     Cn2=[1.0],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)
        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=prop_distance, wavelengthInNm=wavelengthInNm)
        prop1 = AtmoPropagation(
            simul_params,
            source_dict={'source': source},
            wavelengthInNm=wavelengthInNm,
            doFresnel=True,
            upwards=True,
            padding_factor=padding,
            target_device_idx=target_device_idx
        )
        prop1.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])

        prop2 = AtmoPropagation(
            simul_params,
            source_dict={'source': source},
            wavelengthInNm=wavelengthInNm,
            doFresnel=True,
            upwards=True,
            padding_factor=padding,
            target_device_idx=target_device_idx
        )
        prop2.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])

        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()

        prop1.setup()
        prop1.check_ready(1)
        prop1.trigger()
        prop1.post_trigger()
        phase = prop1.outputs['out_source_ef'].phaseInNm

        prop2.setup()
        # reset propagators to compare with different method
        propagators = [[None,
                        self.field_propagator(prop_distance - atmo.layer_list[0].height, xp, pixel_pupil * padding,
                                              wavelengthInNm, pixel_pitch), None]]
        prop2.propagators = propagators
        prop2.check_ready(1)
        prop2.trigger()
        prop2.post_trigger()
        phase_prev = prop2.outputs['out_source_ef'].phaseInNm

        rms = xp.sqrt(xp.mean((phase - phase_prev) ** 2))
        self.assertTrue(rms < 1)


    @cpu_and_gpu
    def test_physicalProp_scaling(self, target_device_idx, xp):
        # Propagation of square aperture
        diam_in = 2e-3
        diam_out = 4e-3
        wvl = 1e-6
        distanceInM = 0.1
        d_in = 9.4848e-6
        d_out = 28.1212e-6
        N = int(2**np.ceil(np.log2(diam_in/(2*d_in) + diam_out/(2*d_out) + (wvl*distanceInM)/(2*d_in*d_out))))

        simul_params = SimulParams(N, d_in)

        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=wvl*1e9, height=0.1)
        layer = Layer(
            dimx=N,
            dimy=N,
            pixel_pitch=d_in,
            height=0,
            target_device_idx=target_device_idx
        )
        vec1 = xp.arange(-N/2, N/2) * d_in
        x1, y1 = xp.meshgrid(vec1, vec1)
        x1 = xp.abs(x1/diam_in)
        y1 = xp.abs(y1/diam_in)
        x_rec = (x1 < 1/2).astype(float)
        y_rec = (y1 < 1/2).astype(float)
        x_rec[x1 == 1 / 2] = 0.5
        x_rec[y1 == 1 / 2] = 0.5
        ap = x_rec * y_rec
        layer.A = ap

        prop = AtmoPropagation(
            simul_params,
            source_dict={'on_axis': on_axis_source},
            wavelengthInNm=wvl*1e9,
            doFresnel=True,
            padding_factor=2,
            target_device_idx=target_device_idx
        )
        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])
        prop.setup()

        # Numerical propagation
        propagator = prop.asm_propagator(distanceInM, d_in, d_out)
        ef_in = xp.zeros([N * 2, N * 2], dtype=complex)
        buffer = xp.zeros([N * 2, N * 2], dtype=complex)
        ef_in[N // 2:N // 2 + N, N // 2:N // 2 + N] = layer.A * xp.exp(1j * layer.phaseInNm)
        angular_spectrum_propagation(ef_in, propagator, buffer, xp)

        # Analytical propagation
        coord = xp.arange(-N/2, N/2)
        x2, y2 = xp.meshgrid(coord*d_out, coord*d_out)
        x2_slice = x2[N // 2, :]
        N_f = (diam_in / 2)**2 / (wvl * distanceInM)
        sa1, ca1 = fresnel(cpuArray(-np.sqrt(2) * (np.sqrt(N_f) + x2_slice / np.sqrt(wvl * distanceInM))))
        sa2, ca2 = fresnel(cpuArray(np.sqrt(2) * (np.sqrt(N_f) - x2_slice / np.sqrt(wvl * distanceInM))))
        sb1, cb1 = fresnel(cpuArray(-np.sqrt(2) * (np.sqrt(N_f))))
        sb2, cb2 = fresnel(cpuArray(np.sqrt(2) * (np.sqrt(N_f))))
        U = 1 / 2j * ((ca2 - ca1) + 1j * (sa2 - sa1)) * ((cb2 - cb1) + 1j * (sb2 - sb1))

        ef_fresnel = prop.ef_fresnel[N // 2:N // 2 + N, N // 2:N // 2 + N]
        amp_asm = np.abs(cpuArray(ef_fresnel[N // 2, :]))**2
        amp_an = np.abs(U)**2
        self.assertTrue(xp.mean(abs(amp_asm) - abs(amp_an)) < 0.01)
        phase_asm = np.angle(cpuArray(ef_fresnel[N // 2, :]))
        phase_an = np.angle(U)
        self.assertTrue(xp.mean(abs(phase_asm) - abs(phase_an)) < 0.03)


    @cpu_and_gpu
    def test_physicalProp_infinite_source(self, target_device_idx, xp):
        # Setup simulation parameters
        pixel_pupil = 120
        pixel_pitch = 0.00833
        wavelengthInNm = 750
        simul_params = SimulParams(pixel_pupil, pixel_pitch)

        seeing = WaveGenerator(constant=0.4, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0, 0, 0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 0, 0], target_device_idx=target_device_idx)

        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,  # [m] Outer scale
                                     heights=[0., 40., 120.],
                                     Cn2=[0.5, 0.4, 0.1],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)
        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        source = Source(polar_coordinates=[0.0, 0.0], magnitude=0,  wavelengthInNm=wavelengthInNm)
        prop_phys = AtmoPropagation(
            simul_params,
            source_dict={'source': source},
            wavelengthInNm=wavelengthInNm,
            doFresnel=True,
            target_device_idx=target_device_idx
        )
        prop_phys.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])

        prop_geom = AtmoPropagation(
            simul_params,
            source_dict={'source': source},
            target_device_idx=target_device_idx
        )
        prop_geom.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])

        for objlist in [[seeing, wind_speed, wind_direction], [atmo], [prop_phys, prop_geom]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()

        phase_phys = prop_phys.outputs['out_source_ef'].phaseInNm
        phase_geom = prop_geom.outputs['out_source_ef'].phaseInNm

        rms = xp.sqrt(xp.mean((phase_phys / np.sum(phase_phys) - phase_geom / np.sum(phase_phys)) ** 2))
        self.assertTrue(rms < 1e-3)

    @cpu_and_gpu
    def test_fraunhoferProp_circular(self, target_device_idx, xp):
        # Setup simulation parameters
        pixel_pupil = 120
        L = 7.5e-3 # total size of grid [m]
        D = 1e-3 # aperture diameter [m]
        pixel_pitch = L / pixel_pupil
        wavelength = 1e-6
        source_height = 20.0
        padding = 1
        k = 2 * np.pi / wavelength

        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, wavelengthInNm=wavelength*1e9, height=source_height)
        layer = Layer(
            dimx=pixel_pupil,
            dimy=pixel_pupil,
            pixel_pitch=pixel_pitch,
            height=0,
            target_device_idx=target_device_idx
        )
        prop = AtmoPropagation(
            simul_params,
            source_dict={'source': source},
            wavelengthInNm=wavelength*1e9,
            doFresnel=True,
            upwards=True,
            padding_factor=padding,
            target_device_idx=target_device_idx
        )

        # circular aperture
        vec = xp.arange(-pixel_pupil / 2, pixel_pupil / 2) * pixel_pitch
        x1, y1 = xp.meshgrid(vec, vec)
        r = xp.sqrt(x1 ** 2 + y1 ** 2)
        aperture = (r < D / 2).astype(float)
        aperture[r == D/2] = 0.5
        layer.A = aperture
        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])
        prop.setup()

        # Numerical propagation
        propagator, x_out, y_out = prop.fraunhofer_propagator(source_height)
        ef_in = xp.zeros([pixel_pupil * padding, pixel_pupil * padding], dtype=complex)
        buffer = xp.zeros([pixel_pupil * padding, pixel_pupil * padding], dtype=complex)
        s = (pixel_pupil * padding - pixel_pupil) // 2
        ef_in[s:s + pixel_pupil, s:s + pixel_pupil] = layer.A * xp.exp(1j * layer.phaseInNm)
        fraunhofer_far_field_propagation(ef_in, propagator, buffer)

        # jinc function
        x = D * xp.sqrt(x_out ** 2 + y_out ** 2) / (wavelength * source_height)
        y = xp.ones_like(x, dtype = float)
        mask = x != 0
        y[mask] = 2.0 * j1(np.pi * x[mask]) / (np.pi * x[mask])

        # Analytical propagation
        ef_analytic = (xp.exp(1j * k / (2 * source_height) * (x_out ** 2 + y_out ** 2))
                   / (1j * wavelength * source_height) * (D ** 2 * np.pi / 4) * y)

        rms = xp.sqrt(xp.mean((abs(ef_in) - abs(ef_analytic)) ** 2))
        self.assertTrue(rms < 2e-3)

    @cpu_and_gpu
    def test_fraunhoferProp_circular_off_axis(self, target_device_idx, xp):
        # Setup simulation parameters
        pixel_pupil = 120
        L = 7.5e-3  # total size of grid [m]
        D = 1e-3  # aperture diameter [m]
        pixel_pitch = L / pixel_pupil
        wavelength = 1e-6
        source_height = 20.0
        padding = 3
        shift = 50
        k = 2 * np.pi / wavelength

        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, wavelengthInNm=wavelength * 1e9,
                        height=source_height)
        layer = Layer(
            dimx=pixel_pupil,
            dimy=pixel_pupil,
            pixel_pitch=pixel_pitch,
            height=0,
            target_device_idx=target_device_idx
        )
        prop = AtmoPropagation(
            simul_params,
            source_dict={'source': source},
            wavelengthInNm=wavelength * 1e9,
            doFresnel=True,
            upwards=True,
            beam_center=[0, shift],
            padding_factor=padding,
            target_device_idx=target_device_idx
        )

        # circular aperture
        vec = xp.arange(-pixel_pupil / 2, pixel_pupil / 2) * pixel_pitch
        x1, y1 = xp.meshgrid(vec, vec)
        r = xp.sqrt(x1 ** 2 + y1 ** 2)
        aperture = (r < D / 2).astype(float)
        aperture[r == D / 2] = 0.5
        layer.A = xp.roll(aperture, shift, axis=1)

        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])
        prop.setup()

        # Numerical propagation
        propagator, x_out, y_out = prop.fraunhofer_propagator(source_height)
        ef_in = xp.zeros([pixel_pupil * padding, pixel_pupil * padding], dtype=complex)
        buffer = xp.zeros([pixel_pupil * padding, pixel_pupil * padding], dtype=complex)
        s = (pixel_pupil * padding - pixel_pupil) // 2
        ef_in[s:s + pixel_pupil, s:s + pixel_pupil] = layer.A * xp.exp(1j * layer.phaseInNm)
        fraunhofer_far_field_propagation(ef_in, propagator, buffer)

        # jinc function
        x = D * xp.sqrt(x_out ** 2 + y_out ** 2) / (wavelength * source_height)
        y = xp.ones_like(x, dtype=float)
        mask = x != 0
        y[mask] = 2.0 * j1(np.pi * x[mask]) / (np.pi * x[mask])

        # Analytical propagation
        ef_analytic = (xp.exp(1j * k / (2 * source_height) * (x_out ** 2 + y_out ** 2))
                       / (1j * wavelength * source_height) * (D ** 2 * np.pi / 4) * y)

        prop_shifted = xp.roll(ef_in, -shift, axis=1)

        rms1 = xp.sqrt(xp.mean((abs(prop.ef_fresnel[:, :pixel_pupil * padding - shift]) - abs(
            ef_analytic[:, :pixel_pupil * padding - shift])) ** 2))
        rms2 = xp.sqrt(xp.mean((abs(prop_shifted[:, :pixel_pupil * padding - shift]) - abs(
            ef_analytic[:, :pixel_pupil * padding - shift])) ** 2))

        self.assertTrue(rms1 > rms2)

    @cpu_and_gpu
    def test_physicalProp_output_reference(self, target_device_idx, xp):
        simul_params = SimulParams(zenithAngleInDeg=0.0, pixel_pupil=120, pixel_pitch=0.008333, time_step=1)
        seeing = WaveGenerator(constant=0.7, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0, 0, 0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 0, 0], target_device_idx=target_device_idx)
        source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=38000e3, wavelengthInNm=1550)
        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,
                                     heights=[0., 40., 120.],
                                     Cn2=[0.5, 0.4, 0.1],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)
        prop_down = AtmoPropagation(simul_params, source_dict={'downlink_source': source},
                                    target_device_idx=target_device_idx, wavelengthInNm=1550, doFresnel=True,
                                    padding_factor=3)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop_down.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])

        for objlist in [[seeing, wind_speed, wind_direction], [atmo], [prop_down]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()

        self.assertTrue(id(prop_down.ef_fresnel) != id(prop_down.ft_ef1))

    def _build_deterministic_propagation(self, target_device_idx, precision, pixel_pupil,
                                         pixel_pitch, wavelengthInNm, layer_height, padding_factor,
                                         phase_pattern, pupil_pattern):
        """Build an AtmoPropagation with fixed, non-random Layers (smooth phase screen
        + circular pupil) so that propagator precision/accuracy can be checked deterministically."""
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, wavelengthInNm=wavelengthInNm)

        atmo_layer = Layer(pixel_pupil, pixel_pupil, pixel_pitch, layer_height,
                           target_device_idx=target_device_idx, precision=precision)
        atmo_layer.phaseInNm[:] = atmo_layer.to_xp(phase_pattern, dtype=atmo_layer.dtype)
        atmo_layer.generation_time = 1  # marks the input as refreshed for checkInputTimes

        pupil_layer = Layer(pixel_pupil, pixel_pupil, pixel_pitch, 0.0,
                            target_device_idx=target_device_idx, precision=precision)
        pupil_layer.A[:] = pupil_layer.to_xp(pupil_pattern, dtype=pupil_layer.dtype)
        pupil_layer.generation_time = 1

        prop = AtmoPropagation(simul_params, source_dict={'src': source},
                               wavelengthInNm=wavelengthInNm, doFresnel=True,
                               padding_factor=padding_factor,
                               target_device_idx=target_device_idx, precision=precision)
        prop.inputs['atmo_layer_list'].set([atmo_layer])
        prop.inputs['common_layer_list'].set([pupil_layer])
        prop.setup()
        return prop

    def _check_propagator_dtypes(self, prop, expected_complex_dtype):
        """Check that all stored propagator arrays have the expected complex dtype."""
        for prop_elem in prop.propagators:
            if prop_elem is None:
                continue
            self.assertEqual(len(prop_elem), 3)
            for arr in prop_elem:
                if arr is not None:
                    self.assertEqual(cpuArray(arr).dtype, expected_complex_dtype)

    @cpu_and_gpu
    def test_propagator_precision_and_accuracy_near_field(self, target_device_idx, xp):
        '''ASM (near-field) propagators: check storage dtype, and that a precision=1
        run matches a precision=0 run of the same deterministic setup.'''
        pixel_pupil = 64
        pixel_pitch = 0.01
        wavelengthInNm = 1550.0
        layer_height = 300.0
        padding_factor = 2

        yy, xx = np.meshgrid(np.arange(pixel_pupil), np.arange(pixel_pupil), indexing='ij')
        phase_pattern = 400.0 * np.sin(2 * np.pi * xx / pixel_pupil) * np.cos(2 * np.pi * yy / pixel_pupil)
        rr = np.sqrt((xx - pixel_pupil / 2 + 0.5) ** 2 + (yy - pixel_pupil / 2 + 0.5) ** 2)
        pupil_pattern = (rr < pixel_pupil / 2).astype(float)

        prop0 = self._build_deterministic_propagation(target_device_idx, 0, pixel_pupil, pixel_pitch,
                                                       wavelengthInNm, layer_height, padding_factor,
                                                       phase_pattern, pupil_pattern)
        prop1 = self._build_deterministic_propagation(target_device_idx, 1, pixel_pupil, pixel_pitch,
                                                       wavelengthInNm, layer_height, padding_factor,
                                                       phase_pattern, pupil_pattern)

        # z=300m is well below z_far_field for this pitch/pupil: ASM branch, not far field
        self.assertFalse(any(ff for ff in prop0.far_field_propagation if ff is not None))
        self._check_propagator_dtypes(prop0, np.complex128)
        self._check_propagator_dtypes(prop1, np.complex64)

        for prop in (prop0, prop1):
            prop.check_ready(1)
            prop.trigger()
            prop.post_trigger()

        phase0 = cpuArray(prop0.outputs['out_src_ef'].phaseInNm)
        phase1 = cpuArray(prop1.outputs['out_src_ef'].phaseInNm)
        rel = np.sqrt(np.mean((phase1 - phase0) ** 2)) / np.max(np.abs(phase0))
        self.assertLess(rel, 1e-5)

    @cpu_and_gpu
    def test_propagator_precision_and_accuracy_far_field(self, target_device_idx, xp):
        '''Fraunhofer (far-field) propagators: check storage dtype,
        and that a precision=1 run matches a precision=0 run of the same deterministic setup.'''
        pixel_pupil = 64
        pixel_pitch = 0.001
        wavelengthInNm = 1550.0
        layer_height = 10000.0  # 10 km: exceeds z_far_field for this pitch/pupil/padding
        padding_factor = 3

        yy, xx = np.meshgrid(np.arange(pixel_pupil), np.arange(pixel_pupil), indexing='ij')
        phase_pattern = 400.0 * np.sin(2 * np.pi * xx / pixel_pupil) * np.cos(2 * np.pi * yy / pixel_pupil)
        rr = np.sqrt((xx - pixel_pupil / 2 + 0.5) ** 2 + (yy - pixel_pupil / 2 + 0.5) ** 2)
        pupil_pattern = (rr < pixel_pupil / 2).astype(float)

        prop0 = self._build_deterministic_propagation(target_device_idx, 0, pixel_pupil, pixel_pitch,
                                                       wavelengthInNm, layer_height, padding_factor,
                                                       phase_pattern, pupil_pattern)
        prop1 = self._build_deterministic_propagation(target_device_idx, 1, pixel_pupil, pixel_pitch,
                                                       wavelengthInNm, layer_height, padding_factor,
                                                       phase_pattern, pupil_pattern)

        self.assertTrue(any(ff for ff in prop0.far_field_propagation if ff is not None))
        self._check_propagator_dtypes(prop0, np.complex128)
        self._check_propagator_dtypes(prop1, np.complex64)

        for prop in (prop0, prop1):
            prop.check_ready(1)
            prop.trigger()
            prop.post_trigger()

        phase0 = cpuArray(prop0.outputs['out_src_ef'].phaseInNm)
        phase1 = cpuArray(prop1.outputs['out_src_ef'].phaseInNm)
        rel = np.sqrt(np.mean((phase1 - phase0) ** 2)) / np.max(np.abs(phase0))
        self.assertLess(rel, 1e-5)

    @cpu_and_gpu
    def test_fresnel_beam_drift_same_direction_up_down(self, target_device_idx, xp):
        '''
        A tilted layer deflects the beam towards +grad(phase) in both propagation
        directions (reciprocity). Regression test: upwards propagation used to
        conjugate the field, which drifted the beam towards -grad(phase).
        '''
        pixel_pupil = 120
        pixel_pitch = 0.008333
        wavelengthInNm = 1550
        layer_height = 2000.0
        source_height = 4000.0
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch, zenithAngleInDeg=0.0)
        x = (np.arange(pixel_pupil) - (pixel_pupil - 1) / 2) * pixel_pitch

        def propagate(upwards, tilt):
            # Flat layer at 0 m (the pupil plane) and tilted layer above it, so that there
            # are layer_height meters of propagation after the tilt in both directions
            ground = Layer(pixel_pupil, pixel_pupil, pixel_pitch, height=0.0,
                           target_device_idx=target_device_idx)
            tilted = Layer(pixel_pupil, pixel_pupil, pixel_pitch, height=layer_height,
                           target_device_idx=target_device_idx)
            for layer in (ground, tilted):
                layer.A[:] = 1.0
                layer.generation_time = 1
            ground.phaseInNm[:] = 0.0
            tilted.phaseInNm[:] = xp.asarray(np.tile(tilt * x * 1e9, (pixel_pupil, 1)))

            source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=source_height,
                            wavelengthInNm=wavelengthInNm)
            prop = AtmoPropagation(simul_params, source_dict={'src': source}, doFresnel=True,
                                   upwards=upwards, wavelengthInNm=wavelengthInNm, padding_factor=3,
                                   target_device_idx=target_device_idx)
            prop.inputs['common_layer_list'].set([ground, tilted])
            prop.setup()
            prop.check_ready(1)
            prop.trigger()
            prop.post_trigger()
            return prop

        def centroid(prop):
            intensity = np.abs(cpuArray(prop.ef_fresnel)) ** 2
            coords = (np.arange(intensity.shape[1]) - (intensity.shape[1] - 1) / 2) * pixel_pitch
            return np.array([(intensity.sum(axis=0) * coords).sum(),
                             (intensity.sum(axis=1) * coords).sum()]) / intensity.sum()

        drift = {}
        for upwards in (False, True):
            prop = propagate(upwards, tilt=2e-5)
            # Relative to the same propagation without tilt, to remove the small
            # offset of the padded grid
            dx, dy = centroid(prop) - centroid(propagate(upwards, tilt=0.0))
            self.assertLess(abs(dy), 1e-3 * abs(dx))

            # Expected drift: output phase tilt (the layer tilt seen in the pupil,
            # scaled by the cone) times the propagation distance after the layer
            ef = prop.outputs['out_src_ef']
            field = cpuArray(ef.A) * np.exp(1j * 2 * np.pi * cpuArray(ef.phaseInNm) / wavelengthInNm)
            out_tilt = np.angle((field[:, 1:] * np.conj(field[:, :-1])).sum()) \
                       / (2 * np.pi / (wavelengthInNm * 1e-9)) / pixel_pitch
            np.testing.assert_allclose(dx, out_tilt * layer_height, rtol=0.1)
            drift[upwards] = dx

        self.assertGreater(drift[False], 0)
        self.assertGreater(drift[True], 0)
        np.testing.assert_allclose(drift[True], drift[False], rtol=0.1)


    @staticmethod
    def _exact_fresnel_square_grid(ef_in, pixel_pitch, wavelengthInNm, distanceInM):
        n = ef_in.shape[0]
        lam = wavelengthInNm * 1e-9
        k = 2 * np.pi / lam
        c = (np.arange(n) - (n - 1) / 2) * pixel_pitch
        kernel = np.exp(1j * k / (2 * distanceInM) * (c[:, None] - c[None, :]) ** 2)
        return pixel_pitch ** 2 / (1j * lam * distanceInM) * (kernel @ ef_in @ kernel.T)


    def _uplink_no_turbulence(self, target_device_idx, xp, source_height, layer_heights,
                              pixel_pupil=120, pixel_pitch=0.00833, padding=4,
                              wavelengthInNm=1550, zenith=50.0, **prop_kwargs):
        simul_params = SimulParams(pixel_pupil, pixel_pitch, zenithAngleInDeg=zenith)
        source = Source(polar_coordinates=[0.0, 0.0], magnitude=0,
                        wavelengthInNm=wavelengthInNm, height=source_height)
        coord = (np.arange(pixel_pupil) - (pixel_pupil - 1) / 2) * pixel_pitch
        xx, yy = np.meshgrid(coord, coord)
        pupil = (xx ** 2 + yy ** 2 <= (pixel_pupil * pixel_pitch / 2) ** 2).astype(float)

        def make_layer(height, A=None):
            layer = Layer(dimx=pixel_pupil, dimy=pixel_pupil, pixel_pitch=pixel_pitch,
                          height=height, target_device_idx=target_device_idx, precision=0)
            layer.A[:] = 1.0 if A is None else xp.asarray(A)
            layer.phaseInNm[:] = 0.0
            return layer

        prop = AtmoPropagation(simul_params, source_dict={'src': source}, doFresnel=True,
                               upwards=True, wavelengthInNm=wavelengthInNm,
                               padding_factor=padding, target_device_idx=target_device_idx,
                               precision=0, **prop_kwargs)
        prop.inputs['common_layer_list'].set([make_layer(0.0, A=pupil)])
        prop.inputs['atmo_layer_list'].set([make_layer(h) for h in layer_heights])
        prop.setup()
        prop.trigger_code()

        # Without turbulence the whole chain is free-space propagation over the slant range
        N = pixel_pupil * padding
        s = (N - pixel_pupil) // 2
        ef_in = np.zeros((N, N))
        ef_in[s:s + pixel_pupil, s:s + pixel_pupil] = pupil
        slant = source_height / np.cos(np.radians(zenith))
        ref = self._exact_fresnel_square_grid(ef_in, pixel_pitch, wavelengthInNm, slant)
        return prop, cpuArray(prop.ef_fresnel), ref


    @cpu_and_gpu
    def test_fresnel_long_distance_not_clipped(self, target_device_idx, xp):
        """Steps longer than the ASM limit must be propagated over their full length.
        """
        prop, ef, ref = self._uplink_no_turbulence(target_device_idx, xp, 400e3, [4000., 20000.])
        z_max_asm = prop.ef_size_padded * prop.pixel_pitch ** 2 / (prop.wavelengthInNm * 1e-9)
        slant_last = (400e3 - 20000.) / np.cos(np.radians(50.0))
        self.assertGreater(slant_last, z_max_asm)
        self.assertTrue(prop.far_field_propagation[-1])

        I, I_ref = np.abs(ef) ** 2, np.abs(ref) ** 2
        self.assertLess(np.abs(I - I_ref).max() / I_ref.max(), 1e-2)


    @cpu_and_gpu
    def test_fresnel_transform_followed_by_another_step(self, target_device_idx, xp):
        """A long intermediate step (Fresnel transform) followed by further steps must be exact.
        """
        prop, ef, ref = self._uplink_no_turbulence(target_device_idx, xp, 36000e3, [4000., 20000.])
        self.assertEqual(prop.far_field_propagation, [False, True, True])

        I, I_ref = np.abs(ef) ** 2, np.abs(ref) ** 2
        self.assertLess(np.abs(I - I_ref).max() / I_ref.max(), 1e-2)
        # the output phase (needed e.g. for coherent detection / ranging) must match as well
        mask = np.abs(ref) > 0.3 * np.abs(ref).max()
        dphi = np.angle(ef[mask] * np.conj(ref[mask]))
        dphi -= np.angle(np.mean(np.exp(1j * dphi)))
        self.assertLess(np.sqrt(np.mean(dphi ** 2)), 1e-2)


    @cpu_and_gpu
    def test_fresnel_no_clipping_warning(self, target_device_idx, xp):
        """No propagation distance is reduced any more, so no reduction warning is logged."""
        with self.assertNoLogs('specula', level='WARNING'):
            self._uplink_no_turbulence(target_device_idx, xp, 400e3, [4000., 20000.],
                                       power_loss_warning_threshold=None)


    @staticmethod
    def _loss_warnings(cm):
        return [r.getMessage() for r in cm.records if 'falls outside the padded grid' in r.getMessage()]


    @cpu_and_gpu
    def test_fresnel_power_loss_warning_last_step(self, target_device_idx, xp):
        """GEO uplink: the beam at the satellite is far larger than the padded grid.
        Only the last step loses light (> 99%).
        """
        with self.assertLogs('specula', level='WARNING') as cm:
            prop, _, _ = self._uplink_no_turbulence(target_device_idx, xp, 36000e3, [4000., 20000.])
        msgs = self._loss_warnings(cm)
        self.assertEqual(len(msgs), 1)
        self.assertIn('(last step)', msgs[0])
        self.assertIn('still correct', msgs[0])
        loss = float(msgs[0].split('%')[0].split()[-1])
        self.assertGreater(loss, 99.0)


    @cpu_and_gpu
    def test_fresnel_power_loss_warning_intermediate_step(self, target_device_idx, xp):
        """Light lost in an intermediate step is missing from all later planes: warn as inaccurate."""
        with self.assertLogs('specula', level='WARNING') as cm:
            self._uplink_no_turbulence(target_device_idx, xp, 400e3, [4000., 20000.], padding=2,
                                       power_loss_warning_threshold=0.005)
        msgs = self._loss_warnings(cm)
        intermediate = [m for m in msgs if '(last step)' not in m]
        self.assertEqual(len(intermediate), 1)
        self.assertIn('not accurate', intermediate[0])
        self.assertIn('propagation over 24891.6 m from the layer at 6222.9 m', intermediate[0])

    @staticmethod
    def _annular_pupil(pixel_pupil, pixel_pitch, obs=0.33):
        c = (np.arange(pixel_pupil) - (pixel_pupil - 1) / 2) * pixel_pitch
        r = np.hypot(c[:, None], c[None, :])
        diameter = pixel_pupil * pixel_pitch
        return ((r <= diameter / 2) & (r >= obs * diameter / 2)).astype(float)

    @staticmethod
    def _von_karman_screen(size, pixel_pitch, seed, r0=0.1, L0=25.):
        """von Karman phase screen [nm] (r0 [m] at 500 nm), size x size pixels"""
        rng = np.random.default_rng(seed)
        fx = np.fft.fftfreq(size, pixel_pitch)
        f2 = fx[:, None] ** 2 + fx[None, :] ** 2 + 1. / L0 ** 2
        psd = 0.023 * r0 ** (-5. / 3) * f2 ** (-11. / 6)
        psd[0, 0] = 0
        c = (rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))) \
            * np.sqrt(psd) / (size * pixel_pitch)
        return np.real(np.fft.ifft2(c)) * size * size * 500 / (2 * np.pi)

    def _fresnel_layers(self, target_device_idx, xp, upwards, pixel_pupil, pixel_pitch, wavelengthInNm,
                        heights, layer_size, pupil, phases=None, padding_factor=4, source_height=np.inf):
        """Fresnel propagation through atmospheric layers (layer_size x layer_size px, phases cut from
        the centre of the given screens) and a pupil stop at 0 m. Returns the propagation object and
        the output amplitude and phase [nm] (numpy)."""
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch, zenithAngleInDeg=0.0)
        source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=source_height,
                        wavelengthInNm=wavelengthInNm)
        prop = AtmoPropagation(simul_params, source_dict={'src': source}, doFresnel=True, upwards=upwards,
                               wavelengthInNm=wavelengthInNm, padding_factor=padding_factor, precision=0,
                               target_device_idx=target_device_idx)

        def make_layer(height, size, phase=None, A=None):
            layer = Layer(dimx=size, dimy=size, pixel_pitch=pixel_pitch, height=height, precision=0,
                          target_device_idx=target_device_idx)
            layer.A[:] = 1.0 if A is None else xp.asarray(A)
            layer.phaseInNm[:] = 0.0 if phase is None else xp.asarray(phase)
            layer.generation_time = 1
            return layer

        atmo_layers = []
        for i, height in enumerate(heights):
            phase = None
            if phases is not None:
                c0 = (phases[i].shape[0] - layer_size) // 2
                phase = phases[i][c0:c0 + layer_size, c0:c0 + layer_size]
            atmo_layers.append(make_layer(height, layer_size, phase))
        prop.inputs['common_layer_list'].set([make_layer(0.0, pixel_pupil, A=pupil)])
        prop.inputs['atmo_layer_list'].set(atmo_layers)
        prop.setup()
        prop.check_ready(1)
        prop.trigger()
        prop.post_trigger()
        ef = prop.outputs['out_src_ef']
        return prop, cpuArray(ef.A).copy(), cpuArray(ef.phaseInNm).copy()

    @staticmethod
    def _fresnel_reference_down(phases, pixel_pitch, wavelengthInNm, heights, pupil):
        """Independent downwards propagation of a plane wave through the whole phase screens
        (periodic standard DFT angular spectrum propagation), cut to the pupil (numpy)"""
        size = phases[0].shape[0]
        pixel_pupil = pupil.shape[0]
        fx = np.fft.fftfreq(size, pixel_pitch)
        f2 = fx[:, None] ** 2 + fx[None, :] ** 2
        field = np.ones((size, size), complex)
        distances = np.diff(np.concatenate(([0.0], heights)))   # from each layer to the next one below
        for i in range(len(heights) - 1, -1, -1):       # from the highest layer down to the pupil
            field *= np.exp(2j * np.pi * phases[i] / wavelengthInNm)
            field = np.fft.ifft2(np.fft.fft2(field) * np.exp(-1j * np.pi * wavelengthInNm * 1e-9 * distances[i] * f2))
        s = (size - pixel_pupil) // 2
        return field[s:s + pixel_pupil, s:s + pixel_pupil] * pupil

    @cpu_and_gpu
    def test_fresnel_full_grid_down_no_turbulence_no_edge(self, target_device_idx, xp):
        """Downwards without turbulence the field in the pupil is exactly the plane wave"""
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1064.0
        heights = [4000., 12000.]
        layer_size = 960
        pupil = self._annular_pupil(pixel_pupil, pixel_pitch)

        prop, A, _ = self._fresnel_layers(target_device_idx, xp, False, pixel_pupil, pixel_pitch,
                                          wavelengthInNm, heights, layer_size, pupil)
        self.assertTrue(prop.fresnel_full_grid)
        np.testing.assert_allclose(A[pupil > 0], 1.0, atol=1e-6)

    @cpu_and_gpu
    def test_fresnel_pupil_window_mode_has_edge_artefact(self, target_device_idx, xp):
        """Layers of pupil size: pupil-window mode, the downwards field is cut at the pupil window
        and its edge diffracts into the pupil"""
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1064.0
        heights = [4000., 12000.]
        pupil = self._annular_pupil(pixel_pupil, pixel_pitch)

        prop, A, _ = self._fresnel_layers(target_device_idx, xp, False, pixel_pupil, pixel_pitch,
                                          wavelengthInNm, heights, pixel_pupil, pupil, padding_factor=2)
        self.assertFalse(prop.fresnel_full_grid)
        I = A[pupil > 0] ** 2
        self.assertGreater(I.std() / I.mean(), 0.1)

    @cpu_and_gpu
    def test_fresnel_full_grid_down_turbulence_matches_reference(self, target_device_idx, xp):
        """Downwards with turbulence: scintillation and phase in the pupil = reference computed on
        the whole 8 m screens"""
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1064.0
        heights = [4000., 12000.]
        layer_size = 960
        pupil = self._annular_pupil(pixel_pupil, pixel_pitch)
        phases = [self._von_karman_screen(layer_size, pixel_pitch, seed) for seed in (1, 2)]

        ref = self._fresnel_reference_down(phases, pixel_pitch, wavelengthInNm, heights, pupil)
        m = pupil > 0
        I_ref = np.abs(ref[m]) ** 2
        self.assertGreater(I_ref.std() / I_ref.mean(), 0.2)                # real scintillation

        prop, A, ph = self._fresnel_layers(target_device_idx, xp, False, pixel_pupil, pixel_pitch,
                                           wavelengthInNm, heights, layer_size, pupil, phases=phases)
        self.assertTrue(prop.fresnel_full_grid)
        I = A[m] ** 2
        err = (I - I_ref).std() / I_ref.mean()
        self.assertLess(err, 0.01, f'intensity error {err:.4f}')
        self.assertGreater(np.corrcoef(I, I_ref)[0, 1], 0.9999)
        field = A * np.exp(2j * np.pi * ph / wavelengthInNm)
        dphi = np.angle(field[m] * np.conj(ref[m]))
        dphi -= np.angle(np.mean(np.exp(1j * dphi)))
        self.assertLess(np.sqrt(np.mean(dphi ** 2)), 0.05)                  # [rad]

    @cpu_and_gpu
    def test_fresnel_pupil_window_mode_is_worse(self, target_device_idx, xp):
        """The same turbulence in the pupil-window mode: much larger error"""
        pixel_pupil = 120
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1064.0
        heights = [4000., 12000.]
        layer_size = 960
        pupil = self._annular_pupil(pixel_pupil, pixel_pitch)
        phases = [self._von_karman_screen(layer_size, pixel_pitch, seed) for seed in (1, 2)]
        m = pupil > 0
        I_ref = np.abs(self._fresnel_reference_down(phases, pixel_pitch, wavelengthInNm, heights, pupil)[m]) ** 2

        _, A, _ = self._fresnel_layers(target_device_idx, xp, False, pixel_pupil, pixel_pitch, wavelengthInNm,
                                       heights, pixel_pupil, pupil, phases=phases, padding_factor=2)
        err_window = (A[m] ** 2 - I_ref).std() / I_ref.mean()
        _, A, _ = self._fresnel_layers(target_device_idx, xp, False, pixel_pupil, pixel_pitch, wavelengthInNm,
                                       heights, layer_size, pupil, phases=phases)
        err_full = (A[m] ** 2 - I_ref).std() / I_ref.mean()
        self.assertGreater(err_window, 20 * err_full)

    @cpu_and_gpu
    def test_asm_frequencies(self, target_device_idx, xp):
        """One angular spectrum step = numpy DFT reference (transfer function at the DFT
        frequencies), for an even and an odd grid size"""
        pixel_pitch = 1 / 120.
        wavelengthInNm = 1064.0
        distance = 1500.0
        rng = np.random.default_rng(5)
        for pixel_pupil, padding_factor in ((60, 2), (61, 3)):
            simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch, zenithAngleInDeg=0.0)
            source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=np.inf,
                            wavelengthInNm=wavelengthInNm)
            prop = AtmoPropagation(simul_params, source_dict={'src': source}, doFresnel=True,
                                   wavelengthInNm=wavelengthInNm, padding_factor=padding_factor, precision=0,
                                   target_device_idx=target_device_idx)
            size = pixel_pupil * padding_factor
            propagator = prop.asm_propagator(distance, pixel_pitch, pixel_pitch)
            field = np.exp(1j * rng.uniform(0, 2 * np.pi, (size, size)))
            ef = xp.asarray(field).copy()               # propagated in place
            angular_spectrum_propagation(ef, propagator, xp.zeros_like(ef), xp)

            fx = np.fft.fftfreq(size, pixel_pitch)
            f2 = fx[:, None] ** 2 + fx[None, :] ** 2
            ref = np.fft.ifft2(np.fft.fft2(field) * np.exp(-1j * np.pi * wavelengthInNm * 1e-9 * distance * f2))
            np.testing.assert_allclose(cpuArray(ef), ref, atol=1e-9)
