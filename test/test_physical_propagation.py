import specula

specula.init(0)  # Default target device

import unittest
from specula import np
from specula.data_objects.source import Source
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_infinite_evolution import AtmoInfiniteEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation
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
        ef_in = xp.zeros([N*2, N*2], dtype=complex)
        ef_in[N // 2:N // 2 + N, N // 2:N // 2 + N] = layer.A * xp.exp(1j * layer.phaseInNm)
        prop.angular_spectrum_propagation(ef_in, propagator)

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
        s = (pixel_pupil * padding - pixel_pupil) // 2
        ef_in[s:s + pixel_pupil, s:s + pixel_pupil] = layer.A * xp.exp(1j * layer.phaseInNm)
        prop.fraunhofer_far_field_propagation(ef_in, propagator)

        # jinc function
        x = D * xp.sqrt(x_out ** 2 + y_out ** 2) / (wavelength * source_height)
        y = xp.ones_like(x, dtype = float)
        mask = x != 0
        y[mask] = 2.0 * j1(np.pi * x[mask]) / (np.pi * x[mask])

        # Analytical propagation
        ef_analytic = (xp.exp(1j * k / (2 * source_height) * (x_out ** 2 + y_out ** 2))
                   / (1j * wavelength * source_height) * (D ** 2 * np.pi / 4) * y)

        rms = xp.sqrt(xp.mean((abs(prop.ef_fresnel) - abs(ef_analytic)) ** 2))
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
        s = (pixel_pupil * padding - pixel_pupil) // 2
        ef_in[s:s + pixel_pupil, s:s + pixel_pupil] = layer.A * xp.exp(1j * layer.phaseInNm)
        prop.fraunhofer_far_field_propagation(ef_in, propagator)

        # jinc function
        x = D * xp.sqrt(x_out ** 2 + y_out ** 2) / (wavelength * source_height)
        y = xp.ones_like(x, dtype=float)
        mask = x != 0
        y[mask] = 2.0 * j1(np.pi * x[mask]) / (np.pi * x[mask])

        # Analytical propagation
        ef_analytic = (xp.exp(1j * k / (2 * source_height) * (x_out ** 2 + y_out ** 2))
                       / (1j * wavelength * source_height) * (D ** 2 * np.pi / 4) * y)

        prop_shifted = xp.roll(prop.ef_fresnel, -shift, axis=1)

        rms1 = xp.sqrt(xp.mean((abs(prop.ef_fresnel[:, :pixel_pupil * padding - shift]) - abs(
            ef_analytic[:, :pixel_pupil * padding - shift])) ** 2))
        rms2 = xp.sqrt(xp.mean((abs(prop_shifted[:, :pixel_pupil * padding - shift]) - abs(
            ef_analytic[:, :pixel_pupil * padding - shift])) ** 2))

        self.assertTrue(rms1 > rms2)
