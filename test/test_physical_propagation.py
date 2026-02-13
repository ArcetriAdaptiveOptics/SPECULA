import specula

specula.init(0)  # Default target device

import unittest
from specula import np
from specula import cpuArray
from specula.data_objects.source import Source
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_infinite_evolution import AtmoInfiniteEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.simul_params import SimulParams
from test.specula_testlib import cpu_and_gpu


class Test(unittest.TestCase):
    @cpu_and_gpu
    def test_physicalProp(self, target_device_idx, xp):
        simul_params = SimulParams(zenithAngleInDeg=0.0, pixel_pupil=120, pixel_pitch=0.008333, time_step=1)

        seeing = WaveGenerator(constant=0.01, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0, 0, 0, 0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 0, 0, 0], target_device_idx=target_device_idx)

        uplink_source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=300., wavelengthInNm=1550)
        downlink_source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=300., wavelengthInNm=1550)

        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,  # [m] Outer scale
                                     heights=[0., 40., 120., 200.],
                                     Cn2=[0.769, 0.104, 0.127, 0.0],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)

        prop_down = AtmoPropagation(simul_params, source_dict={'downlink_source': downlink_source},
                                    target_device_idx=target_device_idx, wavelengthInNm=1550, doFresnel=True)
        prop_up = AtmoPropagation(simul_params, source_dict={'uplink_source': uplink_source},
                                  target_device_idx=target_device_idx, wavelengthInNm=1550, upwards=True,
                                  doFresnel=True)
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
        downlink_phase = cpuArray(prop_down.outputs['out_downlink_source_ef'].phaseInNm)
        uplink_phase = cpuArray(prop_up.outputs['out_uplink_source_ef'].phaseInNm)

        rel_error = xp.mean(abs(downlink_phase - uplink_phase) / abs(uplink_phase))

        # check that upwards and downwards propagated phase are close
        print(rel_error)
        self.assertTrue(rel_error < 1.0)

    @cpu_and_gpu
    def test_physicalProp_padding(self, target_device_idx, xp):
        simul_params = SimulParams(zenithAngleInDeg=0.0, pixel_pupil=120, pixel_pitch=0.008333, time_step=1)

        seeing = WaveGenerator(constant=0.01, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0, 0, 0, 0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 0, 0, 0], target_device_idx=target_device_idx)

        downlink_source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=500., wavelengthInNm=1550)

        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,  # [m] Outer scale
                                     heights=[0., 40., 120., 200.],
                                     Cn2=[0.769, 0.104, 0.127, 0.0],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)

        prop_down1 = AtmoPropagation(simul_params, source_dict={'downlink_source': downlink_source},
                                     target_device_idx=target_device_idx, wavelengthInNm=1550, doFresnel=True,
                                     padding_factor=3)
        prop_down2 = AtmoPropagation(simul_params, source_dict={'downlink_source': downlink_source},
                                     target_device_idx=target_device_idx, wavelengthInNm=1550, doFresnel=True,
                                     padding_factor=2)
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

        downlink_phase1 = cpuArray(prop_down1.outputs['out_downlink_source_ef'].phaseInNm)
        downlink_phase2 = cpuArray(prop_down2.outputs['out_downlink_source_ef'].phaseInNm)

        rel_error = xp.mean(abs(downlink_phase1 - downlink_phase2) / abs(downlink_phase1))

        # check that upwards and downwards propagated phase are close
        print(rel_error)
        self.assertTrue(rel_error < 1.0)

    @cpu_and_gpu
    def test_physicalProp_bandlimit(self, target_device_idx, xp):
        simul_params = SimulParams(zenithAngleInDeg=75.0, pixel_pupil=120, pixel_pitch=0.008333, time_step=1)

        seeing = WaveGenerator(constant=0.01, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0, 0, 0, 0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 0, 0, 0], target_device_idx=target_device_idx)

        uplink_source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=90e3, wavelengthInNm=1550)

        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,  # [m] Outer scale
                                     heights=[0., 40., 120., 200.],
                                     Cn2=[0.769, 0.104, 0.127, 0.0],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)

        prop_down1 = AtmoPropagation(simul_params, source_dict={'uplink_source': uplink_source},
                                     target_device_idx=target_device_idx, wavelengthInNm=1550, upwards=True,
                                     doFresnel=True, band_limit_factor=0.8)
        prop_down2 = AtmoPropagation(simul_params, source_dict={'uplink_source': uplink_source},
                                     target_device_idx=target_device_idx, wavelengthInNm=1550, upwards=True,
                                     doFresnel=True)
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

        uplink_phase1 = cpuArray(prop_down1.outputs['out_uplink_source_ef'].phaseInNm)
        uplink_phase2 = cpuArray(prop_down2.outputs['out_uplink_source_ef'].phaseInNm)

        rel_error = xp.mean(abs(uplink_phase1 - uplink_phase2) / abs(uplink_phase1))

        # check that upwards and downwards propagated phase are close
        print(rel_error)
        self.assertTrue(rel_error < 1.0)
