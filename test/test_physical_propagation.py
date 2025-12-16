import os
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
        wind_speed = WaveGenerator(constant=[0,0,0,0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0,0,0,0], target_device_idx=target_device_idx)

        uplink_source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height = 400., wavelengthInNm=1550)
        downlink_source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height = 400., wavelengthInNm=1550)

        atmo = AtmoInfiniteEvolution(simul_params,
                             L0=20,  # [m] Outer scale
                             heights=[0., 40., 120., 200.],
                             Cn2=[0.769,0.104,0.127,0.0],
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

        rmse = np.sqrt(((downlink_phase - uplink_phase) ** 2).mean())

        # check that upwards and downwards propagated phase are close
        self.assertTrue(rmse < 1.0)
        print(rmse)
