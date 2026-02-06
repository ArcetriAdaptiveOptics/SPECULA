import specula

specula.init(0)  # Default target device

import unittest
import os

from specula import cpuArray

from specula.data_objects.source import Source
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_infinite_evolution import AtmoInfiniteEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.processing_objects.modal_analysis import ModalAnalysis
from specula.data_objects.simul_params import SimulParams
from test.specula_testlib import cpu_and_gpu

import numpy as np

@unittest.skipIf(os.getenv('CI') == 'true',
                     "Disable for CI issues with Ubuntu and Python >=3.11")
class TestModalAnalysisUnwrapping(unittest.TestCase):

    @cpu_and_gpu
    def test_modal_analysis_unwrapping(self, target_device_idx, xp):
        simul_params = SimulParams(zenithAngleInDeg=0.0, pixel_pupil=120,
                                   pixel_pitch=0.01, time_step=1)

        # Atmosphere
        seeing = WaveGenerator(constant=0.9, target_device_idx=target_device_idx)
        wind_speed = WaveGenerator(constant=[0, 0, 0, 0], target_device_idx=target_device_idx)
        wind_direction = WaveGenerator(constant=[0, 0, 0, 0], target_device_idx=target_device_idx)
        atmo = AtmoInfiniteEvolution(simul_params,
                                     L0=20,  # [m] Outer scale
                                     heights=[0., 40., 120., 200.],
                                     Cn2=[0.769, 0.104, 0.127, 0.0],
                                     fov=8.0,
                                     target_device_idx=target_device_idx)

        # Physical and geometrical propagation to source
        uplink_source = Source(polar_coordinates=[0.0, 0.0], magnitude=0, height=300,
                               wavelengthInNm=1550)
        prop_up_phys = AtmoPropagation(simul_params, source_dict={'uplink_source': uplink_source},
                                  target_device_idx=target_device_idx, wavelengthInNm=1550,
                                  upwards=True, doFresnel=True)
        prop_up_geom = AtmoPropagation(simul_params, source_dict={'uplink_source': uplink_source},
                                  target_device_idx=target_device_idx)

        # Modal analysis
        modal_analsis_phys = ModalAnalysis(npixels=120, nmodes=2,
                                           type_str='zernike', wavelengthInNm=1550, dorms=True)
        modal_analsis_geom = ModalAnalysis(npixels=120, nmodes=2,
                                           type_str='zernike', dorms=True)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop_up_phys.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])
        prop_up_geom.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])
        modal_analsis_phys.inputs['in_ef'].set(prop_up_phys.outputs['out_uplink_source_ef'])
        modal_analsis_geom.inputs['in_ef'].set(prop_up_geom.outputs['out_uplink_source_ef'])
        for objlist in [[seeing, wind_speed, wind_direction], [atmo], \
                        [prop_up_phys, prop_up_geom], [modal_analsis_phys, modal_analsis_geom]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
                obj.post_trigger()

        modes_phys = cpuArray(modal_analsis_phys.outputs['out_modes'].value)
        modes_geom = cpuArray(modal_analsis_geom.outputs['out_modes'].value)
        np.testing.assert_allclose(modes_phys, modes_geom, rtol=3)
