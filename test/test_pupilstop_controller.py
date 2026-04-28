import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.pupilstop import Pupilstop
from specula.processing_objects.pupilstop_controller import PupilstopController

from test.specula_testlib import cpu_and_gpu


class TestPupilstopController(unittest.TestCase):

    @cpu_and_gpu
    def test_generation_time_is_refreshed_each_iteration(self, target_device_idx, xp):
        simul_params = SimulParams(pixel_pupil=32, pixel_pitch=1.0)
        pupilstop = Pupilstop(simul_params=simul_params, target_device_idx=target_device_idx)

        controller = PupilstopController(
            pupilstop=pupilstop,
            target_device_idx=target_device_idx
        )
        controller.setup()

        self.assertFalse(controller.update_mask)

        t1 = controller.seconds_to_t(0.001)
        self.assertTrue(controller.check_ready(t1))
        controller.trigger()
        controller.post_trigger()
        self.assertEqual(controller.outputs['out_layer'].generation_time, t1)

        t2 = controller.seconds_to_t(0.002)
        self.assertTrue(controller.check_ready(t2))
        controller.trigger()
        controller.post_trigger()
        self.assertEqual(controller.outputs['out_layer'].generation_time, t2)

    @cpu_and_gpu
    def test_optional_inputs_update_geometry_and_mask(self, target_device_idx, xp):
        simul_params = SimulParams(pixel_pupil=40, pixel_pitch=1.0)
        pupilstop = Pupilstop(simul_params=simul_params, target_device_idx=target_device_idx)

        base_mask = cpuArray(pupilstop.A.copy())

        controller = PupilstopController(
            pupilstop=pupilstop,
            target_device_idx=target_device_idx
        )

        rot = BaseValue(value=[15.0], target_device_idx=target_device_idx)
        shift = BaseValue(value=[2.0, -1.0], target_device_idx=target_device_idx)
        magnification = BaseValue(value=[0.95], target_device_idx=target_device_idx)

        controller.inputs['in_rotation_deg'].set(rot)
        controller.inputs['in_shift_xy_px'].set(shift)
        controller.inputs['in_magnification'].set(magnification)

        controller.setup()

        self.assertTrue(controller.update_mask)

        t = controller.seconds_to_t(0.001)
        rot.generation_time = t
        shift.generation_time = t
        magnification.generation_time = t

        self.assertTrue(controller.check_ready(t))
        controller.trigger()
        controller.post_trigger()

        out_layer = controller.outputs['out_layer']

        self.assertAlmostEqual(float(out_layer.rotInDeg), 15.0)
        np.testing.assert_allclose(cpuArray(out_layer.shiftXYinPixel), [2.0, -1.0], atol=1e-6)
        self.assertAlmostEqual(float(out_layer.magnification), 0.95)
        self.assertEqual(out_layer.generation_time, t)
        self.assertFalse(np.array_equal(cpuArray(out_layer.A), base_mask))

        updated_mask = cpuArray(out_layer.A)
        self.assertEqual(updated_mask.shape, base_mask.shape)
        self.assertTrue(np.any(np.not_equal(updated_mask, base_mask)))
