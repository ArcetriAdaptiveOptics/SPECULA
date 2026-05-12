import specula
specula.init(0)  # Default target device

import unittest
import numpy as np

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.petal_unwrapper import PetalUnwrapper 

from test.specula_testlib import cpu_and_gpu

class TestPetalUnwrapper(unittest.TestCase):

    def _get_dummy_objects(self, dim, n_petals, n_modes, target_device_idx, xp):
        """
        Helper method to generate a dummy Pupilstop and a fake IFunc matrix
        where the last modes represent basic petal sectors.
        """
        # Create a simple circular mask
        y, x = np.indices((dim, dim))
        y = y - dim/2 + 0.5
        x = x - dim/2 + 0.5
        r = np.sqrt(x**2 + y**2)
        mask_cpu = (r <= dim/2 - 2).astype(np.float32)
        mask = xp.array(mask_cpu)

        # Create pupilstop
        simul_params = SimulParams(time_step=1, pixel_pupil=dim, pixel_pitch=1.0)
        pupilstop = Pupilstop(simul_params, input_mask=mask, target_device_idx=target_device_idx)

        # Create fake IFunc where each petal mode is a pure piston on a sector
        theta = np.degrees(np.arctan2(y, x))
        angle_offset = 90.0

        idx = np.where(mask_cpu > 0)
        n_valid = len(idx[0])

        ifunc_data = np.zeros((n_modes, n_valid), dtype=np.float32)

        for i in range(n_petals):
            # Normalize angles to [0, 360)
            th = (theta - angle_offset) % 360
            sector_mask = (th >= i * 360.0 / n_petals) & (th < (i+1) * 360.0 / n_petals)
            sector_mask = sector_mask & (mask_cpu > 0)

            # Put the petals at the end of the mode matrix
            mode_idx = n_modes - n_petals + i
            ifunc_data[mode_idx, :] = sector_mask[idx].astype(np.float32)

        ifunc = IFunc(ifunc=xp.array(ifunc_data), mask=mask, target_device_idx=target_device_idx)

        return pupilstop, ifunc

    @cpu_and_gpu
    def test_initialization(self, target_device_idx, xp):
        """Tests that the offline geometry initialization works without crashing and builds correct shapes."""
        dim = 64
        n_petals = 6
        n_modes = 10
        pupilstop, ifunc = self._get_dummy_objects(dim, n_petals, n_modes, target_device_idx, xp)

        unwrapper = PetalUnwrapper(
            ifunc=ifunc,
            pupilstop=pupilstop,
            n_petals=n_petals,
            angle_offset_deg=90.0,
            thresh_in_nm=350.0,
            target_device_idx=target_device_idx
        )

        # H should map the 6 DOFs to 12 heights (2 heights * 6 spiders)
        self.assertEqual(unwrapper.H.shape, (12, n_petals))
        # H_dagger should be the pseudo-inverse
        self.assertEqual(unwrapper.H_dagger.shape, (n_petals, 12))

    @cpu_and_gpu
    def test_no_trigger_below_threshold(self, target_device_idx, xp):
        """Tests that if the gap is smaller than threshold, the command is left untouched."""
        dim = 64
        n_petals = 6
        n_modes = 10
        pupilstop, ifunc = self._get_dummy_objects(dim, n_petals, n_modes, target_device_idx, xp)

        unwrapper = PetalUnwrapper(
            ifunc=ifunc,
            pupilstop=pupilstop,
            n_petals=n_petals,
            angle_offset_deg=90.0,
            thresh_in_nm=350.0,
            target_device_idx=target_device_idx
        )

        # Command with small jumps (e.g. 100 nm on petal 0)
        in_comm_data = xp.zeros(n_modes, dtype=xp.float32)
        in_comm_data[-n_petals] = 100.0

        in_comm = BaseValue(value=in_comm_data, target_device_idx=target_device_idx)
        in_comm.generation_time = 1
        unwrapper.inputs['in_comm'].set(in_comm)

        unwrapper.setup()
        unwrapper.check_ready(1)
        unwrapper.trigger()
        unwrapper.post_trigger()

        out_comm = unwrapper.outputs['out_comm'].value
        out_ost = unwrapper.outputs['out_ost'].value

        # Should do nothing: out_comm == in_comm, out_ost == 0
        xp.testing.assert_allclose(out_comm, in_comm_data)
        xp.testing.assert_allclose(out_ost, xp.zeros_like(in_comm_data))

    @cpu_and_gpu
    def test_trigger_above_threshold_reset_to_zero(self, target_device_idx, xp):
        """Tests that an error above threshold is fully reset to zero (Hard Limiter)."""
        dim = 64
        n_petals = 6
        n_modes = 10
        pupilstop, ifunc = self._get_dummy_objects(dim, n_petals, n_modes, target_device_idx, xp)

        unwrapper = PetalUnwrapper(
            ifunc=ifunc,
            pupilstop=pupilstop,
            thresh_in_nm=350.0,
            target_device_idx=target_device_idx
        )

        # Inject a 1:1 gap mapping for testing
        unwrapper.H = xp.zeros((12, n_petals), dtype=xp.float32)
        unwrapper.H[0, 0] = 1.0
        unwrapper.H_dagger = xp.linalg.pinv(unwrapper.H)

        # Command 500 nm error (> 350 nm)
        in_comm_data = xp.zeros(n_modes, dtype=xp.float32)
        in_comm_data[-n_petals] = 500.0

        in_comm = BaseValue(value=in_comm_data, target_device_idx=target_device_idx)
        in_comm.generation_time = 1  # <--- L'OROLOGIO DI SPECULA
        unwrapper.inputs['in_comm'].set(in_comm)

        unwrapper.setup()
        unwrapper.check_ready(1)
        unwrapper.trigger()
        unwrapper.post_trigger()

        out_comm = unwrapper.outputs['out_comm'].value

        # With the new logic, 500 nm is entirely subtracted
        self.assertAlmostEqual(float(cpuArray(out_comm)[-n_petals]), 0.0, places=4)
