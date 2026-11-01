import specula
specula.init(0)  # Default target device

import pytest

from specula.data_objects.simul_params import SimulParams
from specula.base_value import BaseValue

from specula.processing_objects.base_filter import BaseFilter
from test.specula_testlib import cpu_and_gpu


class DummyFilter(BaseFilter):
    """Minimal concrete implementation for testing BaseFilter."""

    def trigger_code(self):
        # Simple pass-through behavior
        self.output_buffer[:, 0] = self.delta_comm

    def reset_states(self):
        super().reset_states()


class TestBaseFilter:

    @cpu_and_gpu
    def test_reset_states_resets_output_buffer(self, target_device_idx, xp):
        nfilter = 4
        delay = 2.5

        filt = DummyFilter(nfilter=nfilter, delay=delay,
                           target_device_idx=target_device_idx)

        # Fill buffer with non-zero values
        filt.output_buffer[:] = xp.random.rand(*filt.output_buffer.shape)

        # Ensure buffer is not zero before reset
        assert not xp.allclose(filt.output_buffer, 0)

        # Call reset
        filt.reset_states()

        # Verify buffer reset
        assert xp.allclose(filt.output_buffer, 0)

        # Shape should remain unchanged
        expected_buffer_length = int(xp.ceil(delay)) + 1
        assert filt.output_buffer.shape == (nfilter, expected_buffer_length)

    def test_input_sizes_and_gain_mod(self):
        from specula.base_value import BaseValue
        filt = DummyFilter(nfilter=3, delay=0, target_device_idx=-1)
        filt.local_inputs = {'delta_comm': BaseValue(value=specula.np.ones(2), target_device_idx=-1),
                             'gain_mod': None}
        with pytest.raises(ValueError, match='delta_comm has size 2'):
            filt.prepare_trigger(0)
        # A scalar input is broadcast
        filt.local_inputs['delta_comm'] = BaseValue(value=specula.np.ones(1) * 2, target_device_idx=-1)
        filt.local_inputs['gain_mod'] = BaseValue(value=specula.np.ones(2), target_device_idx=-1)
        with pytest.raises(ValueError, match="gain_mod size 2 doesn't match"):
            filt.prepare_trigger(0)
        filt.local_inputs['gain_mod'] = BaseValue(value=specula.np.ones(1) * 0.5, target_device_idx=-1)
        filt.prepare_trigger(0)
        assert list(filt.delta_comm) == [2, 2, 2]
        assert list(filt._gain_mod) == [0.5, 0.5, 0.5]
        # Without the gain_mod input anymore, the gains go back to 1
        filt.local_inputs['gain_mod'] = None
        filt.prepare_trigger(0)
        assert list(filt._gain_mod) == [1, 1, 1]

