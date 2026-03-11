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
        simul_params = SimulParams(time_step=1)

        filt = DummyFilter(simul_params=simul_params, nfilter=nfilter, delay=delay,
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
