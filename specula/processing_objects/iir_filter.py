from specula.base_processing_obj import InputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.processing_objects.base_filter import BaseFilter
from specula.data_objects.iir_filter_data import IirFilterData


class IirFilter(BaseFilter):
    """ 
    Infinite Impulse Response filter processing object.
    Implements IIR filtering with optional integration control.
    
    Parameters
    ----------
    iir_filter_data : IirFilterData
        Filter coefficients (numerator and denominator)
    delay : float [1], optional
        Delay in frames to apply to the output (default: 0)
    integration : bool
        If False, disables feedback terms (converts IIR to FIR).
        This is done by masking the denominator coefficients while
        preserving the normalizing factor. (default: True)
    target_device_idx : int [1], optional
        Target device for computation (-1 for CPU, >=0 for GPU)
    precision : int [1], optional
        Numerical precision (0 for double, 1 for single)
    
    Notes
    -----
    When integration=False, the filter becomes purely feedforward (FIR),
    removing all feedback/memory from previous outputs while maintaining
    the gain characteristics defined by the numerator coefficients.
    """

    def __init__(self,
                 iir_filter_data: IirFilterData,
                 delay: float = 0,
                 integration: bool = True,
                 target_device_idx=None,
                 precision=None):

        self.iir_filter_data = iir_filter_data

        super().__init__(
            nfilter=iir_filter_data.nfilter,
            delay=delay,
            target_device_idx=target_device_idx,
            precision=precision)

        self.inputs['in_ost'] = InputValue(type=BaseValue, optional=True)

        # IIR-specific state
        self._ist = self.xp.zeros_like(iir_filter_data.num)
        self._ost = self.xp.zeros_like(iir_filter_data.den)

        # Integration control
        self._den_mask = self.xp.ones_like(self.iir_filter_data.den)
        if not integration:
            self._den_mask[:, :-1] = 0

    @classmethod
    def input_names(cls):
        result = super().input_names()
        result.update({
            'in_ost': InputDesc(BaseValue, 'State update to subtract from integrators (optional)')
        })
        return result

    @classmethod
    def output_names(cls):
        return super().output_names()

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        in_ost_input = self.local_inputs.get('in_ost')
        if in_ost_input is not None and in_ost_input.value is not None:
            ost_update = in_ost_input.value
            ost_update_array = self.xp.asarray(ost_update, dtype=self.dtype).ravel()

            # 1. Update the filter state
            for i in range(self.output_buffer.shape[1]):
                self.output_buffer[:, i] -= ost_update_array

            # 2. PURGE THE DELAY PIPELINE
            for j in range(self._ost.shape[1]):
                self._ost[:, j] -= ost_update_array

    def trigger_code(self):
        """IIR filter computation."""
        sden = self.iir_filter_data.den.shape
        snum = self.iir_filter_data.num.shape
        no = sden[1]
        ni = snum[1]

        # Shift state buffers
        self._ost[:, :-1] = self._ost[:, 1:]
        self._ost[:, -1] = 0
        self._ist[:, :-1] = self._ist[:, 1:]
        self._ist[:, -1] = 0

        # New input
        self._ist[:, ni - 1] = self.delta_comm

        # Compute output
        factor = 1 / self.iir_filter_data.den[:, no - 1]
        num_contrib = self.xp.sum(
            self.iir_filter_data.num * self._gain_mod[:, None] * self._ist, axis=1)
        den_contrib = self.xp.sum(
            self.iir_filter_data.den[:, :no - 1] * 
            self._den_mask[:, :no - 1] * 
            self._ost[:, :no - 1], axis=1)

        output = factor * (num_contrib - den_contrib)
        self._ost[:, no - 1] = output

        # Store in buffer
        self.output_buffer[:, 0] = output

    def reset_states(self):
        """Reset IIR internal states."""
        super().reset_states()
        self._ist[:] = 0
        self._ost[:] = 0
