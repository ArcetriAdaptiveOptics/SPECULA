from specula.processing_objects.base_filter import BaseFilter
from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.simul_params import SimulParams


class IirFilter(BaseFilter):
    '''Infinite Impulse Response filter based Time Control
    
    This class implements IIR filtering with optional integration control.
    
    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters containing time step information
    iir_filter_data : IirFilterData
        Filter coefficients (numerator and denominator)
    delay : float, optional
        Delay in frames to apply to the output (default: 0)
    integration : bool, optional
        If False, disables feedback terms (converts IIR to FIR).
        This is done by masking the denominator coefficients while
        preserving the normalizing factor. (default: True)
    target_device_idx : int, optional
        Target device for computation (-1 for CPU, >=0 for GPU)
    precision : int, optional
        Numerical precision (0 for double, 1 for single)
    
    Notes
    -----
    When integration=False, the filter becomes purely feedforward (FIR),
    removing all feedback/memory from previous outputs while maintaining
    the gain characteristics defined by the numerator coefficients.
    '''

    def __init__(self,
                 simul_params: SimulParams,
                 iir_filter_data: IirFilterData,
                 delay: float = 0,
                 integration: bool = True,
                 target_device_idx=None,
                 precision=None):

        self.iir_filter_data = iir_filter_data

        super().__init__(
            simul_params=simul_params,
            nfilter=iir_filter_data.nfilter,
            delay=delay,
            target_device_idx=target_device_idx,
            precision=precision)

        # IIR-specific state
        self._ist = self.xp.zeros_like(iir_filter_data.num)
        self._ost = self.xp.zeros_like(iir_filter_data.den)

        # Integration control
        self._den_mask = self.xp.ones_like(self.iir_filter_data.den)
        if not integration:
            self._den_mask[:, :-1] = 0

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
