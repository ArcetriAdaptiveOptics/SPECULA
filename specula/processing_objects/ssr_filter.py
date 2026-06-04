from specula.processing_objects.base_filter import BaseFilter
from specula.data_objects.ssr_filter_data import SsrFilterData


class SsrFilter(BaseFilter):
    '''
    State Space Representation filter processing object. 
    Based on Time Control filter, which implements discrete-time state-space filtering.
    
    Discrete-time state-space filtering implementation:
    x[k+1] = A*x[k]  + B*u[k]
    y[k]   = C*x[k'] + D*u[k]
    
    where x[k'] is either x[k] or x[k+1] depending on output_uses_new_state parameter.
    
    All filters are handled simultaneously with single matrix operations.
    
    Parameters
    ----------
    ssr_filter_data : SsrFilterData
        State-space matrices (A, B, C, D) in block-diagonal form
    delay : float [1], optional
        Output delay in frames (default: 0)
    output_uses_new_state : bool
        If True, output equation uses updated state: y[k] = C*x[k+1] + D*u[k]
        If False, output equation uses current state: y[k] = C*x[k] + D*u[k]
        (default: True, which is standard for discrete integrators)
    target_device_idx : int [1], optional
        Target device index
    precision : int [1], optional
        Numerical precision
    '''
    def __init__(self,
                 ssr_filter_data: SsrFilterData,
                 delay: float = 0,
                 output_uses_new_state: bool = True,
                 target_device_idx=None,
                 precision=None):

        self.ssr_filter_data = ssr_filter_data
        self.output_uses_new_state = output_uses_new_state

        super().__init__(
            nfilter=ssr_filter_data.nfilter,
            delay=delay,
            target_device_idx=target_device_idx,
            precision=precision)

        # SSR-specific state
        self._x = self.xp.zeros(ssr_filter_data.total_states, dtype=self.dtype)

    @classmethod
    def input_names(cls):
        return super().input_names()

    @classmethod
    def output_names(cls):
        return super().output_names()

    def trigger_code(self):
        """State-space filter computation."""
        A = self.ssr_filter_data.A
        B = self.ssr_filter_data.B
        C = self.ssr_filter_data.C
        D = self.ssr_filter_data.D

        # Modulated input
        u = self.delta_comm * self._gain_mod

        # State update
        x_new = A @ self._x + B @ u

        # Output
        x_for_output = x_new if self.output_uses_new_state else self._x
        y = C @ x_for_output + D @ u

        # Update state
        self._x = x_new

        # Store in buffer
        self.output_buffer[:, 0] = y

    def reset_states(self):
        """Reset SSR internal states."""
        super().reset_states()
        self._x[:] = 0
