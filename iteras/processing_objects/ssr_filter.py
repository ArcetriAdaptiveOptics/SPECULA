from iteras.processing_objects.base_filter import BaseFilter
from iteras.data_objects.ssr_filter_data import SsrFilterData
from iteras.lib.graph_matmul import matmul


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
        # Matrices in the dtype of this object, like the state and the input: the products
        # are then the same with and without a CUDA graph (see graph_matmul.matmul()).
        # Copies only if the dtype differs: later changes to the matrices are then not seen
        self._A, self._B, self._C, self._D = (self.to_xp(m, dtype=self.dtype) for m in
            (ssr_filter_data.A, ssr_filter_data.B, ssr_filter_data.C, ssr_filter_data.D))

    @classmethod
    def input_names(cls):
        return super().input_names()

    @classmethod
    def output_names(cls):
        return super().output_names()

    def can_accelerate(self):
        # The products use graph_matmul.matmul(), which can be captured,
        # and the state is updated in place
        return True

    def trigger_code(self):
        """State-space filter computation."""
        A, B, C, D = self._A, self._B, self._C, self._D

        # Modulated input
        u = self.delta_comm * self._gain_mod

        # State update
        x_new = matmul(A, self._x, self.xp) + matmul(B, u, self.xp)

        # Output
        x_for_output = x_new if self.output_uses_new_state else self._x
        y = matmul(C, x_for_output, self.xp) + matmul(D, u, self.xp)

        # Update state, in place: the CUDA graph of an accelerated object keeps using it
        self._x[:] = x_new

        # Store in buffer
        self.output_buffer[:, 0] = y

    def reset_states(self):
        """Reset SSR internal states."""
        super().reset_states()
        self._x[:] = 0
