from specula.processing_objects.integrator import Integrator
from specula.base_processing_obj import InputDesc
from specula.data_objects.simul_params import SimulParams
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.scalar_values import FloatValue, IntValue


class DynamicIntegrator(Integrator):
    """
    Dynamic integrator with runtime-adjustable gain and reset capability.

    This class extends :class:`Integrator` by allowing dynamic updates of the
    integration gain and providing a reset mechanism for internal filter states
    during runtime.

    Interactive Inputs
    ------------------
    int_gain : BaseValue, optional
        Dynamically update the integrator gain.
    reset : BaseValue, optional
        Trigger to reset the internal integrator state.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 int_gain: float,
                 ff: list=None,
                 n_modes: int=None,
                 delay: float=0,
                 integration: bool=True,
                 target_device_idx: int=None,
                 precision: int=None
                ):
        """
        Parameters
        ----------
        simul_params : SimulParams
            Simulation parameters object.
        int_gain : float [1]
            Initial integrator gain.
        ff : list [1], optional
            Feedforward coefficients for the IIR filter.
        n_modes : int [1], optional
            Number of modes for modal integration.
        delay : float [1], optional
            Delay applied to the integrator (in simulation time units).
            Default is 0.
        integration : bool
            If True, enable integration behavior. Default is True.
        target_device_idx : int [1], optional
            Target device index for computation (e.g., CPU/GPU).
        precision : int [1], optional
            Numerical precision for internal data  (0 for double, 1 for single).
        """
        super().__init__(simul_params=simul_params,
                         int_gain=int_gain,
                         ff=ff,
                         n_modes=n_modes,
                         delay=delay,
                         integration=integration,
                         target_device_idx=target_device_idx,
                         precision=precision)

        self.inputs['reset'] = InputValue(type=IntValue, optional=True)
        self.inputs['int_gain'] = InputValue(type=FloatValue, optional=True)

    @classmethod
    def input_names(cls):
        result = super().input_names()
        result.update({
            'reset': InputDesc(BaseValue, 'Trigger to reset internal integrator state (optional)'),
            'int_gain': InputDesc(BaseValue, 'Dynamic integrator gain update (optional)')
        })
        return result

    @classmethod
    def output_names(cls):
        return super().output_names()

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Reset internal state
        reset_input = self.local_inputs['reset']
        if reset_input is not None and reset_input.generation_time == self.current_time:
            self.reset_states()

        gain_input = self.local_inputs['int_gain']
        if gain_input is not None and gain_input.generation_time == self.current_time:
            self.iir_filter_data.set_gain(gain_input.value)
