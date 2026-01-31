
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue


class OpticalGainEstimator(BaseProcessingObj):
    """
    Optical Gain Estimator based on demodulated signals.
    Uses two demodulated values (from delta-command and command) to estimate
    the optical gain of the system.
    
    The optical gain is updated using:
    opticalGain = opticalGain - (1 - demod_delta_cmd/demod_cmd) * gain * opticalGain
    """

    def __init__(self,
                 gain: float,
                 initial_optical_gain: float = 1.0,
                 #idx_array: list = None, # not supported yet
                 #expression: list = None, # not supported yet
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.gain = gain
        self.initial_optical_gain = initial_optical_gain

        # Optional advanced output mapping
        # Not supported yet
        self.idx_array = None
        self.expression = None

        # Internal optical gain storage
        self.optical_gain = BaseValue(
            value=self.dtype(initial_optical_gain),
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Output value (can be different from internal optical_gain if using expressions)
        self.output = BaseValue(
            value=self.dtype(initial_optical_gain),
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Initialize values
        self.optical_gain.value = self.dtype(self.initial_optical_gain)
        self.output.value = self.dtype(self.initial_optical_gain)

        # Inputs
        self.inputs['in_demod_delta_command'] = InputValue(type=BaseValue)
        self.inputs['in_demod_command'] = InputValue(type=BaseValue)

        # Outputs
        self.outputs['optical_gain'] = self.optical_gain
        self.outputs['output'] = self.output

        self.verbose = False

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        self.current_demod_delta_cmd = self.local_inputs['in_demod_delta_command']
        self.current_demod_cmd = self.local_inputs['in_demod_command']

    def trigger_code(self):
        t = self.current_time

        # Update optical gain if both inputs are ready
        if (self.current_demod_delta_cmd.generation_time == t and
            self.current_demod_cmd.generation_time == t):

            self._update_optical_gain()

        # Calculate output using expressions if provided
        self._calculate_output(t)

    def _update_optical_gain(self):
        """
        Update the internal optical gain based on demodulated signals.
        """
        demod_delta = self.current_demod_delta_cmd.value
        demod_cmd = self.current_demod_cmd.value
        current_gain = self.optical_gain.value

        # Avoid division by zero
        if self.xp.abs(demod_cmd) > 1e-12:
            ratio = demod_delta / demod_cmd
            # Update formula from IDL code
            updated_gain = current_gain - (1.0 - ratio) * self.gain * current_gain

            self.optical_gain.value = updated_gain
            self.optical_gain.generation_time = self.current_time

            if self.verbose:
                print(f"Optical gain updated: {float(current_gain):.6f} -> {float(updated_gain):.6f}")
        else:
            if self.verbose:
                print("Warning: demod_command too small, skipping optical gain update")

    def _calculate_output(self, t):
        """
        Calculate output value, potentially using idx_array and expression.
        """
        if self.idx_array is not None and self.expression is not None:
            # Advanced output calculation using expressions
            # This case is not implemented yet
            raise NotImplementedError("Advanced output calculation with idx_array and expression is not implemented.")
        else:
            # Simple case: output equals optical gain
            output = self.optical_gain.value

        # Ensure output doesn't exceed 1.0 (as in IDL code)
        if hasattr(output, '__iter__'):
            output = self.xp.minimum(output, 1.0)
        else:
            output = min(float(output), 1.0)

        # Handle scalar case
        if hasattr(output, '__len__') and len(output) == 1:
            output = float(output[0])

        self.output.value = output
        self.output.generation_time = t

        if self.verbose:
            print(f'Optical gain output: {output}')
