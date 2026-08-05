from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue


class RoundToMultiple(BaseProcessingObj):
    """
    Round-to-nearest-multiple processing object.

    Computes, element-wise: ``gain * multiple * round(in_value / multiple)``.

    Primary application:
    Resolving an integer number-of-wavelengths ambiguity in a continuous
    residual (e.g. a petal-piston estimate expressed in nm), by snapping it
    to the nearest multiple of ``multiple`` (e.g. lambda_wfs). Typically fed
    from a ground-truth signal (not available to a real WFS) and gated in
    time (e.g. via a ScheduleGenerator) to apply the correction once, after
    the closed-loop transient, rather than every frame.
    """

    def __init__(self,
                 multiple: float,
                 gain: float = 1.0,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if multiple == 0:
            raise ValueError("multiple must be non-zero")

        self.multiple = multiple
        self.gain = gain

        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.out_value = BaseValue(target_device_idx=self.target_device_idx,
                                    precision=self.precision)
        self.outputs['out_value'] = self.out_value

    @classmethod
    def input_names(cls):
        return {'in_value': InputDesc(BaseValue, 'Input vector to be snapped to the nearest multiple')}

    @classmethod
    def output_names(cls):
        return {'out_value': OutputDesc(BaseValue, 'Rounded output vector, gain-scaled')}

    def setup(self):
        super().setup()

        in_value = self.local_inputs['in_value']
        self.out_value.value = self.xp.zeros_like(in_value.get_value())

    def trigger_code(self):
        in_value = self.local_inputs['in_value'].get_value()
        self.out_value.value[:] = self.gain * self.multiple * self.xp.round(in_value / self.multiple)
        self.out_value.generation_time = self.current_time
