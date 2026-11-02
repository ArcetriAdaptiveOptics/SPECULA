from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.slopes import Slopes


class Slopes2D(BaseProcessingObj):
    """
    Converts a Slopes object into its 2D map representation
    (Slopes.get2d(): a pair of x/y slope maps, or a single intensity map),
    as a plain BaseValue. Useful to feed slopes into a 2D-input model (e.g.
    Conv2dNetTrainer/Tester/Rec) or into a DataBuffer for later training,
    since DataBuffer buffers Slopes.get_value() (the flat slopes vector)
    rather than its 2D map.
    """

    def __init__(self, target_device_idx: int = None, precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.inputs['in_slopes'] = InputValue(type=Slopes)
        self.out_value = BaseValue(target_device_idx=target_device_idx, precision=precision)
        self.outputs['out_value'] = self.out_value

    @classmethod
    def input_names(cls):
        return {'in_slopes': InputDesc(Slopes, 'Slopes to convert to a 2D map')}

    @classmethod
    def output_names(cls):
        return {'out_value': OutputDesc(BaseValue, '2D map (Slopes.get2d()) of the input slopes')}

    def trigger_code(self):
        slopes = self.local_inputs['in_slopes']
        self.out_value.value = slopes.get2d()
        self.out_value.generation_time = self.current_time
