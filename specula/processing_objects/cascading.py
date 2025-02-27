from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.electric_field import ElectricField


class Cascading(BaseProcessingObj):

    def __init__(self,
                 size,       # output image size
                 target_device_idx=None,
                 precision=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.pixels = Pixels(size[0], size[1], target_device_idx=target_device_idx)
        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_pixels'] = self.pixels

    def trigger_code(self):
        # self.local_inputs dictionary has a one-to-on correspondence with self.inputs
        # and its content is automatically set up before trigger code is invoked
        # do the computation here
        self.pixels.pixels *= 0

    def post_trigger(self):
        super().post_trigger()
        self.pixels.generation_time = self.current_time



