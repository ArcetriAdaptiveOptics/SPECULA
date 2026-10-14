from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputList
from specula.data_objects.intensity import Intensity


class IntensitySum(BaseProcessingObj):
    """
    Sums a list of intensities pixel by pixel.

    All inputs must have the same shape; the output shape is taken from the first input.
    """

    def __init__(self,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        # Placeholder size, reallocated in setup() from the first input
        self.out_i = Intensity(1, 1, target_device_idx=self.target_device_idx, precision=precision)
        self.inputs['in_i_list'] = InputList(type=Intensity)
        self.outputs['out_i'] = self.out_i

    @classmethod
    def input_names(cls):
        return {'in_i_list': InputDesc(Intensity, 'List of input intensities to sum')}

    @classmethod
    def output_names(cls):
        return {'out_i': OutputDesc(Intensity, 'Sum of the input intensities')}

    def setup(self):
        super().setup()
        in_i_list = self.local_inputs['in_i_list']
        if in_i_list is None or len(in_i_list) == 0:
            raise ValueError('IntensitySum requires at least one intensity in in_i_list')
        shape = in_i_list[0].i.shape
        for k, intensity in enumerate(in_i_list[1:], start=1):
            if intensity.i.shape != shape:
                raise ValueError(f'Input intensity index {k} shape {intensity.i.shape}'
                                 f' does not match index 0 shape {shape}')
        self.out_i.i = self.xp.zeros(shape, dtype=self.dtype)

    def trigger_code(self):
        in_i_list = self.local_inputs['in_i_list']
        self.out_i.i[:] = in_i_list[0].i
        for intensity in in_i_list[1:]:
            self.out_i.i += intensity.i

    def post_trigger(self):
        super().post_trigger()
        self.out_i.generation_time = self.current_time
