import numpy as np
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList


class LinearCombination(BaseProcessingObj):
    def __init__(self,
                 target_device_idx: int = None, 
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)       

        self.inputs['in_vectors_list'] = InputList(type=BaseValue)
        self.out_vector = BaseValue()
        self.outputs['out_vector'] = self.out_vector

    def trigger_code(self):
        in_vectors = self.local_inputs['in_vectors_list']
        lgs = in_vectors[0].value
        focus = in_vectors[1].value
        lift = in_vectors[2].value
        ngs = in_vectors[3].value

        focus *= 0
        lift *= 0
        ngs *= 0

        self.out_vector.value = np.concatenate([lgs, focus, lift, ngs])
        self.out_vector.generation_time = self.current_time