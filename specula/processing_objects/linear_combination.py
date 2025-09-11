import numpy as np
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList


class LinearCombination(BaseProcessingObj):
    def __init__(self,
                 no_focus: bool = False,
                 no_lift: bool = False,
                 plate_scale_idx: int = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.no_focus = no_focus
        self.no_lift = no_lift
        self.plate_scale_idx = plate_scale_idx

        self.inputs['in_vectors_list'] = InputList(type=BaseValue)
        self.out_vector = BaseValue(target_device_idx=self.target_device_idx)
        self.outputs['out_vector'] = self.out_vector

    def trigger_code(self):
        in_vectors = self.local_inputs['in_vectors_list']
        idx = 0
        lgs = in_vectors[idx].value
        idx += 1
        if not self.no_focus:
            focus = in_vectors[idx].value
            idx += 1
        if not self.no_lift:
            lift = in_vectors[idx].value
            idx += 1
        ngs = in_vectors[idx].value

        # TIP / TILT
        lgs[0:2] = ngs[0:2]
        if not self.no_focus:
            lgs[2] = focus[0]
        if self.plate_scale_idx is not None:
            lgs[self.plate_scale_idx:self.plate_scale_idx+3] = ngs[2:]

        self.out_vector.value *= 0.0
        self.out_vector.value[:len(lgs)] = lgs
        self.out_vector.generation_time = self.current_time

    def setup(self):
        super().setup()

        in_vectors = self.local_inputs['in_vectors_list']

        idx = 0
        lgs = in_vectors[idx].value
        idx += 1
        if not self.no_focus:
            idx += 1
        if not self.no_lift:
            lift = in_vectors[idx].value
            idx += 1
        else:
            lift = self.xp.zeros(0,dtype=self.dtype)

        # Note: NGS and FOCUS override LGS values
        self.out_vector.value = np.concatenate([lgs, lift]) * 0.0
