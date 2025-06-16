
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue


class AVC(BaseProcessingObj):
    def __init__(self,
                 target_device_idx: int = None, 
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)       

        self._out_comm = BaseValue()
        self.inputs['in_measurement'] = InputValue(type=BaseValue)
        self.outputs['out_comm'] = self._out_comm