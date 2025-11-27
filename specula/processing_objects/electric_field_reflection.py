from specula.connections import InputValue

from specula.data_objects.electric_field import ElectricField
from specula.base_processing_obj import BaseProcessingObj

class ElectricFieldReflection(BaseProcessingObj):
    """
    Reflects an input electric field (changes the sign of the phase).
    """
    def __init__(self, target_device_idx: int=None, precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self._out_ef = ElectricField(
            dimx=1, dimy=1, pixel_pitch=1, S0=1,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )
        self.outputs['out_ef'] = self._out_ef

    def setup(self):
        super().setup()
        in_ef = self.local_inputs['in_ef']
        self._out_ef.resize(
            dimx=in_ef.A.shape[0],
            dimy=in_ef.A.shape[1],
            pitch=in_ef.pixel_pitch,
        )

    def trigger(self):
        in_ef = self.local_inputs['in_ef']
        # Copy amplitude
        self._out_ef.A[:] = in_ef.A
        # Invert phase sign
        self._out_ef.phaseInNm[:] = -in_ef.phaseInNm
        # Copy S0 and generation time
        self._out_ef.S0 = in_ef.S0
        self._out_ef.generation_time = self.current_time