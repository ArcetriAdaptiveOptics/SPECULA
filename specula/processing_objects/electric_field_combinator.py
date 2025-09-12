from specula.connections import InputValue

from specula.data_objects.electric_field import ElectricField
from specula.base_processing_obj import BaseProcessingObj


class ElectricFieldCombinator(BaseProcessingObj):
    """
    Combines two input electric fields.
    """
    def __init__(self,
                 target_device_idx: int=None,
                 precision: int=None
                 ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.inputs['in_ef1'] = InputValue(type=ElectricField)
        self.inputs['in_ef2'] = InputValue(type=ElectricField)

        self._out_ef = ElectricField(
                dimx=1,  # Will be replaced in setup()
                dimy=1,
                pixel_pitch=1,
                S0=1,
                target_device_idx=self.target_device_idx,
                precision=self.precision
            )

        self.outputs['out_ef'] = self._out_ef

    def setup(self):
        super().setup()
        
        # Get the input electric fields to check their shapes and initialize the output electric field with correct dimensions
        in_ef1 = self.local_inputs['in_ef1']
        in_ef2 = self.local_inputs['in_ef2']

        if in_ef1.A.shape != in_ef2.A.shape:
            raise ValueError(f"Input electric field no. 1 shape {in_ef1.A.shape} does not match electric field no. 2 shape {in_ef2.A.shape}")

        self._out_ef.resize(
            dimx=in_ef1.A.shape[0],
            dimy=in_ef1.A.shape[1],
            pitch=in_ef1.pixel_pitch,
        )

    def trigger(self):
        # Get the input electric fields
        in_ef1 = self.local_inputs['in_ef1']
        in_ef2 = self.local_inputs['in_ef2']

        # Combine the electric fields
        # Add phases
        self._out_ef.phaseInNm[:] = in_ef1.phaseInNm + in_ef2.phaseInNm

        # Multiply amplitudes
        self._out_ef.A[:] = in_ef1.A * in_ef2.A

        # Combine S0 values
        self._out_ef.S0 = in_ef1.S0 + in_ef2.S0

        # Set the generation time to the current time
        self._out_ef.generation_time = self.current_time
