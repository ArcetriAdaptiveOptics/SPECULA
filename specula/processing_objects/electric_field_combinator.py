from specula.connections import InputValue

from specula.data_objects.electric_field import ElectricField
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.simul_params import SimulParams

class ElectricFieldCombinator(BaseProcessingObj):
    """
    Combines two input electric fields.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 target_device_idx: int=None,
                 precision: int=None
                 ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.pixel_pitch = self.simul_params.pixel_pitch

        self.inputs['in_ef1'] = InputValue(type=ElectricField)
        self.inputs['in_ef2'] = InputValue(type=ElectricField)

    def setup(self):
        super().setup()

        in_ef1 = self.local_inputs['in_ef1']
        in_ef2 = self.local_inputs['in_ef2']
        if in_ef1.A.shape != in_ef2.A.shape:
            raise ValueError(f"Input electric field no. 1 shape {in_ef1.A.shape} does not match electric field no. 2 shape {in_ef2.A.shape}")

        S0 = in_ef1.S0+in_ef2.S0
        print('S0:', in_ef1.S0, in_ef2.S0, S0)

        self._out_ef = ElectricField(
            dimx=in_ef1.A.shape[0],
            dimy=in_ef1.A.shape[1],
            pixel_pitch=self.pixel_pitch,
            S0=S0,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )

        self.outputs['out_ef'] = self._out_ef

    def trigger_code(self):
        super().trigger_code()

        # Get the input electric fields
        in_ef1 = self.local_inputs['in_ef1']
        in_ef2 = self.local_inputs['in_ef2']

        # Combine the electric fields
        # Add phases
        self._out_ef.phaseInNm = in_ef1.phaseInNm + in_ef2.phaseInNm

        # Multiply amplitudes
        self._out_ef.A = in_ef1.A * in_ef2.A

        # Set the generation time to the current time
        self._out_ef.generation_time = self.current_time
