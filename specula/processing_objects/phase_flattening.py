from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.simul_params import SimulParams

class PhaseFlattening(BaseProcessingObj):
    """
    Removes the mean phase from an electric field.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.pixel_pitch = self.simul_params.pixel_pitch

        self.inputs['in_ef'] = InputValue(type=ElectricField)

        self._out_ef = ElectricField(
            dimx=1,  # Will be replaced in setup()
            dimy=1,
            pixel_pitch=self.pixel_pitch,
            S0=1,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )

        self.outputs['out_ef'] = self._out_ef

    def setup(self):
        super().setup()
        
        # Get the input electric field to initialize output with correct dimensions
        in_ef = self.local_inputs['in_ef']

        self._out_ef.resize(
            dimx=in_ef.A.shape[0],
            dimy=in_ef.A.shape[1],
        )

    def trigger_code(self):
        # Get the input electric field
        in_ef = self.local_inputs['in_ef']

        # Copy amplitude (unchanged)
        self._out_ef.A[:] = in_ef.A

        # Copy S0 (unchanged)
        self._out_ef.S0 = in_ef.S0

        # Process phase: remove mean only where amplitude > 0
        phase = in_ef.phaseInNm.copy()
        valid_mask = in_ef.A > 0

        if self.xp.any(valid_mask):
            # Calculate mean phase only on valid pixels
            mean_phase = self.xp.mean(phase[valid_mask])

            # Remove mean phase from all pixels
            phase[valid_mask] -= mean_phase

        self._out_ef.phaseInNm[:] = phase

        # Set the generation time to the current time
        self._out_ef.generation_time = self.current_time
