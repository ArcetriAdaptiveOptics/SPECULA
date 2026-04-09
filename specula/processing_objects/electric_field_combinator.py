from specula.connections import InputValue, InputList

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

        self.inputs['in_ef1'] = InputValue(type=ElectricField, optional=True)
        self.inputs['in_ef2'] = InputValue(type=ElectricField, optional=True)
        self.inputs['in_ef_list'] = InputList(type=ElectricField, optional=True)

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

        # Safely fetch inputs without assuming they are connected
        in_ef_list = self.local_inputs.get('in_ef_list')
        in_ef1 = self.local_inputs.get('in_ef1')
        in_ef2 = self.local_inputs.get('in_ef2')

        # Check which input method is being used
        has_list = in_ef_list is not None and len(in_ef_list) > 0
        has_legacy = in_ef1 is not None and in_ef2 is not None

        # Validation: Ensure at least one valid input combination is provided, but not both
        if not has_list and not has_legacy or (has_list and has_legacy):
            raise ValueError(
                "ElectricFieldCombinator requires either 'in_ef_list' to be populated, "
                "or BOTH 'in_ef1' and 'in_ef2' to be connected."
            )

        if has_list:
            first_ef = in_ef_list[0]
            # Verify that all provided fields have matching shapes
            for i, ef in enumerate(in_ef_list[1:]):
                if first_ef.A.shape != ef.A.shape:
                    raise ValueError(f"Input electric field list index {i+1} shape {ef.A.shape} does not match index 0 shape {first_ef.A.shape}")
                if first_ef.pixel_pitch != ef.pixel_pitch:
                    raise ValueError(f"Input electric field list index {i+1} pixel pitch {ef.pixel_pitch} does not match index 0 shape {first_ef.pixel_pitch}")
        else:
            # Same check for pair configuration
            if in_ef1.A.shape != in_ef2.A.shape:
                raise ValueError(f"Input electric field no. 1 shape {in_ef1.A.shape} does not match electric field no. 2 shape {in_ef2.A.shape}")
            if in_ef1.pixel_pitch != in_ef2.pixel_pitch:
                raise ValueError(f"Input electric field no. 1 pixel_pitch {in_ef1.pixel_pitch} does not match electric field no. 2 pixel_pitch {in_ef2.pixel_pitch}")
            first_ef = in_ef1

        self._out_ef.resize(
            dimx=first_ef.A.shape[0],
            dimy=first_ef.A.shape[1],
            pitch=first_ef.pixel_pitch,
        )
        self.has_list = has_list

    def trigger(self):
        if self.has_list:
            in_ef_list = self.local_inputs.get('in_ef_list')
        else:
            in_ef1 = self.local_inputs['in_ef1']
            in_ef2 = self.local_inputs['in_ef2']
            in_ef_list = [in_ef1, in_ef2]
            
        # Initialize the output arrays/values using the first field
        self._out_ef.phaseInNm[:] = in_ef_list[0].phaseInNm
        self._out_ef.A[:] = in_ef_list[0].A
        self._out_ef.S0 = in_ef_list[0].S0
        
        # Accumulate values from the rest of the list
        for in_ef in in_ef_list[1:]:
            self._out_ef.phaseInNm += in_ef.phaseInNm
            self._out_ef.A *= in_ef.A
            self._out_ef.S0 += in_ef.S0
            
        self._out_ef.generation_time = self.current_time
