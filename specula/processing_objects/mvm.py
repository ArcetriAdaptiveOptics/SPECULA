from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.recmat import Recmat


class MVM(BaseProcessingObj):
    '''Matrix-Vector Multiplication - Simplified modal reconstructor for BaseValue inputs'''

    def __init__(self,
                 recmat: Recmat,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if recmat is None:
            raise ValueError('recmat must be provided!')

        self.recmat = recmat

        # Create outputs
        self.output = BaseValue('output from matrix-vector multiplication',
                               target_device_idx=target_device_idx,
                               precision=precision)

        # Define inputs/outputs - solo in_vector
        self.inputs['in_vector'] = InputValue(type=BaseValue)
        self.outputs['out_vector'] = self.output

        # Pre-allocate output
        nmodes_out = self.recmat.nmodes
        self.output.value = self.xp.zeros(nmodes_out, dtype=self.dtype)

    def setup(self):
        super().setup()

        vector = self.local_inputs['in_vector']

        if not vector:
            raise ValueError("'in_vector' must be provided as input")

        # Prepare input vector for processing
        self.input_vector = self.to_xp(vector.value.copy())

        # Validate dimensions
        input_size = len(self.input_vector)
        expected_size = self.recmat.recmat.shape[1]
        if input_size != expected_size:
            raise ValueError(f"Input vector size mismatch: got {input_size}"
                             f", expected {expected_size}")

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        vector = self.local_inputs['in_vector']

        # Update input vector
        self.input_vector[:] = vector.value

    def trigger_code(self):
        if self.recmat.recmat is None:
            print("WARNING: mvm skipping multiplication because recmat is NULL")
            return

        # Simple matrix multiplication
        output_data = self.recmat.recmat @ self.input_vector

        # Store result
        self.output.value = output_data
        self.output.generation_time = self.current_time