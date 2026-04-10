from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.recmat import Recmat


class MVM(BaseProcessingObj):
    """
    Matrix-Vector Multiplication processing object.
    Simplified modal reconstructor for BaseValue inputs
    """

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
                                value = self.xp.zeros(self.recmat.nmodes, dtype=self.dtype),
                                target_device_idx=target_device_idx,
                                precision=precision)

        # Define inputs/outputs - solo in_vector
        self.inputs['in_vector'] = InputValue(type=BaseValue)
        self.outputs['out_vector'] = self.output

    def setup(self):
        super().setup()

        vector = self.local_inputs['in_vector']

        # Validate dimensions
        input_size = len(vector.get_value())
        expected_size = self.recmat.recmat.shape[1]
        if input_size != expected_size:
            raise ValueError(f"Input vector size mismatch: got {input_size}"
                             f", expected {expected_size}")

    def trigger_code(self):
        if self.recmat.recmat is None:
            print("WARNING: mvm skipping multiplication because recmat is NULL")
            return

        vector = self.local_inputs['in_vector']

        # Simple matrix multiplication
        self.output.value[:] = self.recmat.recmat @ vector.get_value()
        self.output.generation_time = self.current_time
