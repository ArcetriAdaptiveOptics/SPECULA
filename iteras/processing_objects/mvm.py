from iteras.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from iteras.base_value import BaseValue
from iteras.connections import InputValue
from iteras.data_objects.matrix import Matrix
from iteras.lib.graph_matmul import matmul


class MVM(BaseProcessingObj):
    """
    Matrix-Vector Multiplication processing object.
    Multiplies the input vector by a matrix: ``out_vector = matrix @ in_vector``.
    """

    def __init__(self,
                 matrix: Matrix = None,
                 recmat: Matrix = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        """
        Parameters
        ----------
        matrix : Matrix
            The matrix, of shape (output size, input size).
        recmat : Matrix
            Previous name of *matrix*, kept for compatibility (``recmat_object``
            in the parameter files).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if matrix is not None and recmat is not None:
            raise ValueError('Only one of matrix and recmat can be given')
        if matrix is None:
            matrix = recmat
        if matrix is None:
            raise ValueError('matrix must be provided!')

        self.matrix = matrix
        # In the dtype of this object, like the input vector at each step: the product
        # is then the same with and without a CUDA graph (see graph_matmul.matmul()).
        # A copy only if the dtype differs: later changes to the matrix are then not seen
        self._matrix = self.to_xp(matrix.matrix, dtype=self.dtype)

        # Create outputs
        self.output = BaseValue('output from matrix-vector multiplication',
                                value = self.xp.zeros(self.matrix.nrows, dtype=self.dtype),
                                target_device_idx=target_device_idx,
                                precision=precision)

        # Define inputs/outputs - solo in_vector
        self.inputs['in_vector'] = InputValue(type=BaseValue)
        self.outputs['out_vector'] = self.output

    @classmethod
    def input_names(cls):
        return {'in_vector': InputDesc(BaseValue, 'Input vector to be multiplied by the matrix')}

    @classmethod
    def output_names(cls):
        return {'out_vector': OutputDesc(BaseValue, 'Output vector after matrix-vector multiplication')}

    def setup(self):
        super().setup()

        vector = self.local_inputs['in_vector']

        # Validate dimensions
        input_size = len(vector.get_value())
        expected_size = self.matrix.ncols
        if input_size != expected_size:
            raise ValueError(f"Input vector size mismatch: got {input_size}"
                             f", expected {expected_size}")

    def can_accelerate(self):
        # The product uses graph_matmul.matmul(), which can be captured
        return True

    def trigger_code(self):
        vector = self.local_inputs['in_vector']

        # Simple matrix multiplication
        self.output.value[:] = matmul(self._matrix, vector.get_value().astype(self.dtype, copy=False), self.xp)
        self.output.generation_time = self.current_time
