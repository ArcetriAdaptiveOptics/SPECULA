from specula.base_value import BaseValue
from specula.base_processing_obj import BaseProcessingObj


class BaseGenerator(BaseProcessingObj):
    """
    Base class for function generators.
    
    All specific generators inherit from this class and implement trigger_code().
    """
    def __init__(self,
                 output_size: int = 1,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Create output
        self.output = BaseValue(
            target_device_idx=target_device_idx,
            precision=precision,
            value=self.xp.zeros(output_size, dtype=self.dtype)
        )
        self.outputs['output'] = self.output

        # Time tracking
        self.iter_counter = 0
        self.current_time_gpu = self.xp.zeros(1, dtype=self.dtype)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.current_time_gpu[:] = self.current_time_seconds

    def trigger_code(self):
        """Implement signal generation logic in subclasses"""
        raise NotImplementedError("Subclasses must implement the trigger_code method to generate signals.")

    def post_trigger(self):
        super().post_trigger()
        self.output.generation_time = self.current_time
        self.iter_counter += 1

    def _validate_array_sizes(self, *arrays, names=None):
        """Utility to validate that arrays have consistent sizes"""
        from specula.lib.utils import is_scalar

        if names is None:
            names = [f"param_{i}" for i in range(len(arrays))]
        if len(names) != len(arrays):
            raise ValueError(f'names list has length {len(names)} while array list has length {len(arrays)}')

        vector_lengths = [arr.shape[0] for arr in arrays if not is_scalar(arr, self.xp)]

        if len(vector_lengths) > 0:
            unique_lengths = set(vector_lengths)
            if len(unique_lengths) > 1:
                details = [f"{name}={arr.shape[0]}" for arr, name in zip(arrays, names)
                          if not is_scalar(arr, self.xp)]
                raise ValueError(
                    f"Shape mismatch: parameter lengths are {details} (must all be equal if not scalar)"
                )
            return unique_lengths.pop()
        return 1
