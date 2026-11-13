from iteras.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from iteras.base_value import BaseValue
from iteras.connections import InputValue

class BaseSlicer(BaseProcessingObj):
    """
    Base Slicer processing object.
    Extracts a subset of values from a BaseValue based on an index, list of indices, or a slice.
    """
    def __init__(self,
                 indices=None,   # int, list
                 slice_args=None,  # list for slice(start, stop, step)
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Validate that indices and slice_args are not both set
        if indices is not None and slice_args is not None:
            raise ValueError("Cannot specify both 'indices' and 'slice_args'")

        # Device copy of the indices, made once: indexing with a host list copies
        # it to the device at each step, which a CUDA graph capture does not allow.
        # An int index gives a one-element output, as allocated below.
        if indices is not None:
            indices = self.xp.asarray([indices] if isinstance(indices, int) else indices,
                                      dtype=self.xp.int64)
        self.indices = indices
        # Use slice arguments to create a slice object
        if slice_args is not None:
            self.slice_obj = slice(*slice_args)
        else:
            self.slice_obj = None
        # ----------------------------------
        # Initialize output BaseValue
        n_elements = None
        if self.indices is not None:
            n_elements = len(self.indices)
        elif self.slice_obj is not None:
            start = self.slice_obj.start
            stop = self.slice_obj.stop
            step = self.slice_obj.step if self.slice_obj.step is not None else 1
            if start is not None and stop is not None:
                n_elements = (stop - start + step - 1) // step
        value = self.xp.zeros(n_elements, dtype=self.dtype) \
            if n_elements is not None else self.xp.array([], dtype=self.dtype)
        # ----------------------------------
        self.out_value = BaseValue(value=value,
                                   target_device_idx=target_device_idx, precision=precision)
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.outputs['out_value'] = self.out_value

    @classmethod
    def input_names(cls):
        return {'in_value': InputDesc(BaseValue, 'Input vector to slice')}

    @classmethod
    def output_names(cls):
        return {'out_value': OutputDesc(BaseValue, 'Output sliced vector')}

    def can_accelerate(self):
        # The indices are on the device, and the output is written in place
        return True

    def trigger_code(self):
        value = self.local_inputs['in_value'].value
        if self.indices is not None:
            sliced = value[self.indices]
        elif self.slice_obj is not None:
            sliced = value[self.slice_obj]
        else:
            sliced = value
        out = self.out_value.value
        if out.shape != sliced.shape or out.dtype != sliced.dtype:
            # First trigger only: output allocated with the shape and type of the input.
            # Afterwards it is written in place, so that the objects reading it (and
            # their CUDA graphs, also when this object is not accelerated) keep the same array
            self.out_value.value = sliced.copy()
        else:
            out[...] = sliced
        self.out_value.generation_time = self.current_time
