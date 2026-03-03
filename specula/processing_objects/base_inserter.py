from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue


class BaseInserter(BaseProcessingObj):
    """
    Inserts a small vector into a larger vector.
    Mirrors the BaseSlicer interface, but for insertion.

    Parameters
    ----------
    output_size : int
        Size of the large output vector.
    indices : list of [src_indices, dest_indices] pairs, optional
        Each pair defines explicit indices in the input and output vectors.
        Example: indices=[[0,1,2], [1,3,5]] inserts src[0,1,2] into dest[1,3,5].
    slice_args : list of [src_slice_args, dest_slice_args] pairs, optional
        Each pair defines a slice in the input and output vectors.
        Example: slice_args=[[0,3], [2,5]] inserts src[0:3] into dest[2:5].
        Multiple pairs: slice_args=[[[0,2],[0,2]], [[2,4],[5,7]]]
    """

    def __init__(self,
                 output_size,
                 indices=None,
                 slice_args=None,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if indices is not None and slice_args is not None:
            raise ValueError("Cannot specify both 'indices' and 'slice_args'")
        if indices is None and slice_args is None:
            raise ValueError("One of 'indices' or 'slice_args' must be specified")

        self._src_selectors = []
        self._dest_selectors = []

        if indices is not None:
            # indices is either [src, dest] or a list of [src, dest] pairs
            if not isinstance(indices[0][0], (list, tuple)):
                indices = [indices]
            for src, dest in indices:
                self._src_selectors.append(src)
                self._dest_selectors.append(dest)
        else:
            # slice_args is either [src, dest] or a list of [src, dest] pairs
            if not isinstance(slice_args[0][0], (list, tuple)):
                slice_args = [slice_args]
            for src, dest in slice_args:
                self._src_selectors.append(slice(*src))
                self._dest_selectors.append(slice(*dest))

        out_array = self.xp.zeros(output_size, dtype=self.dtype)
        self.out_value = BaseValue(value=out_array,
                                   target_device_idx=target_device_idx,
                                   precision=precision)
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.outputs['out_value'] = self.out_value

    def trigger_code(self):
        small = self.local_inputs['in_value'].value
        for src_sel, dest_sel in zip(self._src_selectors, self._dest_selectors):
            self.out_value.value[dest_sel] = small[src_sel]
        self.out_value.generation_time = self.current_time
