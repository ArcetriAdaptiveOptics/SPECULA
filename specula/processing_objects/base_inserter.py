from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue


class BaseInserter(BaseProcessingObj):
    """
    Inserts a small vector into a larger vector, distributing n slices
    of the small vector into n parts of the large vector.

    Parameters
    ----------
    output_size : int
        Size of the large output vector.
    slice_pairs : list of (src, dest) tuples
        Each tuple contains two elements, each of which can be:
        - a list [start, stop] or [start, stop, step] to define a slice
        - a list of indices
        src selects elements from the input (small) vector,
        dest selects where to place them in the output (large) vector.
    """

    def __init__(self,
                 output_size,
                 slice_pairs,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if not slice_pairs:
            raise ValueError("'slice_pairs' must be a non-empty list of (src, dest) pairs")

        self._src_selectors = []
        self._dest_selectors = []
        for src, dest in slice_pairs:
            self._src_selectors.append(self._make_selector(src))
            self._dest_selectors.append(self._make_selector(dest))

        out_array = self.xp.zeros(output_size, dtype=self.dtype)
        self.out_value = BaseValue(value=out_array,
                                   target_device_idx=target_device_idx,
                                   precision=precision)
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.outputs['out_value'] = self.out_value

    @staticmethod
    def _make_selector(spec):
        """
        Convert a spec to a slice or an index array.
        - A tuple of 2 or 3 ints is interpreted as slice args: (start, stop[, step])
        - A list is always interpreted as an explicit index list.
        """
        if isinstance(spec, tuple) and 2 <= len(spec) <= 3 and all(isinstance(x, int) for x in spec):
            return slice(*spec)
        return spec

    def trigger_code(self):
        small = self.local_inputs['in_value'].value
        for src_sel, dest_sel in zip(self._src_selectors, self._dest_selectors):
            self.out_value.value[dest_sel] = small[src_sel]
        self.out_value.generation_time = self.current_time
