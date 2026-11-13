import warnings

from specula.processing_objects.base_modalrec import BaseModalrec
from specula.data_objects.recmat import Recmat
from specula.lib.graph_matmul import matmul


class Modalrec(BaseModalrec):
    """
    Standard modal reconstructor processing object.
    Performs pure matrix-vector multiplication: modes = recmat @ slopes.
    The *nmodes* parameter is deprecated and ignored: the number of modes is
    set by the recmat (use *ncutmodes* to discard the last modes).
    """

    def __init__(self,
                 recmat: Recmat,
                 nmodes: int = None,
                 ncutmodes: int = None,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if nmodes is not None:
            warnings.warn('Modalrec: nmodes is deprecated and ignored, the number of modes '
                          'is set by the recmat (use ncutmodes to discard the last modes)',
                          FutureWarning, stacklevel=2)
        if ncutmodes:
            recmat.reduce_size(ncutmodes)

        self.recmat = recmat
        # In the dtype of this object, like the slopes: inside a CUDA graph capture the
        # product needs equal dtypes. A copy only if the dtype differs
        self._recmat = self.to_xp(recmat.recmat, dtype=self.dtype)
        self.modes.value = self.xp.zeros(self.recmat.nmodes, dtype=self.dtype)

    def can_accelerate(self):
        # The product uses graph_matmul.matmul(), which can be captured
        return True

    def trigger_code(self):
        self.modes.value[:] = matmul(self._recmat, self.slopes, self.xp)
        self.modes.generation_time = self.current_time
