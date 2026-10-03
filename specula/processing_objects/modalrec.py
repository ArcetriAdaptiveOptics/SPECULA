import warnings

from specula.processing_objects.base_modalrec import BaseModalrec
from specula.data_objects.recmat import Recmat


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
        self.modes.value = self.xp.zeros(self.recmat.nmodes, dtype=self.dtype)

    def trigger_code(self):
        self.modes.value[:] = self.recmat.recmat @ self.slopes
        self.modes.generation_time = self.current_time
