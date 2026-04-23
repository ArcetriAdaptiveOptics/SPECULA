from specula.processing_objects.base_modalrec import BaseModalrec
from specula.data_objects.recmat import Recmat


class Modalrec(BaseModalrec):
    """
    Standard modal reconstructor processing object.
    Performs pure matrix-vector multiplication: modes = recmat @ slopes.
    """

    def __init__(self,
                 nmodes: int = None,
                 recmat: Recmat = None,
                 filtmat = None,
                 identity: bool = False,
                 ncutmodes: int = None,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if recmat is None:
            if identity:
                if nmodes is None:
                    raise ValueError('modalrec nmodes key must be set when using identity!')
                recmat = Recmat(self.xp.identity(nmodes),
                                target_device_idx=target_device_idx, precision=precision)

        if ncutmodes:
            if recmat is not None:
                recmat.reduce_size(ncutmodes)
            else:
                self.logger.warning('recmat cannot be reduced because it is null.')

        if filtmat is not None and recmat is not None:
            recmat.recmat = recmat.recmat @ filtmat
            self.logger.info('recmat updated with filtmat!')

        self.recmat = recmat
        if self.recmat is not None:
            nmodes = self.recmat.nmodes

        if nmodes is not None:
            self.modes.value = self.xp.zeros(nmodes, dtype=self.dtype)

    def trigger_code(self):
        if self.recmat is None or self.recmat.recmat is None:
            self.logger.warning("Skipping reconstruction because recmat is NULL")
            return

        # Memory pre-allocation optimization
        self.modes.value[:] = self.recmat.recmat @ self.slopes
        self.modes.generation_time = self.current_time
