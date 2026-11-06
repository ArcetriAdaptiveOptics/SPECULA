import numpy as np

from specula.processing_objects.base_modalrec import BasePolcModalrec
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from specula.lib.graph_matmul import matmul

# Size of the blocks of a host matrix uploaded at a time by _matmul_host()
_BLOCK_BYTES = 256 << 20


class ModalrecImplicitPolc(BasePolcModalrec):
    """
    POLC modal reconstructor processing object.
    Uses implicit Pseudo Open Loop Control (POLC) to reduce computational cost.

    The input matrices are used only to compute C = P R and H = I - C D, and
    can be much larger than them (R is 5 GB for MORFEO): when restored from
    calibration files they stay on the host, and the products are computed
    on the device of the object one block of columns at a time.
    """

    host_data_objects = ('recmat', 'projmat', 'intmat')

    def __init__(self,
                 recmat: Recmat,
                 projmat: Recmat,
                 intmat: Intmat,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # The effective reconstruction matrix becomes C = P * R
        comm_mat_arr = self._matmul_host(self.to_xp(projmat.recmat, dtype=self.dtype), recmat.recmat)
        self.recmat = Recmat(comm_mat_arr, target_device_idx=target_device_idx, precision=precision)

        # Set up the H matrix: H = I - C * D
        h_mat_temp = self._matmul_host(comm_mat_arr, intmat.intmat)
        h_mat_arr = self.xp.identity(h_mat_temp.shape[0], dtype=self.dtype) - h_mat_temp
        self.h_mat = Recmat(h_mat_arr, target_device_idx=target_device_idx, precision=precision)

        self.in_commands_size = intmat.nmodes

        nmodes = self.recmat.recmat.shape[0]
        self.modes.value = self.xp.zeros(nmodes, dtype=self.dtype)

    def _matmul_host(self, a, b):
        '''
        a @ b, with *a* on the device of the object. If *b* is on the host and the
        object on a GPU, b is uploaded in blocks of columns: out[:, j] = a @ b[:, j]
        (each element is the same full sum as in a @ b).
        '''
        if self.xp is np or not isinstance(b, np.ndarray):
            return a @ self.to_xp(b, dtype=self.dtype)
        return self._matmul_host_blocks(a, b)  # pragma: gpu

    def _matmul_host_blocks(self, a, b):  # pragma: gpu
        '''a @ b, with *a* on the GPU and *b* on the host, uploaded in blocks of columns'''
        out = self.xp.empty((a.shape[0], b.shape[1]), dtype=self.dtype)
        cols = max(1, _BLOCK_BYTES // (b.shape[0] * out.itemsize))
        for j0 in range(0, b.shape[1], cols):
            block = self.to_xp(np.ascontiguousarray(b[:, j0:j0 + cols]), dtype=self.dtype)
            out[:, j0:j0 + cols] = a @ block
        return out

    def trigger_code(self):
        if not self.slopes_updated():
            return

        # Memory pre-allocation optimization with self.recmat hosting C
        self.modes.value[:] = (matmul(self.recmat.recmat, self.slopes, self.xp)
                               - matmul(self.h_mat.recmat, self.commands, self.xp))
        self.modes.generation_time = self.current_time
