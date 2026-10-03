from specula.processing_objects.base_modalrec import BasePolcModalrec
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat


class ModalrecImplicitPolc(BasePolcModalrec):
    """
    POLC modal reconstructor processing object.
    Uses implicit Pseudo Open Loop Control (POLC) to reduce computational cost.
    """

    def __init__(self,
                 recmat: Recmat,
                 projmat: Recmat,
                 intmat: Intmat,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # The effective reconstruction matrix becomes C = P * R
        comm_mat_arr = projmat.recmat @ recmat.recmat
        self.recmat = Recmat(comm_mat_arr, target_device_idx=target_device_idx, precision=precision)

        # Set up the H matrix: H = I - C * D
        h_mat_temp = comm_mat_arr @ intmat.intmat
        h_mat_arr = self.xp.identity(h_mat_temp.shape[0], dtype=self.dtype) - h_mat_temp
        self.h_mat = Recmat(h_mat_arr, target_device_idx=target_device_idx, precision=precision)

        self.in_commands_size = intmat.nmodes

        nmodes = self.recmat.recmat.shape[0]
        self.modes.value = self.xp.zeros(nmodes, dtype=self.dtype)

    def trigger_code(self):
        if not self.slopes_updated():
            return

        # Memory pre-allocation optimization with self.recmat hosting C
        self.modes.value[:] = self.recmat.recmat @ self.slopes - self.h_mat.recmat @ self.commands
        self.modes.generation_time = self.current_time
