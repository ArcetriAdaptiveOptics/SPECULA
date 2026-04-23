from specula.processing_objects.base_modalrec import BaseModalrec
from specula.base_value import BaseValue
from specula.connections import InputList, InputValue
from specula.base_processing_obj import InputDesc
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat


class ModalrecImplicitPolc(BaseModalrec):
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

        self.commands = None  # to be allocated in setup()

        self.inputs['in_commands'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_commands_list'] = InputList(type=BaseValue, optional=True)

    @classmethod
    def input_names(cls):
        # Merge base inputs with the new command inputs
        inputs = super().input_names()
        inputs.update({
            'in_commands': InputDesc(BaseValue,
                           'Current output command vector for implicit POLC (optional)'),
            'in_commands_list': InputDesc(BaseValue,
                                'List of current command vectors for implicit POLC (optional)')
        })
        return inputs

    def setup(self):
        super().setup()

        commands = self.local_inputs['in_commands']
        commands_list = self.local_inputs['in_commands_list']

        if not commands and (not commands_list or not all(commands_list)):
            raise ValueError("Either 'in_commands' or 'in_commands_list' must be given as an input")

        self.commands = self.xp.zeros(self.in_commands_size, dtype=self.dtype)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        commands = self.local_inputs['in_commands']
        commands_list = self.local_inputs['in_commands_list']

        if commands is None:
            self.commands[:] = self.xp.hstack([x.value for x in commands_list])
        else:
            if commands.value is None:
                self.commands[:] = 0.0
            else:
                self.commands[:] = commands.value

    def trigger_code(self):
        slopes = self.local_inputs['in_slopes']
        slopes_list = self.local_inputs['in_slopes_list']
        slopes_time = slopes.generation_time if slopes is not None else slopes_list[0].generation_time

        if slopes_time != self.current_time:
            return

        # Memory pre-allocation optimization with self.recmat hosting C
        self.modes.value[:] = self.recmat.recmat @ self.slopes - self.h_mat.recmat @ self.commands
        self.modes.generation_time = self.current_time
