from specula.processing_objects.base_modalrec import BaseModalrec
from specula.base_value import BaseValue
from specula.connections import InputList, InputValue
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat


class ModalrecExplicitPolc(BaseModalrec):
    """
    Explicit Pseudo Open Loop Control (POLC) modal reconstructor processing object.
    
    This class explicitly reconstructs the slopes by summing the measured slopes
    and the estimated commands contribution (via interaction matrix), 
    and then subtracts the applied commands from the projected modes.
    """

    def __init__(self,
                 recmat: Recmat,
                 projmat: Recmat = None,
                 intmat: Intmat = None,
                 nSlopesToBeDiscarded: int = None,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.recmat = recmat
        self.projmat = projmat
        self.intmat = intmat
        if self.intmat is not None:
            if nSlopesToBeDiscarded:
                self.intmat.reduce_slopes(nSlopesToBeDiscarded)
            in_commands_size = self.intmat.nmodes
        else:
            in_commands_size = self.recmat.nmodes

        self.in_commands_size = in_commands_size

        # Properly initialize nmodes based on projmat presence
        # If no projection matrix is provided, the output size matches the reconstructor
        if self.projmat is not None:
            n_out_modes = self.projmat.nmodes
        else:
            n_out_modes = self.recmat.nmodes

        # Re-initialize output modes with the correct size
        self.modes.value = self.xp.zeros(n_out_modes, dtype=self.dtype)

        # Pseudo open-loop modes always match the size of the reconstruction space
        self.pseudo_ol_modes = BaseValue('output POL modes from explicit POLC',
                                         target_device_idx=target_device_idx,
                                         precision=precision)
        self.pseudo_ol_modes.value = self.xp.zeros(self.recmat.nmodes, dtype=self.dtype)

        self.commands = None  # to be allocated in setup()

        self.inputs['in_commands'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_commands_list'] = InputList(type=BaseValue, optional=True)

        self.outputs['out_pseudo_ol_modes'] = self.pseudo_ol_modes

    @classmethod
    def input_names(cls):
        # Extend base inputs with command ports
        inputs = super().input_names()
        inputs.update({
            'in_commands': InputDesc(BaseValue,
                           'Current output command vector for explicit POLC (optional)'),
            'in_commands_list': InputDesc(BaseValue,
                                'List of current command vectors for explicit POLC (optional)')
        })
        return inputs

    @classmethod
    def output_names(cls):
        # Extend base outputs with pseudo open-loop modes
        outputs = super().output_names()
        outputs.update({
            'out_pseudo_ol_modes': OutputDesc(BaseValue, 'Pseudo open-loop modal estimate')
        })
        return outputs

    def setup(self):
        super().setup()

        # Dimension validation
        if self.intmat is not None and self.intmat.intmat is not None:
            expected_slopes_size = self.intmat.nslopes
            if expected_slopes_size != len(self.slopes):
                raise ValueError(f"Dimension mismatch in POLC mode: "
                                 f"intmat @ commands will produce {expected_slopes_size} slopes, "
                                 f"but input slopes has size {len(self.slopes)}")

        commands = self.local_inputs['in_commands']
        commands_list = self.local_inputs['in_commands_list']

        if not commands and (not commands_list or not all(commands_list)):
            raise ValueError("Either 'in_commands' or 'in_commands_list' must be given as an input")

        self.commands = self.xp.zeros(self.in_commands_size, dtype=self.dtype)


    def prepare_trigger(self, t):
        # Handle slopes via base class
        super().prepare_trigger(t)

        # Handle commands locally
        commands = self.local_inputs['in_commands']
        commands_list = self.local_inputs['in_commands_list']

        if commands is None:
            # Handle list of commands (e.g. from multiple DMs)
            self.commands[:] = self.xp.hstack([x.value for x in commands_list])
        else:
            if commands.value is None:
                self.commands[:] = 0.0
            else:
                self.commands[:] = commands.value

    def trigger_code(self):
        # Check refresh based on slopes (standard POLC logic)
        slopes = self.local_inputs['in_slopes']
        slopes_list = self.local_inputs['in_slopes_list']
        slopes_time = slopes.generation_time if slopes is not None else slopes_list[0].generation_time

        if slopes_time != self.current_time:
            return

        # (1) Compute pseudo open loop modes
        if self.intmat is not None:
            comm_slopes = self.intmat.intmat @ self.commands
            self.pseudo_ol_modes.value[:] = self.recmat.recmat @ (self.slopes + comm_slopes)
        else:
            # If no interaction matrix is provided, we assume the pseudo open-loop modes
            # are just the reconstruction of the measured slopes
            self.pseudo_ol_modes.value[:] = self.recmat.recmat @ self.slopes

        self.pseudo_ol_modes.generation_time = self.current_time

        # (2) Project to output modes
        if self.projmat is None:
            # Avoid aliasing by creating a copy of the array
            # This ensures that step (3) doesn't modify pseudo_ol_modes.value in-place
            output_modes = self.pseudo_ol_modes.value.copy()
        else:
            output_modes = self.projmat.recmat @ self.pseudo_ol_modes.value

        # (3) Remove the effect of the commands: m_out = P * m_pol - c
        output_modes -= self.commands

        # Final update with memory-safe assignment
        self.modes.value[:] = output_modes
        self.modes.generation_time = self.current_time
