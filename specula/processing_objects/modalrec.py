from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList, InputValue
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from specula.data_objects.slopes import Slopes


class Modalrec(BaseProcessingObj):
    '''Modal reconstructor'''

    def __init__(self,
                 nmodes: int=None,      # TODO =0,
                 recmat: Recmat=None,
                 projmat: Recmat=None,
                 intmat: Intmat=None,
                 polc: bool=False,
                 in_commands_size: int=None,
                 filtmat = None,
                 identity: bool=False,
                 ncutmodes: int=None,
                 nSlopesToBeDiscarded: int=None,
                 dmNumber: int=0,
                 noProj: bool=False,
                 input_modes_index: list=None,
                 input_modes_slice: list=None,
                 output_slice: list=None,
                 target_device_idx: int=None,
                 precision: int=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if polc:
            if identity:
                raise ValueError('identity cannot be set with POLC.')
            if ncutmodes is not None:
                raise ValueError('ncutmodes cannot be set with POLC.')
        else:
            if recmat is None:
                if identity:
#                    if nmodes<=0:  # TODO new code to be tested
                    if nmodes is None:
                        raise ValueError('modalrec nmodes key must be set!')
                    recmat = Recmat(self.xp.identity(nmodes),
                                    target_device_idx=target_device_idx, precision=precision)
                elif intmat:
                    if nmodes:
                        nmodesintmat = intmat.size[0]
                        intmat.reduce_size(nmodesintmat - nmodes)
                    if nSlopesToBeDiscarded:
                        intmat.reduce_slopes(nSlopesToBeDiscarded)
                    recmat = Recmat(intmat.intmat,
                                    target_device_idx=target_device_idx, precision=precision)

            if ncutmodes:
                if recmat is not None:
                    recmat.reduce_size(ncutmodes)
                else:
                    print('recmat cannot be reduced because it is null.')


        if recmat is not None:
            if projmat is None and recmat.proj_list and not noProj:
                if dmNumber is not None:
                    if dmNumber <= 0:
                        raise ValueError('dmNumber must be > 0')
                    projmat = Recmat(recmat.proj_list[dmNumber - 1],
                                     target_device_idx=target_device_idx, precision=precision)
                else:
                    raise ValueError('dmNumber (>0) must be defined if projmat_tag is not defined!')

        if filtmat is not None and recmat is not None:
            recmat.recmat = recmat.recmat @ filtmat
            print('recmat updated with filmat!')

        if polc:
            if not intmat:
                raise ValueError("Intmat object not valid")

        self.recmat = recmat
        self.projmat = projmat
        self.intmat = intmat
        self.polc = polc
        self.in_commands_size = in_commands_size
        self.input_modes_index = input_modes_index

        if output_slice is not None:
            self.output_slice = slice(*output_slice)
            start = self.output_slice.start if self.output_slice.start is not None else 0
            stop = self.output_slice.stop if self.output_slice.stop is not None else self.recmat.recmat.shape[0]
            step = abs(self.output_slice.step) if self.output_slice.step is not None else 1
            nmodes = (stop - start) // step
        else:
            self.output_slice = slice(None, None, None)
            if polc:
                nmodes = self.projmat.nmodes
            else:
                nmodes = self.recmat.nmodes

        if input_modes_slice is not None:
            self.input_modes_slice = slice(*input_modes_slice)
        else:
            self.input_modes_slice = slice(None, None, None)

        self.modes = BaseValue('output modes from modal reconstructor', target_device_idx=target_device_idx)
        self.pseudo_ol_modes = BaseValue('output POL modes from modal reconstructor', target_device_idx=target_device_idx)        

        self.inputs['in_slopes'] = InputValue(type=Slopes, optional=True)
        self.inputs['in_slopes_list'] = InputList(type=Slopes, optional=True)
        self.outputs['out_modes'] = self.modes
        self.outputs['out_pseudo_ol_modes'] = self.pseudo_ol_modes
        
        # TODO static allocation but polc not supported (should use projmat)
        self.modes.value = self.xp.zeros(nmodes, dtype=self.dtype)
        self.pseudo_ol_modes.value = self.xp.zeros(nmodes, dtype=self.dtype)
        
        if self.polc:
            self.out_comm = BaseValue('output commands from modal reconstructor', target_device_idx=target_device_idx)
            self.inputs['in_commands'] = InputValue(type=BaseValue, optional=True)
            self.inputs['in_commands_list'] = InputList(type=BaseValue, optional=True)            
            # TODO complete static allocation above

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        slopes = self.local_inputs['in_slopes']
        slopes_list = self.local_inputs['in_slopes_list']

        if slopes is None:
            self.slopes[:] = self.xp.hstack([x.slopes for x in slopes_list])
        else:
            self.slopes[:] = slopes.slopes

        if self.polc:
            commands = self.local_inputs['in_commands']
            commands_list = self.local_inputs['in_commands_list']
            if commands is None:
                self.commands[:] = self.xp.hstack([x.commands for x in commands_list])
            else:
                if commands.value is None:
                    # value will be None in the first iteration
                    self.commands[:] = 0.0
                else:
                    self.commands[:] = commands.value

    def trigger_code(self):
        if self.recmat.recmat is None:
            print("WARNING: modalrec skipping reconstruction because recmat is NULL")
            return

        # In the polc case, commands may be *alwats* refreshed if they are set with -1
        # (it might result in a kind of loop in the yml file)
        # Therefor we check the slopes input time and only run when they have been refreshed.
        if self.polc:

            slopes = self.local_inputs['in_slopes']
            slopes_list = self.local_inputs['in_slopes_list']

            if slopes is not None:
                slopes_time = slopes.generation_time
            else:
                slopes_time = slopes_list[0].generation_time

            if slopes_time != self.current_time:
                return


        if self.polc:
   
            if self.input_modes_index is not None:
                commands = self.commands[self.input_modes_index]
            elif self.input_modes_slice is not None:
                commands = self.commands[self.input_modes_slice]
            else:
                commands = self.commands

            comm_slopes = self.intmat.intmat @ commands
            self.pseudo_ol_modes.value = self.recmat.recmat @ (self.slopes + comm_slopes)
            self.pseudo_ol_modes.generation_time = self.current_time
            if self.projmat is None:
                output_modes = self.pseudo_ol_modes.value
            else:
                output_modes = self.projmat.recmat @ self.pseudo_ol_modes.value
            output_modes -= commands
            
        else:
            output_modes = self.recmat.recmat @ self.slopes

        self.modes.value = output_modes[self.output_slice]
        self.modes.generation_time = self.current_time

    def setup(self):
        super().setup()

        slopes = self.local_inputs['in_slopes']
        slopes_list = self.local_inputs['in_slopes_list']

        if not slopes and (not slopes_list or not all(slopes_list)):
            raise ValueError("Either 'slopes' or 'slopes_list' must be given as an input")
        
        if slopes is None:
            self.slopes = self.xp.hstack([x.slopes for x in slopes_list])
        else:
            self.slopes = self.to_xp(slopes.slopes.copy())

        if self.polc:
            commands = self.local_inputs['in_commands']
            commands_list = self.local_inputs['in_commands_list']
            if not commands and (not commands_list or not all(commands_list)):
                raise ValueError("When POLC is used, either 'commands' or 'commands_list' must be given as an input")

            self.commands = self.xp.zeros(self.in_commands_size, dtype=self.dtype)

