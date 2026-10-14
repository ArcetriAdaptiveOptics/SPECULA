from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputList, InputValue
from specula.data_objects.slopes import Slopes


class BaseModalrec(BaseProcessingObj):
    """
    Base Modal Reconstructor processing object.
    Handles common slope inputs, modal outputs, and memory allocation
    for all modal reconstructors. Specific algorithms implement trigger_code().
    """

    def __init__(self, target_device_idx: int = None, precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.slopes = None  # to be allocated in setup()

        self.modes = BaseValue('output modes',
                               target_device_idx=target_device_idx,
                               precision=precision)

        self.inputs['in_slopes'] = InputValue(type=Slopes, optional=True)
        self.inputs['in_slopes_list'] = InputList(type=Slopes, optional=True)
        self.outputs['out_modes'] = self.modes

    @classmethod
    def input_names(cls):
        return {
            'in_slopes': InputDesc(Slopes,
                         'Input wavefront slope vector (optional)'),
            'in_slopes_list': InputDesc(Slopes,
                              'List of input slope vectors (optional)')
        }

    @classmethod
    def output_names(cls):
        return {'out_modes': OutputDesc(BaseValue, 'Reconstructed modal command vector')}

    def setup(self):
        super().setup()

        slopes = self.local_inputs['in_slopes']
        slopes_list = self.local_inputs['in_slopes_list']

        if not slopes and (not slopes_list or not all(slopes_list)):
            raise ValueError("Either 'in_slopes' or 'in_slopes_list' must be given as an input")

        if slopes is None:
            self.slopes = self.xp.hstack([x.slopes for x in slopes_list])
        else:
            self.slopes = self.to_xp(slopes.slopes.copy())

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        slopes = self.local_inputs['in_slopes']
        slopes_list = self.local_inputs['in_slopes_list']

        if slopes is None:
            self.xp.concatenate([x.slopes for x in slopes_list], out=self.slopes)
        else:
            self.slopes[:] = slopes.slopes

    def trigger_code(self):
        raise NotImplementedError("Subclasses must implement the trigger_code method.")


class BasePolcModalrec(BaseModalrec):
    """
    Base class for the Pseudo Open Loop Control (POLC) modal reconstructors.
    Adds the command inputs, copied into self.commands at each step.
    Derived classes must set self.in_commands_size in their constructor.
    """

    def __init__(self, target_device_idx: int = None, precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.commands = None  # to be allocated in setup()

        self.inputs['in_commands'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_commands_list'] = InputList(type=BaseValue, optional=True)

    @classmethod
    def input_names(cls):
        inputs = super().input_names()
        inputs.update({
            'in_commands': InputDesc(BaseValue,
                           'Current output command vector for POLC (optional)'),
            'in_commands_list': InputDesc(BaseValue,
                                'List of current command vectors for POLC (optional)')
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
            # Handle list of commands (e.g. from multiple DMs)
            self.xp.concatenate([x.value for x in commands_list], out=self.commands)
        else:
            if commands.value is None:
                self.commands[:] = 0.0
            else:
                self.commands[:] = commands.value

    def slopes_updated(self):
        '''True if the input slopes have been updated at the current time'''
        slopes = self.local_inputs['in_slopes']
        slopes_list = self.local_inputs['in_slopes_list']
        slopes_time = slopes.generation_time if slopes is not None else slopes_list[0].generation_time
        return slopes_time == self.current_time
