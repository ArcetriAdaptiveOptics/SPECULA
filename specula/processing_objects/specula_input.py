import queue

from specula.connections import split_output
from specula.base_processing_obj import BaseProcessingObj
from specula.scalar_values import IntValue, FloatValue, StringValue


class SpeculaInput(BaseProcessingObj):
    """
    Specula input processing object. Handles interactive inputs

    This class is meant to provide outputs that can be set
    interactively and/or asynchronously wrt. the normal simulation run.

    Derived classes receive inputs from the "outside" (typically in a
    background thread) and pass them to put_input(), which validates
    them and stores them in a thread-safe queue. The queue is emptied
    at each trigger call, so that new values are applied only at step
    boundaries and their generation_time matches the current step.
    """
    def __init__(self,
                 output_list: list,
                 target_device_idx: int=None,
                 precision:int =None):
        """
        output_list: list of strings
            List of output names to be generated
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int, optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """
        super().__init__(target_device_idx=target_device_idx,
                         precision=precision)

        for name in output_list:
            desc = split_output(name, detect_types=True)
            typ = desc.type
            real_name = desc.obj_name
            if typ == 'float':
                self.outputs[real_name] = FloatValue(0.0)
            elif typ == 'int':
                self.outputs[real_name] = IntValue(0)
            elif typ == 'str':
                self.outputs[real_name] = StringValue('')
            else:
                raise ValueError(f'Unsupported type {typ} for output {real_name}')

        self.q = queue.Queue()

    def put_input(self, name, value):
        """
        Validate an input value and queue it, to be applied to output
        "name" at the next trigger. Safe to call from any thread.

        Raises KeyError if "name" is not a known output, and ValueError
        if "value" cannot be converted to the output type.
        """
        if name not in self.outputs:
            raise KeyError(f'Unknown output: {name}')
        cast_func = self.outputs[name].type
        try:
            converted = cast_func(value)
        except (TypeError, ValueError):
            raise ValueError(f'Rejected input {value} for output {name}: '
                             f'cannot convert to type {cast_func.__name__}') from None
        self.q.put((name, converted))

    def trigger_code(self):
        """
        Apply all values queued by put_input() since the last trigger.
        """
        while True:
            try:
                name, value = self.q.get_nowait()
            except queue.Empty:
                break
            self.outputs[name].value = value
            self.outputs[name].generation_time = self.current_time
