
import queue
import multiprocessing as mp

from specula.base_value import BaseValue
from specula.base_processing_obj import BaseProcessingObj


class SpeculaInput(BaseProcessingObj):
    """
    Specula input processing object. Handles interactive inputs

    This class is meant to provide outputs that can be set
    interactively and/or asynchronously wrt. the normal simulation run.

    Derived classes must implement a function or callable that receives
    inputs from the "outside" and puts them into a queue that will be
    emptied at each trigger call.
    This function is registered passing it to "set_input_task"
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
            self.outputs[name] = BaseValue(target_device_idx=target_device_idx, precision=precision)

    def set_input_task(self, task):
        """
        "task" must be a Python callable that accepts one argument.
        The argument will be set to a mp.Queue() instance, on which
        the callable must call put() with a tuple of two values:
        output name and output value.
        """
        self.q = mp.Queue()
        self.p = mp.Process(target=task, args=(self.q,))
        self.p.start()

    def trigger_code(self):
        """
        Get new values from the input queue and set corresponding outputs,
        repeat until the input queue is empty.

        We don't use self.q.empty() to check the queue status, since
        it does not guarantee that the subsequent get() won't block.
        """
        try:
            while True:
                name, value = self.q.get(block=False)
                try:
                    # Replace value directly instead of using set_value
                    # so that we can accept any valid type instead of just numeric ones
                    self.outputs[name].value = value
                    self.outputs[name].generation_time = self.current_time
                except KeyError:
                    print('Unknown output', name)
        except queue.Empty:
            pass
