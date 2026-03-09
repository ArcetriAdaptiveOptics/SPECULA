
import queue
import multiprocessing as mp

from specula.processing_objects.specula_input import SpeculaInput


class SpeculaTerminalInput(SpeculaInput):
    """
    Specula terminal input processing object. Handles input from a terminal.
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
        super().__init__(output_list,
                         target_device_idx=target_device_idx,
                         precision=precision)

        def terminal_task(q):
            tokens = input().split()
            if len(tokens) == 1:
                q.put(tokens[0], True)
            elif len(tokens) == 2:
                q.put(tokens[0], tokens[1])
            else:
                print('Input not recognized')

        self.set_input_task(terminal_task)

