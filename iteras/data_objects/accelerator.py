'''
Options of the accelerator (see :mod:`iteras.accelerator`).

A section of the parameter file whose class is Accelerator makes Simul run the GPU
work of the objects as CUDA graphs chained on one stream per device, without
the host waiting for the GPU between objects.
'''


class Accelerator:
    '''
    Accelerator options.

    Parameters
    ----------
    exclude : list of str, optional
        Names of objects that are not accelerated (they run as without accelerator).
    capture_after : int, optional
        Number of triggers of each object before its CUDA graph is captured
        (default 1): the first triggers allocate the buffers and build the
        FFT plans that the graph reuses.
    check_pointers : bool, optional
        Before each launch of the graph of an object, check that the arrays of
        the object and of its inputs have not been reallocated since the
        capture (default True). An object whose arrays change is not accelerated
        anymore: its graph would read stale memory.
    report_file : str, optional
        File where the accelerator writes, at the end of the run, which objects were
        accelerated and why the others were not (default: only logged).
    '''

    def __init__(self,
                 exclude: list = None,
                 capture_after: int = 1,
                 check_pointers: bool = True,
                 report_file: str = None):
        self.exclude = list(exclude or [])
        if int(capture_after) < 1:
            raise ValueError(f'Accelerator: capture_after must be at least 1, got {capture_after}')
        self.capture_after = int(capture_after)
        self.check_pointers = bool(check_pointers)
        self.report_file = report_file

    def as_dict(self):
        return {'exclude': sorted(self.exclude), 'capture_after': self.capture_after,
                'check_pointers': self.check_pointers}
