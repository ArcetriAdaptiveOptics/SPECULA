
from functools import wraps
from inspect import signature

from specula import np, cp, to_xp, process_rank
from specula import global_precision, default_target_device, default_target_device_idx
from specula import cpu_float_dtype_list, gpu_float_dtype_list
from specula import cpu_complex_dtype_list, gpu_complex_dtype_list


class BaseTimeObj:
    def __init__(self, target_device_idx=None, precision=None):
        """
        Creates a new base_time object.

        Parameters:
        precision (int, optional): if None will use the global_precision, otherwise pass 0 for double, 1 for single
        target_device_idx (int, optional): if None will use the default_target_device_idx, otherwise pass -1 for cpu, i for GPU of index i
        """
        self._time_resolution = int(1e9)
        self.gpu_bytes_used = 0

        if precision is None:
            self.precision = global_precision
        else:
            self.precision = precision

        if target_device_idx is None:
            self.target_device_idx = default_target_device_idx
        else:
            self.target_device_idx = target_device_idx

        if self.target_device_idx>=0:
            self._target_device = cp.cuda.Device(self.target_device_idx)      # GPU case
            self.dtype = gpu_float_dtype_list[self.precision]
            self.complex_dtype = gpu_complex_dtype_list[self.precision]
            self.xp = cp
            self.xp_str = 'cp'
        else:
            self._target_device = default_target_device                # CPU case
            self.dtype = cpu_float_dtype_list[self.precision]
            self.complex_dtype = cpu_complex_dtype_list[self.precision]
            self.xp = np
            self.xp_str = 'np'

        if self.target_device_idx>=0:
            from cupyx.scipy.ndimage import rotate as ndimage_rotate
            from cupyx.scipy.ndimage import shift as ndimage_shift
            from cupyx.scipy.fft import ifft2 as scipy_ifft2
            from cupyx.scipy.linalg import lu_factor, lu_solve

            self._target_device.use()
            self.gpu_bytes_used_before = cp.get_default_memory_pool().used_bytes()
            from cupy._util import PerformanceWarning
            self.PerformanceWarning = PerformanceWarning
        else:
            from scipy.ndimage import rotate as ndimage_rotate
            from scipy.ndimage import shift as ndimage_shift
            from scipy.fft import ifft2 as scipy_ifft2
            from scipy.linalg import lu_factor, lu_solve
            self.PerformanceWarning = None

        self.ndimage_rotate = ndimage_rotate
        self.ndimage_shift = ndimage_shift
        self._lu_factor = lu_factor
        self._lu_solve = lu_solve
        self._scipy_ifft2 = scipy_ifft2

    def t_to_seconds(self, t):
        return float(t) / float(self._time_resolution)

    def seconds_to_t(self, seconds):
        return int(round(seconds, ndigits=9) * self._time_resolution)

    def startMemUsageCount(self):
        if hasattr(self, 'target_device_idx') and self.target_device_idx >= 0:
            self.gpu_bytes_used_before = cp.get_default_memory_pool().used_bytes()

    def stopMemUsageCount(self):
        if hasattr(self, 'target_device_idx') and self.target_device_idx >= 0:
            self.gpu_bytes_used_after = cp.get_default_memory_pool().used_bytes()
            self.gpu_bytes_used += self.gpu_bytes_used_after - self.gpu_bytes_used_before
            self.gpu_bytes_used_before = self.gpu_bytes_used_after

    def printMemUsage(self):
        if hasattr(self, 'target_device_idx') and self.target_device_idx >= 0:
            print(process_rank, f'\tcupy memory used by {self.__class__.__name__}: {self.gpu_bytes_used / (1024*1024)} MB')

    def monitorMem(f):

        @wraps(f)
        def monitorMem_wrapper(*args, **kwargs):
            self = args[0]
            self.startMemUsageCount()
            retval = f(*args, **kwargs)
            self.stopMemUsageCount()
            return retval

        monitorMem_wrapper.__signature__ = signature(f)    # Needed to track type hints in __init__ for object creation
        return monitorMem_wrapper

    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        methods = ['__init__', 'setup']

        for name, attr in cls.__dict__.items():
            if name in methods:
                setattr(cls, name, BaseTimeObj.monitorMem(attr))

    def to_xp(self, v, dtype=None, force_copy=False):
        '''
        Method wrapping the global to_xp function.
        '''
        return to_xp(self.xp, v, dtype, force_copy)

