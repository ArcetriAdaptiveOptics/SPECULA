from specula import np, cp, to_xp
from specula import global_precision, default_target_device, default_target_device_idx, DummyDecoratorAndContextManager
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
        else:
            self._target_device = default_target_device                # CPU case
            self.dtype = cpu_float_dtype_list[self.precision]
            self.complex_dtype = cpu_complex_dtype_list[self.precision]
            self.xp = np

        if self.target_device_idx>=0:
            from cupyx.scipy.ndimage import rotate
            from cupyx.scipy.ndimage import shift
            from cupyx.scipy.interpolate import RegularGridInterpolator
            from cupyx.scipy.fft import fft2 as scipy_fft2
            from cupyx.scipy.fft import ifft2 as scipy_ifft2
            from cupyx.scipy.fft import get_fft_plan
            from cupyx.scipy.linalg import lu_factor, lu_solve

            self._target_device.use()
            self.gpu_bytes_used_before = cp.get_default_memory_pool().used_bytes()
        else:
            from scipy.ndimage import rotate
            from scipy.ndimage import shift
            from scipy.interpolate import RegularGridInterpolator
            from scipy.linalg import lu_factor, lu_solve
            from scipy.fft import fft2 as scipy_fft2
            from scipy.fft import ifft2 as scipy_ifft2
            def get_fft_plan(*args, **kwargs):
                return DummyDecoratorAndContextManager()

        self.rotate = rotate
        self.shift = shift
        self.RegularGridInterpolator = RegularGridInterpolator
        self._get_fft_plan = get_fft_plan
        self._lu_factor = lu_factor
        self._lu_solve = lu_solve
        self._scipy_fft2 = scipy_fft2
        self._scipy_ifft2 = scipy_ifft2
        self._get_fft_plan = get_fft_plan

    def t_to_seconds(self, t):
        return float(t) / float(self._time_resolution)

    def seconds_to_t(self, seconds):
        if self._time_resolution == 0:
            return 0

        ss = f"{float(seconds):.9f}".rstrip('0').rstrip('.')
        if '.' not in ss:
            ss += '.0'

        dotpos = ss.find('.')
        intpart = ss[:dotpos]
        fracpart = ss[dotpos + 1:]

        return (int(intpart) * self._time_resolution +
                int(fracpart) * (self._time_resolution // (10 ** len(fracpart))))

    def startMemUsageCount(self):
        if self.target_device_idx >= 0:
            self.gpu_bytes_used_before = cp.get_default_memory_pool().used_bytes()

    def stopMemUsageCount(self):
        if self.target_device_idx >= 0:
            self.gpu_bytes_used_after = cp.get_default_memory_pool().used_bytes()
            self.gpu_bytes_used += self.gpu_bytes_used_after - self.gpu_bytes_used_before
            self.gpu_bytes_used_before = self.gpu_bytes_used_after

    def printMemUsage(self):
        if self.target_device_idx >= 0:
            print(f'\tcupy memory used by {self.__class__.__name__}: {self.gpu_bytes_used / (1024*1024)} MB')

    # not used for now
    def monitorMem(function):
        def monitorMem_wrapper(*args):
            self = args[0]
            self.startMemUsageCount()
            function(*args)
            self.stopMemUsageCount()
        return monitorMem_wrapper

    def to_xp(self, v, dtype=None):
        '''
        Method wrapping the global to_xp function.
        '''
        return to_xp(self.xp, v, dtype)

