
import warnings
from copy import copy
from functools import lru_cache

from specula import cp, np
from specula.base_time_obj import BaseTimeObj


# We use lru_cache() instead of cache() for python 3.8 compatibility
@lru_cache(maxsize=None)
def get_properties(cls):
    result = []
    classlist = cls.__mro__
    for cc in classlist:
        result.extend([attr for attr, value in vars(cc).items() if isinstance(value, property) ]) 
    return result


class BaseDataObj(BaseTimeObj):
    def __init__(self, target_device_idx=None, precision=None):
        """
        Initialize the base data object.

        Parameters:
        precision (int, optional):if None will use the global_precision, otherwise pass 0 for double, 1 for single
        """
        super().__init__(target_device_idx, precision)
        self._generation_time = -1

    @property
    def generation_time(self):
        return self._generation_time

    @generation_time.setter
    def generation_time(self, value):
        self._generation_time = value

    def transferDataTo(self, destobj, force_reallocation=False):
        '''
        Copy CPU/GPU arrays into an existing data object:
        iterate over all self attributes and, if a CPU or GPU array
        is detected, copy data into *destobj* without reallocating.

        Destination (CPU or GPU device) is inferred by *destobj.target_device_idx*,
        which must be set correctly before calling this method.
        '''
        # Get a list of all attributes, but skip properties
        pp = get_properties(type(self))
        attr_list = [attr for attr in dir(self) if attr not in pp]       

        for attr in attr_list:
            self_attr = getattr(self, attr)
            self_type = type(self_attr)
            if self_type not in [cp.ndarray, np.ndarray]:
                continue

            dest_attr = getattr(destobj, attr)
            dest_type = type(dest_attr)

            if dest_type not in [cp.ndarray, np.ndarray]:
                print(f'Warning: destination attribute is not a cupy/numpy array, forcing reallocation ({destobj}.{attr})')
                force_reallocation = True

            # Detect whether the array types are correct for all three cases:
            # Device to CPU, CPU to device, and device-to-device.
            DtD = (self_type == cp.ndarray) and (dest_type == cp.ndarray) and destobj.target_device_idx >= 0
            DtH = (self_type == cp.ndarray) and (dest_type == np.ndarray) and destobj.target_device_idx == -1
            HtD = (self_type == np.ndarray) and (dest_type == cp.ndarray) and destobj.target_device_idx >= 0
            HtH = (self_type == np.ndarray) and (dest_type == np.ndarray) and destobj.target_device_idx == -1

            # Destination array had the correct type: perform in-place data copy
            if not force_reallocation:
                if DtD:
                    # Performance warnings here are expected, because we might
                    # trigger a peer-to-peer transfer between devices
                    with warnings.catch_warnings():
                        if self.PerformanceWarning:
                            warnings.simplefilter("ignore", category=self.PerformanceWarning)
                        dest_attr[:] = self_attr
                elif DtH:
                    # Do not set blocking=True for cupy 12.x compatibility.
                    # Blocking is True by default in later versions anyway
                    self_attr.get(out=dest_attr)
                elif HtD:
                    dest_attr.set(self_attr)
                elif HtH:
                    dest_attr[:] = self_attr
                else:
                    print(f'Warning: mismatch between target_device_idx and array allocation, forcing reallocation ({destobj}.{attr})')
                    force_reallocation = True

            # Otherwise, reallocate
            if force_reallocation:
                DtD = (self_type == cp.ndarray) and destobj.target_device_idx >= 0
                DtH = (self_type == cp.ndarray) and destobj.target_device_idx == -1
                HtD = (self_type == np.ndarray) and destobj.target_device_idx >= 0
                HtH = (self_type == np.ndarray) and destobj.target_device_idx == -1

                if DtD:
                    # Performance warnings here are expected, because we might
                    # trigger a peer-to-peer transfer between devices
                    with warnings.catch_warnings():
                        if self.PerformanceWarning:
                            warnings.simplefilter("ignore", category=self.PerformanceWarning)
                        setattr(destobj, attr, cp.asarray(self_attr))
                if DtH:
                    # Do not set blocking=True for cupy 12.x compatibility.
                    # Blocking is True by default in later versions anyway
                    setattr(destobj, attr, self_attr.get())
                if HtD:
                    setattr(destobj, attr, cp.asarray(self_attr))
                if HtH:
                    setattr(destobj, attr, np.asarray(self_attr))

        destobj.generation_time = self.generation_time

    def copyTo(self, target_device_idx):
        '''
        Duplicate a data object on another device,
        alllocating all CPU/GPU arrays on the new device.
        '''
        if target_device_idx == self.target_device_idx:
            return self
        else:
            cloned = copy(self)

            if target_device_idx >= 0:
                cloned.xp = cp
            else:
                cloned.xp = np
            cloned.target_device_idx = target_device_idx

            self.transferDataTo(cloned, force_reallocation=True)
            return cloned
