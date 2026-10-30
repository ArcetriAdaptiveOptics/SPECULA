
import warnings
from copy import copy
from functools import cache

import iteras
from iteras import np
from iteras.base_time_obj import BaseTimeObj


@cache
def get_properties(cls):
    result = []
    classlist = cls.__mro__
    for cc in classlist:
        result.extend([attr for attr, value in vars(cc).items() if isinstance(value, property) ])
    return result


def _peer_source(array, target_device_idx):
    '''
    *array*, or a contiguous copy of it on its own device if it must be copied
    to another device: CuPy copies only contiguous arrays between devices.
    '''
    if array.device.id == target_device_idx or array.flags.c_contiguous:
        return array
    with array.device:
        copy = iteras.cp.ascontiguousarray(array)
    # The copy to the other device must follow this one
    _stream_wait(target_device_idx, array.device.id)
    return copy


def _stream_wait(waiting_device, device):
    '''
    Make the current stream of *waiting_device* wait (on the GPU, the host goes
    on) for the work queued so far on the current stream of *device*.
    '''
    cp = iteras.cp
    with cp.cuda.Device(device):
        event = cp.cuda.get_current_stream().record()
    with cp.cuda.Device(waiting_device):
        cp.cuda.get_current_stream().wait_event(event)


class BaseDataObj(BaseTimeObj):

    def __init__(self, target_device_idx: int=None, precision: int=None):
        """
        Initialize the base data object.

        Parameters:
        target_device_idx: int, optional
            device to be targeted for data storage. Set to -1 for CPU,
            to 0 for the first GPU device, 1 for the second GPU device, etc.
        precision: int, optional
            if None will use the global_precision, otherwise set to 0 for double, 1 for single
        """
        super().__init__(target_device_idx, precision)
        self.generation_time = -1
        self.tag = ''

    @classmethod
    def track_consumers_in_replay(cls):
        '''
        Override to return True for data types whose producers must not be
        silently left out of a targeted replay: Simul.build_targeted_replay()
        reports the objects that produce an output of such a type, consume an
        output of a replayed object, and are not part of the replay (for
        example, an object adding a contribution to a propagated field).
        '''
        return False

    def debug_summary(self):
        '''
        Value shown for this object in the MPI debug log of the inputs
        (see BaseProcessingObj.get_all_inputs()). Override to show its data.
        '''
        return self

    def transferDataTo(self, destobj, force_reallocation=False):
        '''
        Copy CPU/GPU arrays into an existing data object:
        iterate over all self attributes and, if a CPU or GPU array
        is detected, copy data into *destobj* without reallocating.

        Destination (CPU or GPU device) is inferred by *destobj.target_device_idx*,
        which must be set correctly before calling this method.
        '''
        # Copy between two GPUs: the copy, queued on the destination device, must
        # follow the work that produces the data on the source device
        peer = iteras.cp is not None and self.target_device_idx >= 0 \
            and destobj.target_device_idx >= 0 and self.target_device_idx != destobj.target_device_idx
        if peer:
            _stream_wait(destobj.target_device_idx, self.target_device_idx)

        # The arrays are attributes of the instance (dir() would also list the
        # class attributes and the properties, and is slow: this runs at each step)
        attr_list = list(self.__dict__)

        for attr in attr_list:
            self_attr = getattr(self, attr)
            self_type = type(self_attr)
            if self_type not in iteras.array_types:
                continue

            dest_attr = getattr(destobj, attr)
            dest_type = type(dest_attr)

            if dest_type not in iteras.array_types:
                self.logger.warning(f'destination attribute is not a cupy/numpy array, forcing reallocation ({destobj}.{attr})')
                force_reallocation = True

            # Destination array had the correct type: perform in-place data copy
            if not force_reallocation:
                # Detect whether the array types are correct for all four cases:
                # Device to CPU, CPU to device, device-to-device, and CPU-CPU. Also check whether
                # the target_device_idx is set correctly for the destination object.
                DtD = iteras.cp is not None and (self_type == iteras.cp.ndarray) and (dest_type == iteras.cp.ndarray) and destobj.target_device_idx >= 0
                DtH = iteras.cp is not None and (self_type == iteras.cp.ndarray) and (dest_type == np.ndarray) and destobj.target_device_idx == -1
                HtD = iteras.cp is not None and (self_type == np.ndarray) and (dest_type == iteras.cp.ndarray) and destobj.target_device_idx >= 0
                HtH = (self_type == np.ndarray) and (dest_type == np.ndarray) and destobj.target_device_idx == -1
                if DtD:
                    # Performance warnings here are expected, because we might
                    # trigger a peer-to-peer transfer between devices
                    with warnings.catch_warnings():
                        if self.PerformanceWarning:
                            warnings.simplefilter("ignore", category=self.PerformanceWarning)
                        try:
                            dest_attr[...] = _peer_source(self_attr, destobj.target_device_idx)
                        except Exception:
                            self.logger.warning(f'copy between devices failed, forcing reallocation ({destobj}.{attr})')
                            force_reallocation = True
                elif DtH:
                    # Do not set blocking=True for cupy 12.x compatibility.
                    # Blocking is True by default in later versions anyway
                    self_attr.get(out=dest_attr)
                elif HtD:
                    dest_attr.set(self_attr)
                elif HtH:
                    dest_attr[...] = self_attr
                else:
                    self.logger.warning(f'mismatch between target_device_idx and array allocation, forcing reallocation ({destobj}.{attr})')
                    force_reallocation = True

            # Otherwise, reallocate
            if force_reallocation:
                DtD = iteras.cp is not None and (self_type == iteras.cp.ndarray) and destobj.target_device_idx >= 0
                DtH = iteras.cp is not None and (self_type == iteras.cp.ndarray) and destobj.target_device_idx == -1
                HtD = (self_type == np.ndarray) and destobj.target_device_idx >= 0
                HtH = (self_type == np.ndarray) and destobj.target_device_idx == -1

                if DtD:
                    # Performance warnings here are expected, because we might
                    # trigger a peer-to-peer transfer between devices
                    with warnings.catch_warnings():
                        if self.PerformanceWarning:
                            warnings.simplefilter("ignore", category=self.PerformanceWarning)
                        with iteras.cp.cuda.Device(destobj.target_device_idx):
                            setattr(destobj, attr, iteras.cp.asarray(
                                _peer_source(self_attr, destobj.target_device_idx)))
                if DtH:
                    # Do not set blocking=True for cupy 12.x compatibility.
                    # Blocking is True by default in later versions anyway
                    setattr(destobj, attr, self_attr.get())
                if HtD:
                    setattr(destobj, attr, iteras.cp.asarray(self_attr))
                if HtH:
                    setattr(destobj, attr, np.asarray(self_attr))

        if peer:
            # ... and the source device must not overwrite the data before it is copied
            _stream_wait(self.target_device_idx, destobj.target_device_idx)
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
                cloned.xp = iteras.cp
            else:
                cloned.xp = np
            cloned.target_device_idx = target_device_idx

            self.transferDataTo(cloned, force_reallocation=True)
            return cloned
