
import weakref
from functools import wraps
from inspect import signature

import specula
from specula import np, cp, to_xp
from specula import global_precision, default_target_device, default_target_device_idx
from specula import cpu_float_dtype_list, gpu_float_dtype_list
from specula import cpu_complex_dtype_list, gpu_complex_dtype_list
from specula.log import get_specula_logger, INIT_PLACEHOLDER_NAME

# GPU memory tracking (see BaseTimeObj.startMemUsageCount()).
# Objects whose memory is being counted right now, innermost last.
_mem_count_stack = []
# Objects not built inside another object on the same device, in creation order.
# Every other object is nested into one of these, which includes its memory.
_top_level_objs = weakref.WeakValueDictionary()
_top_level_counter = 0
# GPUs used by some object, which therefore already have a CUDA context.
_used_devices = set()

MB = 1024 * 1024


def _pool_used_bytes(device_idx):
    '''Bytes used in the cupy memory pool of a device, whatever the current device.'''
    with cp.cuda.Device(device_idx):
        return cp.get_default_memory_pool().used_bytes()


def mem_registry_mark():
    '''Current position in the registry of top-level objects, see top_level_objs().'''
    return _top_level_counter


def top_level_objs(since=0):
    '''Live top-level objects registered after mem_registry_mark() returned *since*.'''
    return [obj for idx, obj in list(_top_level_objs.items()) if idx >= since]


def _owner(obj):
    ref = obj.__dict__.get('_mem_owner')
    return ref() if ref is not None else None


def _top_owner(obj):
    while (owner := _owner(obj)) is not None:
        obj = owner
    return obj


def gpu_mem_report(objs, names=None):
    '''
    Table of the GPU memory of *objs* (top-level objects), grouped by device.

    For each object: its own memory, the total including the objects nested
    into it, and the memory held in its arrays. For each device, the sum of the
    totals is compared with the memory pool: the difference is memory that is not
    attributed to any object (allocated outside __init__/setup/first trigger,
    or by objects not in *objs*).
    Arrays shared between objects are credited to the first one listed.
    '''
    names = names or {}
    by_device = {}
    for obj in objs:
        if getattr(obj, 'target_device_idx', -1) >= 0:
            by_device.setdefault(obj.target_device_idx, []).append(obj)

    seen = set()
    lines = []
    for dev in sorted(by_device):
        lines.append(f'  GPU {dev}:')
        lines.append(f'    {"object":30s} {"class":26s} {"own MB":>10s} {"total MB":>10s} {"held MB":>10s}')
        tot = 0
        for obj in by_device[dev]:
            name = names.get(id(obj)) or getattr(obj, 'name', None) or \
                   f'{type(obj).__name__}({getattr(obj, "tag", "")})'
            held = obj.gpu_bytes_held(seen)
            tot += obj.gpu_bytes_used
            lines.append(f'    {name:30.30s} {type(obj).__name__:26.26s} {obj.gpu_bytes_own/MB:10.2f}'
                         f' {obj.gpu_bytes_used/MB:10.2f} {held/MB:10.2f}')
        pool = _pool_used_bytes(dev)
        lines.append(f'    sum of totals {tot/MB:.2f} MB, memory pool {pool/MB:.2f} MB,'
                     f' unattributed {(pool-tot)/MB:.2f} MB')
    return '\n'.join(lines)


def _check_specula_initialized():
    if specula.global_precision is None:
        raise RuntimeError(
            'SPECULA is not initialized: call specula.init(device_idx, precision) '
            'before creating any SPECULA object')
    if global_precision is None:
        # init() was called, but after this module was imported, so the
        # names imported above from specula are stale
        raise RuntimeError(
            'specula.init() was called after importing SPECULA submodules. '
            'Call specula.init() right after "import specula", before any other '
            'specula.* import')


class BaseTimeObj:
    # GPU memory counted by start/stopMemUsageCount(), total and of the nested objects
    gpu_bytes_used = 0
    gpu_bytes_children = 0

    def __init__(self, target_device_idx=None, precision=None):
        """
        Creates a new base_time object.

        Parameters:
        precision (int, optional): if None will use the global_precision, otherwise pass 0 for double, 1 for single
        target_device_idx (int, optional): if None will use the default_target_device_idx,
        otherwise pass -1 for cpu, i for GPU of index i
        """
        _check_specula_initialized()

        self.logger = get_specula_logger('specula.'+self.__class__.__name__)
        self.logger.set_instance_name(INIT_PLACEHOLDER_NAME)

        self._time_resolution = int(1e9)

        if precision is None:
            self.precision = global_precision
        else:
            self.precision = precision

        if target_device_idx is None:
            self.target_device_idx = default_target_device_idx
        else:
            self.target_device_idx = target_device_idx

        if self.target_device_idx >= 0 and cp is None:
            raise RuntimeError(
                f'target_device_idx={self.target_device_idx} requested, but cupy is not available '
                '(not installed, failed to import, or disabled by SPECULA_DISABLE_GPU)')

        if self.target_device_idx >= 0:
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

        if self.target_device_idx >= 0:
            from cupyx.scipy.ndimage import rotate as ndimage_rotate
            from cupyx.scipy.ndimage import shift as ndimage_shift
            from cupyx.scipy.ndimage import center_of_mass as ndimage_center_of_mass
            from cupyx.scipy.fft import ifft2 as scipy_ifft2
            from cupyx.scipy.fft import idct, dct
            from cupyx.scipy.linalg import lu_factor, lu_solve

            self._target_device.use()
            _used_devices.add(self.target_device_idx)
            # If the count started before the device was known and could not
            # read it (see startMemUsageCount), start it from here.
            before = self.__dict__.get('_mem_before')
            if self.__dict__.get('_mem_count_depth', 0) > 0 and before is not None \
                    and self.target_device_idx not in before:
                before[self.target_device_idx] = _pool_used_bytes(self.target_device_idx)
            from cupy._util import PerformanceWarning
            self.PerformanceWarning = PerformanceWarning
        else:
            from scipy.ndimage import rotate as ndimage_rotate
            from scipy.ndimage import shift as ndimage_shift
            from scipy.ndimage import center_of_mass as ndimage_center_of_mass
            from scipy.fft import ifft2 as scipy_ifft2
            from scipy.fftpack import idct, dct
            from scipy.linalg import lu_factor, lu_solve
            self.PerformanceWarning = None

        self.ndimage_rotate = ndimage_rotate
        self.ndimage_shift = ndimage_shift
        self.ndimage_center_of_mass = ndimage_center_of_mass
        self._lu_factor = lu_factor
        self._lu_solve = lu_solve
        self._scipy_ifft2 = scipy_ifft2
        self.dct = dct
        self.idct = idct

    def init_logging(self, level=None):
        name = getattr(self, 'name', None)
        self.logger.set_instance_name(name)
        if level is not None:
            self.logger.setLevel(level)

    def t_to_seconds(self, t):
        return float(t) / float(self._time_resolution)

    def seconds_to_t(self, seconds):
        return int(round(seconds, ndigits=9) * self._time_resolution)

    def __getstate__(self):
        # Weak references cannot be pickled (e.g. objects sent via MPI):
        # a copy is not nested into the original owner.
        state = self.__dict__.copy()
        state.pop('_mem_owner', None)
        state.pop('_mem_parent', None)
        return state

    @property
    def gpu_bytes_own(self):
        '''GPU memory allocated by this object, excluding the objects nested into it.'''
        return self.gpu_bytes_used - self.gpu_bytes_children

    def startMemUsageCount(self, device_hint=None):
        '''
        Start counting the GPU memory allocated by this object.
        *device_hint* is the device the object is expected to use, if known
        before BaseTimeObj.__init__ (the target_device_idx argument of __init__).
        Called around __init__() and setup() (see monitorMem), and by LoopControl
        around setup() and the calls up to the first trigger.

        Calls can nest (a subclass __init__/setup calling super(), or LoopControl
        wrapping setup()): only the outermost one measures.

        An object built while another one on the same device is being counted
        is nested into it: its memory is also added to the other object's total.
        '''
        depth = self.__dict__.get('_mem_count_depth', 0)
        self._mem_count_depth = depth + 1
        if depth > 0:
            return
        if '_mem_parent' not in self.__dict__:
            # First count, from the __init__ wrapper
            parent = _mem_count_stack[-1] if _mem_count_stack else None
            self._mem_parent = weakref.ref(parent) if parent is not None else None
        _mem_count_stack.append(self)

        device_idx = getattr(self, 'target_device_idx', None)
        if cp is None or (device_idx is not None and device_idx < 0):
            self._mem_before = {}
        elif device_idx is None:
            # Before BaseTimeObj.__init__ the device is not known. Read only the GPUs
            # already used or about to be: reading the memory pool of any other GPU
            # would create a CUDA context (hundreds of MB) on it.
            devices = set(_used_devices)
            for d in (default_target_device_idx, device_hint):
                if d is not None and d >= 0:
                    devices.add(d)
            self._mem_before = {d: _pool_used_bytes(d) for d in devices}
        else:
            self._mem_before = {device_idx: _pool_used_bytes(device_idx)}

    def stopMemUsageCount(self):
        self._mem_count_depth -= 1
        if self._mem_count_depth > 0:
            return
        _mem_count_stack.pop()

        device_idx = getattr(self, 'target_device_idx', -1)
        if device_idx in self._mem_before:
            delta = _pool_used_bytes(device_idx) - self._mem_before[device_idx]
        else:
            delta = 0
        self.gpu_bytes_used += delta

        if '_mem_owner' not in self.__dict__:
            # End of the first count: the device is known, decide where this object belongs
            parent = self._mem_parent() if self._mem_parent is not None else None
            del self._mem_parent
            if device_idx >= 0 and parent is not None and \
                    getattr(parent, 'target_device_idx', -1) == device_idx:
                self._mem_owner = weakref.ref(parent)
            else:
                global _top_level_counter
                self._mem_owner = None
                _top_level_objs[_top_level_counter] = self
                _top_level_counter += 1

        # Add to the owners' totals, up to the first one that is being counted
        # right now, whose own measurement already includes this memory.
        owner = _owner(self)
        while owner is not None:
            owner.gpu_bytes_children += delta
            if owner.__dict__.get('_mem_count_depth', 0) > 0:
                break
            owner.gpu_bytes_used += delta
            owner = _owner(owner)

    def gpu_bytes_held(self, seen=None):
        '''
        GPU memory of the cupy arrays reachable from this object's attributes,
        through the objects nested into it, but not through other top-level
        objects or through the inputs.
        Unlike gpu_bytes_used, it does not include FFT plans, CUDA graphs,
        caches or other memory not stored in an array.

        Parameters:
        seen (set, optional): pointers of memory already counted, updated in
        place. Pass the same set to several objects to count shared arrays once.
        '''
        if cp is None:
            return 0
        if seen is None:
            seen = set()
        total = 0
        visited = set()
        todo = [self]
        while todo:
            obj = todo.pop()
            if id(obj) in visited:
                continue
            visited.add(id(obj))
            if isinstance(obj, cp.ndarray):
                mem = obj.data.mem
                if mem.ptr not in seen:
                    seen.add(mem.ptr)
                    total += mem.size
            elif isinstance(obj, dict):
                todo.extend(obj.values())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                todo.extend(obj)
            elif isinstance(obj, BaseTimeObj):
                if obj is self or _top_owner(obj) is self:
                    todo.extend(v for k, v in vars(obj).items() if k not in ('inputs', 'local_inputs'))
            elif type(obj).__module__.startswith('specula.') and hasattr(obj, '__dict__') \
                    and not isinstance(obj, type):
                todo.extend(vars(obj).values())
        return total

    def printMemUsage(self):
        if hasattr(self, 'target_device_idx') and self.target_device_idx >= 0:
            self.logger.info(f'cupy memory used by {getattr(self, "name", self.__class__.__name__)}:'
                             f' {self.gpu_bytes_used / MB:.2f} MB'
                             f' (own {self.gpu_bytes_own / MB:.2f} MB)')

    def monitorMem(f):

        @wraps(f)
        def monitorMem_wrapper(*args, **kwargs):
            self = args[0]
            self.startMemUsageCount(device_hint=kwargs.get('target_device_idx'))
            try:
                return f(*args, **kwargs)
            finally:
                self.stopMemUsageCount()

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

