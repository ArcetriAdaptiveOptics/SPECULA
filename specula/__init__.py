import numpy as np
import os
import functools
from functools import wraps

cpu_float_dtype_list = [np.float64, np.float32]
cpu_complex_dtype_list = [np.complex128, np.complex64]

gpuEnabled = False
cp = None
xp = None
global_precision = None
float_dtype_list = None
complex_dtype_list = None
gpu_float_dtype_list = cpu_float_dtype_list
gpu_complex_dtype_list = cpu_complex_dtype_list
float_dtype = None
complex_dtype = None
default_target_device_idx = None
default_target_device = None
ASEC2RAD = np.pi / (3600 * 180)
RAD2ASEC = 1.0 / ASEC2RAD

# precision = 0 -> double precision
# precision = 1 -> single precision

# target_device = -1 -> CPU
# target_device = i>-1 -> GPUi

# you might have a GPU working and cupy installed
# and still want to use the CPU (idx==-1) as default_target_device
# in this case you might still want to allocate some objects on
# a GPU device (idx>=0).
# This can be checked later looking at the  value of gpuEnabled.

def init(device_idx=-1, precision=0):
    global xp
    global cp
    global gpuEnabled
    global global_precision
    global float_dtype_list
    global complex_dtype_list
    global gpu_float_dtype_list
    global gpu_complex_dtype_list
    global float_dtype
    global complex_dtype
    global default_target_device_idx
    global default_target_device
    
    default_target_device_idx = device_idx
    systemDisable = os.environ.get('SPECULA_DISABLE_GPU', 'FALSE')
    if systemDisable=='FALSE':
        try:
            import cupy as cp
            print("Cupy import successfull. Installed version is:", cp.__version__)
            gpuEnabled = True
            cp = cp
        except:
            print("Cupy import failed. SPECULA will fall back to CPU use.")
            cp = None
            xp = np
            default_target_device_idx=-1
    else:
        print("env variable SPECULA_DISABLE_GPU prevents using the GPU.")
        cp = None
        xp = np
        default_target_device_idx=-1


    if default_target_device_idx>=0:
        xp = cp
        gpu_float_dtype_list = [cp.float64, cp.float32]
        gpu_complex_dtype_list = [cp.complex128, cp.complex64]
        default_target_device = cp.cuda.Device(default_target_device_idx)
        default_target_device.use()
        print('Default device is GPU number ', default_target_device_idx)
        # print('Using device: ', cp.cuda.runtime.getDeviceProperties(default_target_device)['name'])
        # attributes = default_target_device.attributes
        # properties = cp.cuda.runtime.getDeviceProperties(default_target_device)
        # print('Number of multiprocessors:', attributes['MultiProcessorCount'])
        # print('Global memory size (GB):', properties['totalGlobalMem'] / (1024**3))
    else:
        print('Default device is CPU')
        xp = np

    float_dtype_list = [xp.float64, xp.float32]
    complex_dtype_list = [xp.complex128, xp.complex64]
    global_precision = precision
    float_dtype = float_dtype_list[global_precision]
    complex_dtype = complex_dtype_list[global_precision]
    
    # Patch cupy's missing RandomState.random() method
    if cp is not None:
        cp.random.RandomState.random = cp.random.RandomState.random_sample

# should be used as less as a possible and prefarably outside time critical computations
def cpuArray(v):
    if cp and isinstance(v, cp.ndarray):
        # which one is better, xp.asnumpy(v) or v.get() ? almost the same but asnumpy is more general
        return cp.asnumpy(v)
    else:
        if isinstance(v, np.ndarray):
            # Avoid reallocation of v if not needed
            return v
        else:
            return np.array(v)


def to_xp(xp, v, dtype=None):
    '''
    Make sure that v is allocated as an array on this object's device.
    Works for all combinations of np and cp, whether installed or not.
    Optionally casts to the required dtype (no copy is made if
    the dtype is already the correct one)

    The main trigger for this function is that np.array() cannot
    be used on a cupy array.
    '''
    if xp is cp:
        if isinstance(v, cp.ndarray):
            reval =  v
        else:
            reval =  cp.array(v)
    else:
        if cp is not None and isinstance(v, cp.ndarray):
            reval = v.get()
        elif isinstance(v, np.ndarray):
            # Avoid extra copy (enabled by numpy default)
            reval = v
        else:
            reval = np.array(v)
    if dtype is None:
        return reval
    else:
        return reval.astype(dtype, copy=False)


class DummyDecoratorAndContextManager():
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass
    def __call__(self, f):
        def caller(*args, **kwargs):
            return f(*args, **kwargs)
        return caller


def show_in_profiler(message=None, color_id=None, argb_color=None, sync=False):
    '''
    Decorator to allow using cupy's TimeRangeDecorator
    in a safe way even when cupy is not installed
    Parameters are the same as TimeRangeDecorator
    '''
    try:
        from cupyx.profiler import time_range

        return time_range(message=message,
                          color_id=color_id,
                          argb_color=argb_color,
                          sync=sync)

    except ImportError:
        return DummyDecoratorAndContextManager()


def fuse(kernel_name=None):
    '''
    Replacement of cupy.fuse() allowing runtime
    dispatch to cupy or numpy.
    
    Fused function takes an xp argument that will
    cause it to run as a fused kernel or a standard
    numpy function. The xp argument can be used
    inside the function as usual.

    Parameters are the same as cp.fuse()
    '''
    def decorator(f):
        f_cp = functools.partial(f, xp=cp)
        f_np = functools.partial(f, xp=np)
        f_cpu = f_np
        if cp:
            f_gpu = cp.fuse(kernel_name=kernel_name)(f_cp)
        else:
            f_gpu = None
        @wraps(f)
        def wrapper(*args, xp, **kwargs):
            if xp == cp:
                return f_gpu(*args, **kwargs)
            else:
                return f_cpu(*args, **kwargs)
        return wrapper
    return decorator
