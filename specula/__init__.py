import numpy as np
import os
import functools
from functools import wraps

cpu_float_dtype_list = [np.float64, np.float32]
cpu_complex_dtype_list = [np.complex128, np.complex64]
array_types = []

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
process_comm = None
process_rank = None
ASEC2RAD = np.pi / (3600 * 180)
RAD2ASEC = 1.0 / ASEC2RAD
MPI_DBG = False

MPI_SEND_DBG = False

# precision = 0 -> double precision
# precision = 1 -> single precision

# target_device = -1 -> CPU
# target_device = i>-1 -> GPUi

# you might have a GPU working and cupy installed
# and still want to use the CPU (idx==-1) as default_target_device
# in this case you might still want to allocate some objects on
# a GPU device (idx>=0).
# This can be checked later looking at the  value of gpuEnabled.

def init(device_idx=-1, precision=0, rank=None, comm=None, mpi_dbg=False):
    global xp
    global cp
    global gpuEnabled
    global global_precision
    global float_dtype_list
    global complex_dtype_list
    global gpu_float_dtype_list
    global gpu_complex_dtype_list
    global array_types
    global float_dtype
    global complex_dtype
    global default_target_device_idx
    global default_target_device
    global process_comm
    global process_rank
    global MPI_DBG
    
    MPI_DBG = mpi_dbg
    process_comm = comm
    process_rank = rank

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

    if cp is not None:
        array_types = [np.ndarray, cp.ndarray]
    else:
        array_types = [np.ndarray]

    float_dtype_list = [xp.float64, xp.float32]
    complex_dtype_list = [xp.complex128, xp.complex64]
    global_precision = precision
    float_dtype = float_dtype_list[global_precision]
    complex_dtype = complex_dtype_list[global_precision]
    
    # Patch cupy's missing RandomState.random() method
    if cp is not None:
        cp.random.RandomState.random = cp.random.RandomState.random_sample


# should be used as less as a possible and preferably outside time critical computations
def cpuArray(v, dtype=None, force_copy=False):
    return to_xp(np, v, dtype=dtype, force_copy=force_copy)


def to_xp(xp, v, dtype=None, force_copy=False):
    '''
    Make sure that v is allocated as an array on this object's device.
    Works for all combinations of np and cp, whether installed or not.

    Optionally casts to the required dtype (no copy is made if
    the dtype is already the correct one)

    The main trigger for this function is that np.array() cannot
    be used on a cupy array.
    '''
    if xp is cp:
        if isinstance(v, cp.ndarray) and not force_copy:
            retval =  v
        else:
            retval =  cp.array(v)
    else:
        if cp is not None and isinstance(v, cp.ndarray):
            retval = v.get()
        elif isinstance(v, np.ndarray) and not force_copy:
            # Avoid extra copy (enabled by numpy default)
            retval = v
        else:
            retval = np.array(v)
    if dtype is None and not force_copy:
        return retval
    else:
        return retval.astype(dtype, copy=force_copy)


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


def main_simul(yml_files: list,
               nsimul = 1,
               cpu: bool=False,
               overrides: str=None,
               target: int=0,
               profile: bool=False,
               mpi: bool=False,
               mpidbg: bool=False,
               diagram: bool=False,
               diagram_title: str=None,
               diagram_filename: str=None):

    if mpi:
        try:
            from mpi4py import MPI
            from mpi4py.util import pkl5
            print("mpi4py import successfull. Installed version is:", MPI.Get_version())
        except ImportError:
            print("mpi4py import failed.")
            raise

        comm = pkl5.Intracomm(MPI.COMM_WORLD)
        rank = comm.Get_rank()
        N = 10000000
        datatype = MPI.FLOAT
        num_bytes = N * (datatype.Pack_size(count=1, comm=comm) + MPI.BSEND_OVERHEAD)

        print(f'MPI buffer size: {num_bytes/1024**2:.2f} MB')
        attached_buf = bytearray(num_bytes)
        MPI.Attach_buffer(attached_buf)
    else:
        rank = None
        comm = None
        
    if cpu:
        target_device_idx = -1
    else:
        target_device_idx = target

    init(target_device_idx, precision=1, rank=rank, comm=comm, mpi_dbg=mpidbg)
    from specula.simul import Simul

    if profile:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()

    for simul_idx in range(nsimul):
        print(yml_files)
        Simul(*yml_files,
            simul_idx=simul_idx,
            overrides=overrides,
            diagram=diagram,
            diagram_filename=diagram_filename,
            diagram_title=diagram_title,
        ).run()

    if profile:
        pr.disable
        stats = pstats.Stats(pr).sort_stats("cumtime")
        stats.print_stats(r"\((?!\_).*\)$", 200)
        
    if mpi:
        MPI.Detach_buffer()


