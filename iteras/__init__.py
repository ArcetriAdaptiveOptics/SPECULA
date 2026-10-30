'''
iteras: a framework for end-to-end, time-domain simulations of systems made
of data objects and processing objects, running on CPU (numpy) or GPU (cupy),
optionally distributed over several processes with MPI.

Call :func:`init` before creating any object. The iteras modules read the
device, precision and MPI settings when objects are created, so they can be
imported before init(). Modules that copy these settings when they are imported
(e.g. with ``from iteras import cp``) must be imported after init(): use the
``xp`` and ``dtype`` attributes of the objects instead.
'''
import numpy as np
import logging
import traceback
import functools
from functools import wraps

from iteras import registry
from iteras.log import get_iteras_logger, init_logging

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

# precision = 0 -> double precision
# precision = 1 -> single precision

# target_device = -1 -> CPU
# target_device = i>-1 -> GPUi

# you might have a GPU working and cupy installed
# and still want to use the CPU (idx==-1) as default_target_device
# in this case you might still want to allocate some objects on
# a GPU device (idx>=0).
# This can be checked later looking at the value of gpuEnabled.


def init(device_idx=-1,
         precision=0,
         rank=None,
         comm=None,
         log_level=logging.INFO,
         ):
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

    process_comm = comm
    process_rank = rank

    init_logging(log_level=log_level, process_rank=rank)
    logger = get_iteras_logger(__name__)

    default_target_device_idx = device_idx
    systemDisable = registry.getenv('ITERAS_DISABLE_GPU', 'FALSE')
    if systemDisable == 'FALSE':
        try:
            import cupy as cp
            logger.info("Cupy import successfull. Installed version is: "+ cp.__version__)
            gpuEnabled = True
            cp = cp
        except:
            logger.warning("Cupy import failed. Falling back to CPU use.")
            cp = None
            xp = np
            default_target_device_idx = -1
    else:
        logger.info("GPU use disabled by environment variable (ITERAS_DISABLE_GPU or an alias).")
        cp = None
        xp = np
        default_target_device_idx = -1

    if default_target_device_idx >= 0:
        xp = cp
        gpu_float_dtype_list = [cp.float64, cp.float32]
        gpu_complex_dtype_list = [cp.complex128, cp.complex64]
        default_target_device = cp.cuda.Device(default_target_device_idx)
        default_target_device.use()
        logger.info(f'Default device is GPU number {default_target_device_idx}')
        # self.logger.debug('Using device: ', cp.cuda.runtime.getDeviceProperties(default_target_device)['name'])
        # attributes = default_target_device.attributes
        # properties = cp.cuda.runtime.getDeviceProperties(default_target_device)
        # self.logger.debug('Number of multiprocessors:', attributes['MultiProcessorCount'])
        # self.logger.debug('Global memory size (GB):', properties['totalGlobalMem'] / (1024**3))
    else:
        logger.info('Default device is CPU')
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
            retval = v
        else:
            retval = cp.array(v)
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


def capturing():
    '''True while the current CUDA stream is capturing a graph (see iteras.accelerator)'''
    return cp is not None and cp.cuda.get_current_stream().is_capturing()


def xp_fft(xp):
    '''
    FFT functions for the array module *xp*: scipy.fft for numpy (faster than
    numpy.fft, especially in single precision), xp.fft otherwise.
    '''
    if xp is np:
        import scipy.fft
        return scipy.fft
    return xp.fft


def fuse(kernel_name=None):
    '''
    Replacement of cupy.fuse() allowing runtime
    dispatch to cupy or numpy.

    Fused function takes an xp argument that will
    cause it to run as a fused kernel or a standard
    numpy function. The xp argument can be used
    inside the function as usual.

    Parameters are the same as cp.fuse()

    The fused kernel is created at the first call with xp=cupy,
    so that functions can be decorated before init() is called.
    '''
    def decorator(f):
        f_cpu = functools.partial(f, xp=np)
        f_gpu = None

        @wraps(f)
        def wrapper(*args, xp, **kwargs):
            nonlocal f_gpu
            if cp is not None and xp is cp:
                if f_gpu is None:
                    f_gpu = cp.fuse(kernel_name=kernel_name)(functools.partial(f, xp=cp))
                return f_gpu(*args, **kwargs)
            else:
                return f_cpu(*args, **kwargs)
        return wrapper
    return decorator


def main_simul(yml_files: list,
               nsimul=1,
               cpu: bool=False,
               overrides: str=None,
               target: int=0,
               precision: int=1,
               profile: bool=False,
               mpi: bool=False,
               stepping: bool=False,
               async_displays: bool=False,
               diagram: bool=False,
               diagram_title: str=None,
               diagram_filename: str=None,
               diagram_colors_on: bool=False,
               no_speed_report: bool=False,
               log_level: str='INFO',
               trace_file: str=None,
               trace_sync: bool=False,
               trace_skip: int=0,
               trace_gpu_events: bool=False,
               allocate: bool=False,
               reallocate: bool=False,
               accelerate: bool=False,
               ):

    # Set logging level for the "parent" loggers of iteras and its plugins
    for root in registry.plugin_roots():
        logging.getLogger(root).setLevel(log_level.upper())
    logger = logging.getLogger('iteras')

    if mpi:
        try:
            from mpi4py import MPI
            from mpi4py.util import pkl5
            logger.info(f"mpi4py import successfull. Installed version is: {MPI.Get_version()}")
        except ImportError:
            logger.error("mpi4py import failed.")
            raise

        comm = pkl5.Intracomm(MPI.COMM_WORLD)
        rank = comm.Get_rank()
        N = 10000000
        datatype = MPI.FLOAT
        num_bytes = N * (datatype.Pack_size(count=1, comm=comm) + MPI.BSEND_OVERHEAD)

        logger.debug(f'MPI buffer size: {num_bytes/1024**2:.2f} MB')
        attached_buf = bytearray(num_bytes)
        MPI.Attach_buffer(attached_buf)
    else:
        rank = None
        comm = None

    if cpu:
        target_device_idx = -1
    else:
        target_device_idx = target

    init(target_device_idx, precision=precision, rank=rank, comm=comm)
    from iteras.simul import Simul

    if profile:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()

    from iteras.tracing import tracer
    if trace_file:
        tracer.open(trace_file, sync=trace_sync, rank=rank, skip=trace_skip,
                    gpu_events=trace_gpu_events)
    elif trace_sync or trace_skip or trace_gpu_events:
        logger.warning('--trace-sync, --trace-skip and --trace-gpu-events '
                       'have no effect without --trace-file')

    try:
        for simul_idx in range(nsimul):
            logger.debug(f'{yml_files=}')
            Simul(*yml_files,
                simul_idx=simul_idx,
                overrides=overrides,
                stepping=stepping,
                async_displays=async_displays,
                diagram=diagram,
                diagram_filename=diagram_filename,
                diagram_title=diagram_title,
                diagram_colors_on=diagram_colors_on,
                speed_report=not no_speed_report,
                log_level=log_level,
                allocate=allocate,
                reallocate=reallocate,
                accelerate=accelerate,
            ).run()
    except Exception:
        if mpi:
            traceback.print_exc()
            # Force other ranks to abort as well,
            # otherwise they might remain stuck indefinitely
            # waiting for input from this rank
            MPI.COMM_WORLD.Abort(1)
        else:
            raise
    finally:
        # Print summary only if tracer was enabled
        summary = tracer.close()
        if summary:
            logger.info('Timing summary:\n' + summary)

    if profile:
        pr.disable()
        stats = pstats.Stats(pr).sort_stats("cumtime")
        stats.print_stats(r"\((?!\_).*\)$", 200)
        
    if mpi:
        MPI.Detach_buffer()


