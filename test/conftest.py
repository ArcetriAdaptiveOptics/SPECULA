import os

# Limit BLAS/OpenMP threads before numpy is imported by any test module.
# Test matrices are small, and with the default (one thread per core) the
# LAPACK calls are dominated by threading overhead, especially on a loaded machine.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')


import pytest

# Values set by init() (iteras.init(), also available as specula.init()).
# The framework reads them when objects are created.
_INIT_STATE = ['xp', 'cp', 'gpuEnabled', 'global_precision', 'float_dtype_list',
               'complex_dtype_list', 'gpu_float_dtype_list', 'gpu_complex_dtype_list',
               'array_types', 'float_dtype', 'complex_dtype', 'default_target_device_idx',
               'default_target_device', 'process_comm', 'process_rank']


@pytest.fixture(autouse=True)
def _restore_init_state():
    '''
    Restore the values set by init() after each test, so that a test calling
    init(), directly or through main_simul() (which uses single precision by
    default), does not change the device and precision of the objects created
    by the following tests.
    '''
    import iteras
    saved = {name: getattr(iteras, name) for name in _INIT_STATE}
    yield
    for name, value in saved.items():
        setattr(iteras, name, value)
