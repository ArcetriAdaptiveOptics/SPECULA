import os

# Limit BLAS/OpenMP threads before numpy is imported by any test module.
# Test matrices are small, and with the default (one thread per core) the
# LAPACK calls are dominated by threading overhead, especially on a loaded machine.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')
