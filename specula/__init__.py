'''
SPECULA: adaptive optics simulations built on the iteras simulation framework.

The framework (device and precision setup, base classes, connections,
loop control, MPI support, ...) lives in the iteras package. This module
registers SPECULA as an iteras plugin and keeps the historical names,
so that ``specula.init()``, ``from specula import cpuArray, cp, xp`` and
``specula.global_precision`` work as before.
'''
import numpy as np

import iteras
from iteras import registry as _registry
from iteras import (init, main_simul, cpuArray, to_xp, fuse,
                   cpu_float_dtype_list, cpu_complex_dtype_list)
from specula.calib_subdirs import CALIB_SUBDIRS as _CALIB_SUBDIRS

ASEC2RAD = np.pi / (3600 * 180)
RAD2ASEC = 1.0 / ASEC2RAD

_registry.register_plugin(
    'specula',
    packages=['specula.processing_objects',
              'specula.data_objects',
              'specula.display'],
    calib_subdirs=_CALIB_SUBDIRS,
    env_aliases={'ITERAS_DISABLE_GPU': 'SPECULA_DISABLE_GPU',
                 'ITERAS_PORT': 'SPECULA_PORT'},
)

# Values set by init(): they are read from iteras when accessed,
# so that they are always up to date.
_ITERAS_RUNTIME_NAMES = frozenset([
    'gpuEnabled',
    'cp',
    'xp',
    'global_precision',
    'float_dtype_list',
    'complex_dtype_list',
    'gpu_float_dtype_list',
    'gpu_complex_dtype_list',
    'array_types',
    'float_dtype',
    'complex_dtype',
    'default_target_device_idx',
    'default_target_device',
    'process_comm',
    'process_rank',
])


def __getattr__(name):
    if name in _ITERAS_RUNTIME_NAMES:
        return getattr(iteras, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
