# Logging moved to iteras.log. This module keeps the historical SPECULA names.
from iteras.log import *  # noqa: F401,F403
from iteras.log import (INIT_PLACEHOLDER_NAME, MPI_DBG_LEVEL, MPI_SEND_DBG_LEVEL,
                       init_logging, reset_logging, get_iteras_logger,
                       IterasLogAdapter, IterasLogFilter, IterasLogFormatter)

get_specula_logger = get_iteras_logger
SpeculaLogAdapter = IterasLogAdapter
SpeculaLogFilter = IterasLogFilter
SpeculaLogFormatter = IterasLogFormatter
