import logging


def get_specula_logger(name):
    '''
    Replacement of logging.getLogger() that returns a SpeculaAdapter instead of a standard logger,
    so that we can use our custom log levels and formatting.
    '''
    orig_logger = logging.getLogger(name)
    return SpeculaAdapter(orig_logger)


def init_logging(log_format=None, log_level=logging.INFO, process_rank=None):
    '''
    Initialize logging with a custom format that includes the process rank if provided,
    and set up logging with our SpeculaFilter enabled.
    '''
    if log_format is None:
        if process_rank is None:
            log_format="%(asctime)s [%(levelname)s]: [%(name)s]: %(message)s"
        else:
            log_format="%(asctime)s [%(levelname)s]: [rank %(process_rank)s] [%(name)s]: %(message)s"

    logging.basicConfig(level=log_level, format=log_format)

    # Make sure all loggers use our filter
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(SpeculaFilter(process_rank))


class SpeculaFilter(logging.Filter):
    '''
    The logger name is usually the class name. This filter replaces it
    with the instance name if available, while for DEBUG or below, 
    both class and instance names are shown.
    Also add the process rank to the log record.
    '''
    def __init__(self, process_rank):
        super().__init__()
        self.process_rank = process_rank

    def filter(self, record):
        if hasattr(record, "instance_name") and record.instance_name:
            if record.levelno <= logging.DEBUG or record.instance_name in ['initialising']:
                record.name = f'{record.name} - {record.instance_name}'
            else:
                record.name = record.instance_name
        record.process_rank = self.process_rank
        return True


class SpeculaAdapter(logging.LoggerAdapter):
    '''
    Logger adapter that defines custom log levels for MPI debugging, below the standard DEBUG level (10):
    - MPI_DBG_LEVEL (6): General MPI debugging messages
    - MPI_SEND_DBG_LEVEL (5): Detailed messages for MPI send/receive operations
    Also manages the instance name for log records, allowing it to be included in the log output.
    '''
    def __init__(self, logger):
        super().__init__(logger, {})

    # Custom log levels for MPI debugging, below the standard DEBUG level (10)
    MPI_DBG_LEVEL = 6
    MPI_SEND_DBG_LEVEL = 5

    def mpi_debug(self, msg, *args, **kwargs):
        self.log(6, msg, *args, **kwargs)
    def mpi_send_debug(self, msg, *args, **kwargs):
        self.log(5, msg, *args, **kwargs)

    @property
    def level(self):
        return self.logger.level
    
    def set_instance_name(self, instance_name):
        '''
        Set the instance name for this logger
        '''
        self.extra['instance_name'] = instance_name

