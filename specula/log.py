import sys
import logging

# String to show in the logs in place
# of an instance name  while the object
# is initialising and the name is not know yet

INIT_PLACEHOLDER_NAME = 'initialising'

MPI_DBG_LEVEL = 6
MPI_SEND_DBG_LEVEL = 5

logging.addLevelName(MPI_DBG_LEVEL, 'MPI_DBG')
logging.addLevelName(MPI_SEND_DBG_LEVEL, 'MPI_SEND_DBG')


def get_specula_logger(name):
    '''
    Replacement of logging.getLogger() that returns a SpeculaLogAdapter instead of a standard logger,
    so that we can use our custom log levels and formatting.
    '''
    orig_logger = logging.getLogger(name)
    return SpeculaLogAdapter(orig_logger)


def init_logging(log_level=logging.INFO, process_rank=None):
    '''
    Initialize logging with a custom format that includes the process rank if provided,
    and set up logging with our SpeculaLogFilter enabled.
    '''

    formatter = SpeculaLogFormatter(
        fmt_with_rank="%(asctime)s [%(levelname)s]: [rank %(process_rank)s] [%(display_name)s]: %(message)s",
        fmt_without_rank="%(asctime)s [%(levelname)s]: [%(display_name)s]: %(message)s",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(handler)

    # Make sure all loggers use our filter
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(SpeculaLogFilter(process_rank))


class SpeculaLogFilter(logging.Filter):
    '''
    Add the process rank to the log record, if defined.
    '''
    def __init__(self, process_rank):
        super().__init__()
        self.process_rank = process_rank

    def filter(self, record):
        if self.process_rank is not None:
            record.process_rank = self.process_rank
        return True


class SpeculaLogAdapter(logging.LoggerAdapter):
    '''
    Logger adapter that defines custom log levels for MPI debugging, below the standard DEBUG level (10):
    - MPI_DBG_LEVEL (6): General MPI debugging messages
    - MPI_SEND_DBG_LEVEL (5): Detailed messages for MPI send/receive operations
    Also manages the instance name for log records, allowing it to be included in the log output.
    '''
    def __init__(self, logger):
        super().__init__(logger, {})

    def mpi_debug(self, msg, *args, **kwargs):
        self.log(MPI_DBG_LEVEL, msg, *args, **kwargs)
    def mpi_send_debug(self, msg, *args, **kwargs):
        self.log(MPI_SEND_DBG_LEVEL, msg, *args, **kwargs)

    @property
    def level(self):
        return self.logger.level
    
    def set_instance_name(self, instance_name):
        '''
        Set the instance name for this logger
        '''
        self.extra['instance_name'] = instance_name


class SpeculaLogFormatter():
    '''
    Dispatcher class sending logs to one of two different formatters
    depending on whether the process rank is present or not.
    '''
    def __init__(self, fmt_with_rank, fmt_without_rank, *args, **kwargs):
        self.fmt_with_rank = logging.Formatter(fmt_with_rank, *args, **kwargs)
        self.fmt_without_rank = logging.Formatter(fmt_without_rank, *args, **kwargs)

    def format(self, record):
        instance_name = getattr(record, "instance_name", None)

        # Show both name and instance_name if level is DEBUG or less,
        # otherwise the instance_name only if available,
        # or the name (which will be the class name) as a last resort.
        if instance_name:
            if record.levelno <= logging.DEBUG:
                record.display_name = f"{record.name} ({instance_name})"
            else:
                record.display_name = instance_name
        else:
            record.display_name = record.name

        # Choose formatter based on process_rank
        if hasattr(record, "process_rank"):
            formatter = self.fmt_with_rank
        else:
            formatter = self.fmt_without_rank

        return formatter.format(record)

